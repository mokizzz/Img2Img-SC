import argparse
import os
import time
from itertools import islice

import lpips as lp
import numpy as np
import PIL
import torch
import torch.nn as nn
from einops import rearrange, repeat
from imwatermark import WatermarkEncoder
from omegaconf import OmegaConf
from PIL import Image
from pytorch_lightning import seed_everything
from SSIM_PIL import compare_ssim
from torch import autocast, optim
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm, trange
from transformers import pipeline

from ldm.data.dataset import get_test_dataset
from ldm.data.flickr8k import get_datasets
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.util import instantiate_from_config
from scripts.clip_ci import load_ci_model
from scripts.dataset import Flickr8kDataset, Only_images_Flickr8kDataset
from scripts.metrics_moki import calculate_all_metrics, calculate_fid
from scripts.qam import qam16ModulationString, qam16ModulationTensor

"""

INIT DATASET AND DATALOADER

"""
batch_size = 1

test = get_test_dataset(None, name="urban100")

test_dataloader = DataLoader(dataset=test, batch_size=batch_size, shuffle=False)


"""
MODEL CHECKPOINT

"""


model_ckpt_path = "/home/zhaozr/v1-5-pruned.ckpt"  # "G:/Giordano/stablediffusion/checkpoints/v1-5-pruned.ckpt"             #v2-1_512-ema-pruned.ckpt"
config_path = "configs/stable-diffusion/v1-inference.yaml"  # "G:/Giordano/stablediffusion/configs/stable-diffusion/v1-inference.yaml"


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model


def load_img(path):
    image = Image.open(path).convert("RGB")
    w, h = image.size
    # w, h = (512, 512)  # image.size
    # print(f"loaded input image of size ({w}, {h}) from {path}")
    w, h = map(lambda x: x - x % 64, (w, h))  # resize to integer multiple of 64
    image = image.resize((w, h), resample=PIL.Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.0 * image - 1.0


def test(
    dataloader,
    snr=10,
    num_images=100,
    batch_size=1,
    num_images_per_sample=2,
    outpath="",
    model=None,
    device=None,
    sampler=None,
    strength=0.8,
    ddim_steps=50,
    scale=9.0,
):

    # blip = pipeline("image-to-text", model="Salesforce/blip-image-captioning-large")
    ci = load_ci_model()
    i = 0

    sampling_steps = int(strength * 50)
    print(sampling_steps)
    sampler.make_schedule(
        ddim_num_steps=50, ddim_eta=0.0, verbose=False
    )  # attenzione ai parametri
    sample_path = os.path.join(outpath, f"Test-samples-{snr}-{sampling_steps}")
    os.makedirs(sample_path, exist_ok=True)

    text_path = os.path.join(outpath, f"Test-text-samples-{snr}-{sampling_steps}")
    os.makedirs(text_path, exist_ok=True)

    sample_orig_path = os.path.join(
        outpath, f"Test-samples-orig-{snr}-{sampling_steps}"
    )
    os.makedirs(sample_orig_path, exist_ok=True)

    lpips = lp.LPIPS(net="alex")
    lpips_values = []
    ssim_values = []
    time_values = []
    all_metrics = []
    all_origi_images = []
    all_recon_images = []

    tq = tqdm(dataloader, total=num_images)
    for data in tq:

        if "image_path" in data.keys():
            img_file_path = data["image_path"][0]
        elif "hr" in data.keys():
            img_file_path = data["hr"][0]
        else:
            raise ValueError("Image path not found in data")

        # Open Image
        init_image = Image.open(img_file_path)

        # Automatically extract caption using BLIP model
        prompt = ci.interrogate_fast(init_image)
        prompt_original = prompt

        base_count = len(os.listdir(sample_path))

        assert os.path.isfile(img_file_path)
        init_image = load_img(img_file_path).to(device)
        init_image = repeat(init_image, "1 ... -> b ...", b=batch_size)
        init_latent = model.get_first_stage_encoding(
            model.encode_first_stage(init_image)
        )  # move to latent space

        # print(init_latent.shape,init_latent.type())

        """
        CHANNEL SIMULATION
        """

        init_latent = qam16ModulationTensor(init_latent.cpu(), snr_db=snr).to(device)

        prompt = qam16ModulationString(prompt, snr_db=snr)  # NOISY BLIP PROMPT

        data = [batch_size * [prompt]]
        assert 0.0 <= strength <= 1.0, "can only work with strength in [0.0, 1.0]"
        t_enc = int(strength * ddim_steps)

        precision_scope = autocast
        with torch.no_grad():
            with precision_scope("cuda"):
                with model.ema_scope():
                    all_samples = list()
                    for n in range(1):
                        for prompts in data:
                            start_time = time.time()
                            uc = None
                            if scale != 1.0:
                                uc = model.get_learned_conditioning(batch_size * [""])
                            if isinstance(prompts, tuple):
                                prompts = list(prompts)
                            c = model.get_learned_conditioning(prompts)
                            # encode (scaled latent)
                            z_enc = sampler.stochastic_encode(
                                init_latent,
                                torch.tensor([t_enc] * batch_size).to(device),
                            )
                            # z_enc = init_latent
                            # decode it
                            samples = sampler.decode(
                                z_enc,
                                c,
                                t_enc,
                                unconditional_guidance_scale=scale,
                                unconditional_conditioning=uc,
                            )
                            x_samples = model.decode_first_stage(samples)

                            x_samples = torch.clamp(
                                (x_samples + 1.0) / 2.0, min=0.0, max=1.0
                            )

                            end_time = time.time()
                            execution_time = end_time - start_time

                            time_values.append(execution_time)

                            for x_sample in x_samples:
                                x_sample = 255.0 * rearrange(
                                    x_sample.cpu().numpy(), "c h w -> h w c"
                                )
                                img = Image.fromarray(x_sample.astype(np.uint8))
                                # SAVE TEXT
                                f = open(
                                    os.path.join(text_path, f"{base_count:05}.txt"), "a"
                                )
                                f.write(prompt_original)
                                f.close()

                                # SAVE ORIGINAL IMAGE
                                init_image_copy = Image.open(img_file_path)
                                # init_image_copy = init_image_copy.resize(
                                #     (512, 512), resample=PIL.Image.LANCZOS
                                # )
                                init_image_copy.save(
                                    os.path.join(
                                        sample_orig_path, f"{base_count:05}.png"
                                    )
                                )

                                # Compute SSIM
                                img = img.resize(init_image_copy.size)
                                # SAVE IMAGE
                                img.save(
                                    os.path.join(sample_path, f"{base_count:05}.png")
                                )
                                # ssim_values.append(compare_ssim(init_image_copy, img))
                                all_origi_images.append(init_image_copy.convert("RGB"))
                                all_recon_images.append(img.convert("RGB"))

                                metrics = calculate_all_metrics(
                                    PIL.Image.open(
                                        os.path.join(
                                            sample_orig_path, f"{base_count:05}.png"
                                        )
                                    ).convert("RGB"),
                                    PIL.Image.open(
                                        os.path.join(
                                            sample_path, f"{base_count:05}.png"
                                        )
                                    ).convert("RGB"),
                                )
                                all_metrics.append(metrics)

                                base_count += 1
                            all_samples.append(x_samples)

                    # Compute LPIPS
                    sample_out = (all_samples[0][0] * 2) - 1
                    lp_score = lpips(init_image[0].cpu(), sample_out.cpu()).item()

                    tq.set_postfix(
                        lpips=lp_score, lpips_2=metrics["lpips"], ssim=metrics["ssim"]
                    )

                    if not np.isnan(lp_score):
                        lpips_values.append(lp_score)

        i += 1
        if i == num_images:
            break

    print(f"mean lpips score at snr={snr} : {sum(lpips_values)/len(lpips_values)}")
    # print(f"mean ssim score at snr={snr} : {sum(ssim_values)/len(ssim_values)}")
    print(
        f"mean time with sampling iterations {sampling_steps} : {sum(time_values)/len(time_values)}"
    )

    # 7. Calculate and log average metrics
    avg_metrics = {
        k: sum(m[k] for m in all_metrics) / len(all_metrics) for k in all_metrics[0]
    }
    avg_metrics["fid"] = calculate_fid(all_origi_images, all_recon_images)
    for metric_name, value in avg_metrics.items():
        print(f"---- Average {metric_name}: {value:.6g}")
    print(
        "Results: "
        + "|".join(
            [
                f"{avg_metrics[k]:.6g}"
                for k in ["psnr", "ssim", "ms_ssim", "lpips", "fid"]
            ]
        )
    )

    return 1


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/img2img-samples",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )

    opt = parser.parse_args()
    seed_everything(opt.seed)

    config = OmegaConf.load(f"{config_path}")
    model = load_model_from_config(config, f"{model_ckpt_path}")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    sampler = DDIMSampler(model)

    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir

    # INIZIO TEST

    # Strength is used to modulate the number of sampling steps. Steps=50*strength
    # for snr in [10, 8.75, 7.5, 6.25, 5]:
    #     test(
    #         test_dataloader,
    #         snr=snr,
    #         num_images=100,
    #         batch_size=1,
    #         num_images_per_sample=1,
    #         outpath=outpath,
    #         model=model,
    #         device=device,
    #         sampler=sampler,
    #         strength=0.6,
    #         scale=9,
    #     )

    # SNR=100
    test(
        test_dataloader,
        snr=100,
        num_images=100,
        batch_size=1,
        num_images_per_sample=1,
        outpath=outpath,
        model=model,
        device=device,
        sampler=sampler,
        strength=0.6,
        scale=9,
    )
