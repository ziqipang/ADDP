import os
import sys
import time
import einops
import argparse
from PIL import Image
from einops import rearrange
from omegaconf import OmegaConf

import torch
import torch.nn as nn
from torch import autocast
from torch.utils.data import DataLoader

import k_diffusion as K
sys.path.append("./stable_diffusion")
from ldm.util import instantiate_from_config

from refseg_dataset import RefSegDataset


class CompVisDenoiser(K.external.DiscreteEpsDDPMDenoiser):
    """A wrapper for CompVis diffusion models."""
    def __init__(self, model, quantize=False, parameterization="eps", device="cpu"):
        super().__init__(model, model.alphas_cumprod, quantize=quantize)
        self.parameterization = parameterization

    def get_eps(self, *args, **kwargs):
        return self.inner_model.apply_model(*args, **kwargs)

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return self.inner_model._predict_eps_from_xstart(x_t, t, pred_xstart)

    def forward(self, input, sigma, **kwargs):
        c_out, c_in = [K.utils.append_dims(x, input.ndim) for x in self.get_scalings(sigma)]
        output = self.get_eps(input * c_in, self.sigma_to_t(sigma), **kwargs)
        if self.parameterization == "eps":
            eps = output
        elif self.parameterization == "x0":
            eps = self._predict_eps_from_xstart(input * c_in, self.sigma_to_t(sigma).long(), output)
        return input + eps * c_out


class CFGDenoiser(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.inner_model = model

    def forward(self, z, sigma, cond, uncond, text_cfg_scale, image_cfg_scale):
        cfg_z = z.unsqueeze(1).repeat(1, 3, 1, 1, 1).flatten(0, 1)
        cfg_sigma = sigma.unsqueeze(1).repeat(1, 3).flatten(0, 1)

        c_crossattn = torch.cat((
            cond["c_crossattn"].unsqueeze(1),
            uncond["c_crossattn"].unsqueeze(1),
            uncond["c_crossattn"].unsqueeze(1),
        ), dim=1).flatten(0, 1)
        c_concat = torch.cat((
            cond["c_concat"].unsqueeze(1),
            cond["c_concat"].unsqueeze(1),
            uncond["c_concat"].unsqueeze(1),
        ), dim=1).flatten(0, 1)
        cfg_cond = {
            "c_crossattn": [c_crossattn],
            "c_concat": [c_concat],
        }

        output = self.inner_model(cfg_z, cfg_sigma, cond=cfg_cond)
        out_cond, out_img_cond, out_uncond = output[::3], output[1:][::3], output[2:][::3]
        return out_uncond + text_cfg_scale * (out_cond - out_img_cond) + image_cfg_scale * (out_img_cond - out_uncond)

    def save_img(self, z, save_path):
        x = self.inner_model.inner_model.decode_first_stage(z)
        x = torch.clamp((x + 1.0) / 2.0, min=0.0, max=1.0)
        x = 255.0 * rearrange(x, "l c h w -> l h w c")
        edited_image = x.type(torch.uint8).cpu().numpy()
        im = Image.fromarray(edited_image[0])
        im.save(save_path)


def load_model_from_config(config, ckpt, vae_ckpt=None, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    if vae_ckpt is not None:
        print(f"Loading VAE from {vae_ckpt}")
        vae_sd = torch.load(vae_ckpt, map_location="cpu")["state_dict"]
        sd = {
            k: vae_sd[k[len("first_stage_model.") :]] if k.startswith("first_stage_model.") else v
            for k, v in sd.items()
        }
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=str, required=True)
    parser.add_argument("--dataset-name", type=str, default="refcoco")
    parser.add_argument("--split", type=str, default="val_flatten")
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--model-ckpt", type=str, required=True)
    parser.add_argument("--vae-ckpt", type=str, default=None)
    parser.add_argument("--parameterization", type=str, choices=["x0", "eps"], default="x0")
    parser.add_argument("--save-dir", type=str, default="/tmp")
    parser.add_argument("--save-name", type=str, default="batch")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--cfg-text", type=float, default=3.0)
    parser.add_argument("--cfg-image", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.parameterization == "x0":
        args.config = "configs/generate_x0.yaml"
    elif args.parameterization == "eps":
        args.config = "configs/generate.yaml"

    config = OmegaConf.load(args.config)
    model = load_model_from_config(config, args.model_ckpt, args.vae_ckpt)
    model.eval().cuda()
    model_wrap = CompVisDenoiser(model, parameterization=args.parameterization)
    model_wrap_cfg = CFGDenoiser(model_wrap)
    null_token = model.get_learned_conditioning([""])

    dataset = RefSegDataset(args.dataset_root, args.dataset_name, args.split, args.resolution)
    data_loader = DataLoader(dataset, batch_size=32, shuffle=False, drop_last=False)

    save_dir = os.path.join(args.save_dir, args.save_name)
    os.makedirs(save_dir, exist_ok=True)

    cnt = 0
    with torch.no_grad(), autocast("cuda"), model.ema_scope():
        for index, data in enumerate(data_loader):
            print(f"Processing: {index + 1}/{len(data_loader)}")

            start_time = time.time()
            prompts = data["edit"]["c_crossattn"]
            prompt_num = len(prompts)

            cond = {}
            cond["c_crossattn"] = [model.get_learned_conditioning([prompt]) for prompt in prompts]
            cond["c_crossattn"] = torch.cat(cond["c_crossattn"], dim=0)  # [32 * 77 * 768]
            input_image = data["edit"]["c_concat"].to(model.device)
            encode_first_stage_result = model.encode_first_stage(input_image).mode()
            cond["c_concat"] = encode_first_stage_result  # [32 * 4 * 32 * 32]

            uncond = {}
            uncond["c_crossattn"] = einops.repeat(null_token, "1 ... -> n ...", n=prompt_num)  # [32 * 77 * 768]
            uncond["c_concat"] = torch.zeros_like(cond["c_concat"])  # [32 * 4 * 32 * 32]

            sigmas = model_wrap.get_sigmas(args.steps)
            extra_args = {
                "cond": cond,
                "uncond": uncond,
                "text_cfg_scale": args.cfg_text,
                "image_cfg_scale": args.cfg_image,
            }

            torch.manual_seed(0)
            z = torch.randn_like(cond["c_concat"]) * sigmas[0]  # [32 * 4 * 32 * 32]
            z = K.sampling.sample_euler_ancestral(model_wrap_cfg, z, sigmas, extra_args=extra_args)

            x = model.decode_first_stage(z)
            x = torch.clamp((x + 1.0) / 2.0, min=0.0, max=1.0)
            x = 255.0 * rearrange(x, "l c h w -> l h w c")
            edited_image = x.type(torch.uint8).cpu().numpy()

            net_time = time.time()

            for i in range(edited_image.shape[0]):
                im = Image.fromarray(edited_image[i])
                im.save(os.path.join(save_dir, f"{cnt}.png"))
                cnt += 1

            save_time = time.time()
            print("net time: {:}, save time: {:}".format(net_time - start_time, save_time - net_time))


if __name__ == "__main__":
    main()
