import copy
import os

import numpy as np
import torch
from torch import nn, einsum, Tensor
from torch.nn import Module
import torch.nn.functional as F
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from einops import rearrange, repeat
from utils import count_params, count_trainable_params

from perfusion_pytorch.ldm.models.diffusion.ddpm import LatentDiffusion
from perfusion_pytorch.ldm.util import log_txt_as_img, instantiate_from_config, default
from perfusion_pytorch.embedding import OpenClipEmbedWrapper
from perfusion_pytorch import save, save_load
from perfusion_pytorch.ldm.models.diffusion.ddim import DDIMSampler
from PIL import Image


def roe_state_dict(model: torch.nn.Module):
    sd = model.state_dict()
    to_return = {}
    for k in sd:
        if 'target_output' in k or 'target_input' in k:
            to_return[k] = sd[k]
    return to_return

def set_submodule(module, submodule_name, new_submodule):
    submodule_names = submodule_name.split('.')
    current_module = module
    for name in submodule_names[:-1]:
        current_module = getattr(current_module, name)
    setattr(current_module, submodule_names[-1], new_submodule)


class Perfusion(LatentDiffusion):
    def __init__(
            self,
            superclass_string='teddy',
            ema_p=0.99,
            beta=0.75,
            tau=0.1,
            *args, **kwargs):
        """
        Args:
            C_inv_path: path to the inverse of the uncentered covariance metric.
            ema_p: p for calculating exponential moving average on target input.
            beta: bias used in gated rank-1 editing.
            tau: temperature used in gated rank-1 editing.
        """
        super().__init__(*args, **kwargs)
        self.ema_p = ema_p
        self.beta = torch.tensor(beta)
        self.tau = torch.tensor(tau)
        self.device_ = torch.device('cuda')

        self.wrapped_clip_with_new_concept = OpenClipEmbedWrapper(
            transformer=self.cond_stage_model.transformer,
            tokenizer=self.cond_stage_model.tokenizer,
            superclass_string=superclass_string
        )

    def run(self, batch, torch_dtype):
        x = batch['images']
        captions = batch['captions']

        text_enc, superclass_enc, mask, indices = self.wrapped_clip_with_new_concept(captions)

        x = rearrange(x, 'b h w c -> b c h w')
        x = x.to(memory_format=torch.contiguous_format).float()

        x = x.to(self.device_)
        # x = x.to(torch.float16)
        encoder_posterior = self.encode_first_stage(x)
        x_start = self.get_first_stage_encoding(encoder_posterior).detach()

        t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=self.device).long()
        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        # x_noisy = x_noisy.to(torch.float16)
        output = self.model.diffusion_model(x_noisy, t, context=text_enc, mask=mask, concept_indices=indices, text_enc_with_superclass=superclass_enc)

        loss_simple = self.get_loss(output.float(), noise.float(), mean=False).mean([1, 2, 3])
        loss = loss_simple.mean()
        return loss

    def predict(self, model, prompt, num_imgs=1):
        model.eval()
        start_code = torch.randn([1, 4, 512 // 8, 512 // 8], device=torch.device('cuda'))
        promptss = [[prompt]]

        sampler = DDIMSampler(model)
        images = []

        with torch.no_grad():

            all_samples = list()
            for n in range(num_imgs):
                for prompt in promptss:
                    uc = model.get_learned_conditioning([""])

                    prompts = list(prompt)
                    c = model.get_learned_conditioning(prompts)
                    shape = [4, 512 // 8, 512 // 8]
                    samples_ddim, _ = sampler.sample(S=50,
                                                     conditioning=c,
                                                     batch_size=1,
                                                     shape=shape,
                                                     verbose=False,
                                                     unconditional_guidance_scale=7.5,
                                                     unconditional_conditioning=uc,
                                                     eta=0.0,
                                                     x_T=start_code)

                    x_samples_ddim = model.decode_first_stage(samples_ddim)
                    x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                    x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()

                    x_checked_image_torch = torch.from_numpy(x_samples_ddim).permute(0, 3, 1, 2)


                    for x_sample in x_checked_image_torch:
                        x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                        img = Image.fromarray(x_sample.astype(np.uint8))
                        images.append(img)
            model.train()
            return images




    def init_from_personalized_ckpt(self, path):
        sd = torch.load(path, map_location="cpu")
        self.C_inv.data = sd['C_inv']
        self.target_input.data = sd['target_input']
        self.embedding_manager.load_state_dict(sd['embedding'])
        self.model.diffusion_model.load_state_dict(sd['target_output'], strict=False)


def exists(val):
    return val is not None

if __name__ == '__main__':
    from omegaconf import OmegaConf
    from perfusion_pytorch.embedding import EmbeddingWrapper
    from ldm.modules.diffusionmodules.rankoneediting import Rank1EditModule
    cfg = r"D:\\PycharmProjects\\perfusion\\configs\\perfusion_teddy.yaml"
    config = OmegaConf.load(cfg)
    text_image_model = instantiate_from_config(config.model)

    for param in text_image_model.parameters():
        param.requires_grad = False
    count_params(text_image_model, verbose=True)

    embed_params = None
    key_value_params = []
    for module in text_image_model.modules():
        if isinstance(module, EmbeddingWrapper):
            assert not exists(embed_params), 'there should only be one wrapped EmbeddingWrapper'
            module.concepts.requires_grad = True
            embed_params = module.concepts.data

        elif isinstance(module, Rank1EditModule):
            # module.ema_concept_text_encs.requires_grad = True
            if not module.is_key_proj:
                module.concept_outputs.requires_grad = True
                key_value_params.append([
                    # module.ema_concept_text_encs.data,
                    module.concept_outputs.data
                ])
            # else:
            #     key_value_params.append([
            #         module.ema_concept_text_encs.data])

    count_trainable_params(text_image_model, verbose=True)
    pkg = dict(
        embed_params = embed_params,
        key_value_params = key_value_params,
    )

    torch.save(pkg, 'tmp.pth')
