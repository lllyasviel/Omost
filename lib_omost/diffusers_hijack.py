import numpy as np
import copy

from tqdm.auto import trange
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl_img2img import *
from diffusers.models.transformers import Transformer2DModel


original_Transformer2DModel_forward = Transformer2DModel.forward


def hacked_Transformer2DModel_forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        timestep: Optional[torch.LongTensor] = None,
        added_cond_kwargs: Dict[str, torch.Tensor] = None,
        class_labels: Optional[torch.LongTensor] = None,
        cross_attention_kwargs: Dict[str, Any] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
):
    cross_attention_kwargs = cross_attention_kwargs or {}
    cross_attention_kwargs['hidden_states_original_shape'] = hidden_states.shape
    return original_Transformer2DModel_forward(
        self, hidden_states, encoder_hidden_states, timestep, added_cond_kwargs, class_labels, cross_attention_kwargs,
        attention_mask, encoder_attention_mask, return_dict)


Transformer2DModel.forward = hacked_Transformer2DModel_forward


@torch.no_grad()
def sample_dpmpp_2m(model, x, sigmas, extra_args=None, callback=None, disable=None):
    """DPM-Solver++(2M)."""
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    sigma_fn = lambda t: t.neg().exp()
    t_fn = lambda sigma: sigma.log().neg()
    old_denoised = None

    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        t, t_next = t_fn(sigmas[i]), t_fn(sigmas[i + 1])
        h = t_next - t
        if old_denoised is None or sigmas[i + 1] == 0:
            x = (sigma_fn(t_next) / sigma_fn(t)) * x - (-h).expm1() * denoised
        else:
            h_last = t - t_fn(sigmas[i - 1])
            r = h_last / h
            denoised_d = (1 + 1 / (2 * r)) * denoised - (1 / (2 * r)) * old_denoised
            x = (sigma_fn(t_next) / sigma_fn(t)) * x - (-h).expm1() * denoised_d
        old_denoised = denoised
    return x


class KModel:
    def __init__(self, unet, timesteps=1000, linear_start=0.00085, linear_end=0.012):
        betas = torch.linspace(linear_start ** 0.5, linear_end ** 0.5, timesteps, dtype=torch.float64) ** 2
        alphas = 1. - betas
        alphas_cumprod = torch.tensor(np.cumprod(alphas, axis=0), dtype=torch.float32)

        self.sigmas = ((1 - alphas_cumprod) / alphas_cumprod) ** 0.5
        self.log_sigmas = self.sigmas.log()
        self.sigma_data = 1.0
        self.unet = unet
        return

    @property
    def sigma_min(self):
        return self.sigmas[0]

    @property
    def sigma_max(self):
        return self.sigmas[-1]

    def timestep(self, sigma):
        log_sigma = sigma.log()
        dists = log_sigma.to(self.log_sigmas.device) - self.log_sigmas[:, None]
        return dists.abs().argmin(dim=0).view(sigma.shape).to(sigma.device)

    def get_sigmas_karras(self, n, rho=7.):
        ramp = torch.linspace(0, 1, n)
        min_inv_rho = self.sigma_min ** (1 / rho)
        max_inv_rho = self.sigma_max ** (1 / rho)
        sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
        return torch.cat([sigmas, sigmas.new_zeros([1])])

    def __call__(self, x, sigma, **extra_args):
        x_ddim_space = x / (sigma[:, None, None, None] ** 2 + self.sigma_data ** 2) ** 0.5
        t = self.timestep(sigma)
        cfg_scale = extra_args['cfg_scale']
        eps_positive = self.unet(x_ddim_space, t, return_dict=False, **extra_args['positive'])[0]
        eps_negative = self.unet(x_ddim_space, t, return_dict=False, **extra_args['negative'])[0]
        noise_pred = eps_negative + cfg_scale * (eps_positive - eps_negative)
        return x - noise_pred * sigma[:, None, None, None]


class OmostSelfAttnProcessor:
    def __call__(self, attn, hidden_states, encoder_hidden_states, hidden_states_original_shape, *args, **kwargs):
        batch_size, sequence_length, _ = hidden_states.shape

        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        hidden_states = torch.nn.functional.scaled_dot_product_attention(
            query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states


class OmostCrossAttnProcessor:
    def __call__(self, attn, hidden_states, encoder_hidden_states, hidden_states_original_shape, *args, **kwargs):
        B, C, H, W = hidden_states_original_shape

        conds = []
        masks = []

        for m, c in encoder_hidden_states:
            m = torch.nn.functional.interpolate(m[None, None, :, :], (H, W), mode='nearest-exact').flatten().unsqueeze(1).repeat(1, c.size(1))
            conds.append(c)
            masks.append(m)

        conds = torch.cat(conds, dim=1)
        masks = torch.cat(masks, dim=1)

        mask_bool = masks > 0.5
        mask_scale = (H * W) / torch.sum(masks, dim=0, keepdim=True)

        batch_size, sequence_length, _ = conds.shape

        query = attn.to_q(hidden_states)
        key = attn.to_k(conds)
        value = attn.to_v(conds)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        mask_bool = mask_bool[None, None, :, :].repeat(query.size(0), query.size(1), 1, 1)
        mask_scale = mask_scale[None, None, :, :].repeat(query.size(0), query.size(1), 1, 1)

        sim = query @ key.transpose(-2, -1) * attn.scale
        sim = sim * mask_scale.to(sim)
        sim.masked_fill_(mask_bool.logical_not(), float("-inf"))
        sim = sim.softmax(dim=-1)

        h = sim @ value
        h = h.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)

        h = attn.to_out[0](h)
        h = attn.to_out[1](h)
        return h


class StableDiffusionXLOmostPipeline(StableDiffusionXLImg2ImgPipeline):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.k_model = KModel(unet=self.unet)

        attn_procs = {}
        for name in self.unet.attn_processors.keys():
            if name.endswith("attn2.processor"):
                attn_procs[name] = OmostCrossAttnProcessor()
            else:
                attn_procs[name] = OmostSelfAttnProcessor()

        self.unet.set_attn_processor(attn_procs)
        return

    @torch.inference_mode()
    def encode_bag_of_subprompts_greedy(self, prefixes: list[str], suffixes: list[str]):
        device = self.text_encoder.device

        @torch.inference_mode()
        def greedy_partition(items, max_sum):
            bags = []
            current_bag = []
            current_sum = 0

            for item in items:
                num = item['length']
                if current_sum + num > max_sum:
                    if current_bag:
                        bags.append(current_bag)
                    current_bag = [item]
                    current_sum = num
                else:
                    current_bag.append(item)
                    current_sum += num

            if current_bag:
                bags.append(current_bag)

            return bags

        @torch.inference_mode()
        def get_77_tokens_in_torch(subprompt_inds, tokenizer):
            # Note that all subprompt are theoretically less than 75 tokens (without bos/eos)
            result = [tokenizer.bos_token_id] + subprompt_inds[:75] + [tokenizer.eos_token_id] + [tokenizer.pad_token_id] * 75
            result = result[:77]
            result = torch.tensor([result]).to(device=device, dtype=torch.int64)
            return result

        @torch.inference_mode()
        def merge_with_prefix(bag):
            merged_ids_t1 = copy.deepcopy(prefix_ids_t1)
            merged_ids_t2 = copy.deepcopy(prefix_ids_t2)

            for item in bag:
                merged_ids_t1.extend(item['ids_t1'])
                merged_ids_t2.extend(item['ids_t2'])

            return dict(
                ids_t1=get_77_tokens_in_torch(merged_ids_t1, self.tokenizer),
                ids_t2=get_77_tokens_in_torch(merged_ids_t2, self.tokenizer_2)
            )

        @torch.inference_mode()
        def double_encode(pair_of_inds):
            inds = [pair_of_inds['ids_t1'], pair_of_inds['ids_t2']]
            text_encoders = [self.text_encoder, self.text_encoder_2]

            pooled_prompt_embeds = None
            prompt_embeds_list = []

            for text_input_ids, text_encoder in zip(inds, text_encoders):
                prompt_embeds = text_encoder(text_input_ids, output_hidden_states=True)

                # Only last pooler_output is needed
                pooled_prompt_embeds = prompt_embeds.pooler_output

                # "2" because SDXL always indexes from the penultimate layer.
                prompt_embeds = prompt_embeds.hidden_states[-2]
                prompt_embeds_list.append(prompt_embeds)

            prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
            return prompt_embeds, pooled_prompt_embeds

        # Begin with tokenizing prefixes

        prefix_length = 0
        prefix_ids_t1 = []
        prefix_ids_t2 = []

        for prefix in prefixes:
            ids_t1 = self.tokenizer(prefix, truncation=False, add_special_tokens=False).input_ids
            ids_t2 = self.tokenizer_2(prefix, truncation=False, add_special_tokens=False).input_ids
            assert len(ids_t1) == len(ids_t2)
            prefix_length += len(ids_t1)
            prefix_ids_t1 += ids_t1
            prefix_ids_t2 += ids_t2

        # Then tokenizing suffixes

        allowed_suffix_length = 75 - prefix_length
        suffix_targets = []

        for subprompt in suffixes:
            # Note that all subprompt are theoretically less than 75 tokens (without bos/eos)
            # So we can safely just crop it to 75
            ids_t1 = self.tokenizer(subprompt, truncation=False, add_special_tokens=False).input_ids[:75]
            ids_t2 = self.tokenizer_2(subprompt, truncation=False, add_special_tokens=False).input_ids[:75]
            assert len(ids_t1) == len(ids_t2)
            suffix_targets.append(dict(
                length=len(ids_t1),
                ids_t1=ids_t1,
                ids_t2=ids_t2
            ))

        # Then merge prefix and suffix tokens

        suffix_targets = greedy_partition(suffix_targets, max_sum=allowed_suffix_length)
        targets = [merge_with_prefix(b) for b in suffix_targets]

        # Encode!

        conds, poolers = [], []

        for target in targets:
            cond, pooler = double_encode(target)
            conds.append(cond)
            poolers.append(pooler)

        conds_merged = torch.concat(conds, dim=1)
        poolers_merged = poolers[0]

        return dict(cond=conds_merged, pooler=poolers_merged)

    @torch.inference_mode()
    def all_conds_from_canvas(self, canvas_outputs, negative_prompt):
        mask_all = torch.ones(size=(90, 90), dtype=torch.float32)
        negative_cond, negative_pooler = self.encode_cropped_prompt_77tokens(negative_prompt)
        negative_result = [(mask_all, negative_cond)]

        positive_result = []
        positive_pooler = None

        for item in canvas_outputs['bag_of_conditions']:
            current_mask = torch.from_numpy(item['mask']).to(torch.float32)
            current_prefixes = item['prefixes']
            current_suffixes = item['suffixes']

            current_cond = self.encode_bag_of_subprompts_greedy(prefixes=current_prefixes, suffixes=current_suffixes)

            if positive_pooler is None:
                positive_pooler = current_cond['pooler']

            positive_result.append((current_mask, current_cond['cond']))

        return positive_result, positive_pooler, negative_result, negative_pooler

    @torch.inference_mode()
    def encode_cropped_prompt_77tokens(self, prompt: str):
        device = self.text_encoder.device
        tokenizers = [self.tokenizer, self.tokenizer_2]
        text_encoders = [self.text_encoder, self.text_encoder_2]

        pooled_prompt_embeds = None
        prompt_embeds_list = []

        for tokenizer, text_encoder in zip(tokenizers, text_encoders):
            text_input_ids = tokenizer(
                prompt,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            ).input_ids

            prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=True)

            # Only last pooler_output is needed
            pooled_prompt_embeds = prompt_embeds.pooler_output

            # "2" because SDXL always indexes from the penultimate layer.
            prompt_embeds = prompt_embeds.hidden_states[-2]
            prompt_embeds_list.append(prompt_embeds)

        prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
        prompt_embeds = prompt_embeds.to(dtype=self.unet.dtype, device=device)

        return prompt_embeds, pooled_prompt_embeds

    @torch.inference_mode()
    def __call__(
            self,
            initial_latent: torch.FloatTensor = None,
            strength: float = 1.0,
            num_inference_steps: int = 25,
            guidance_scale: float = 5.0,
            batch_size: Optional[int] = 1,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            prompt_embeds: Optional[torch.FloatTensor] = None,
            negative_prompt_embeds: Optional[torch.FloatTensor] = None,
            pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
            negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
            cross_attention_kwargs: Optional[dict] = None,
    ):

        device = self.unet.device
        cross_attention_kwargs = cross_attention_kwargs or {}

        # Sigmas

        sigmas = self.k_model.get_sigmas_karras(int(num_inference_steps / strength))
        sigmas = sigmas[-(num_inference_steps + 1):].to(device)

        # Initial latents

        _, C, H, W = initial_latent.shape
        noise = randn_tensor((batch_size, C, H, W), generator=generator, device=device, dtype=self.unet.dtype)
        latents = initial_latent.to(noise) + noise * sigmas[0].to(noise)

        # Shape

        height, width = latents.shape[-2:]
        height = height * self.vae_scale_factor
        width = width * self.vae_scale_factor

        add_time_ids = list((height, width) + (0, 0) + (height, width))
        add_time_ids = torch.tensor([add_time_ids], dtype=self.unet.dtype)
        add_neg_time_ids = add_time_ids.clone()

        # Batch

        latents = latents.to(device)
        add_time_ids = add_time_ids.repeat(batch_size, 1).to(device)
        add_neg_time_ids = add_neg_time_ids.repeat(batch_size, 1).to(device)
        prompt_embeds = [(k.to(device), v.repeat(batch_size, 1, 1).to(noise)) for k, v in prompt_embeds]
        negative_prompt_embeds = [(k.to(device), v.repeat(batch_size, 1, 1).to(noise)) for k, v in negative_prompt_embeds]
        pooled_prompt_embeds = pooled_prompt_embeds.repeat(batch_size, 1).to(noise)
        negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.repeat(batch_size, 1).to(noise)

        # Feeds

        sampler_kwargs = dict(
            cfg_scale=guidance_scale,
            positive=dict(
                encoder_hidden_states=prompt_embeds,
                added_cond_kwargs={"text_embeds": pooled_prompt_embeds, "time_ids": add_time_ids},
                cross_attention_kwargs=cross_attention_kwargs
            ),
            negative=dict(
                encoder_hidden_states=negative_prompt_embeds,
                added_cond_kwargs={"text_embeds": negative_pooled_prompt_embeds, "time_ids": add_neg_time_ids},
                cross_attention_kwargs=cross_attention_kwargs
            )
        )

        # Sample

        results = sample_dpmpp_2m(self.k_model, latents, sigmas, extra_args=sampler_kwargs, disable=False)

        return StableDiffusionXLPipelineOutput(images=results)
