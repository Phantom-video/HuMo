# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Inference codes adapted from [SeedVR]
# https://github.com/ByteDance-Seed/SeedVR/blob/main/projects/inference_seedvr2_7b.py

import math
import os
import gc
import random
import sys
import mediapy
import torch
import torch.distributed as dist
from omegaconf import DictConfig, ListConfig, OmegaConf
from einops import rearrange
from omegaconf import OmegaConf
from PIL import Image, ImageOps
from torchvision.transforms import ToTensor
from tqdm import tqdm
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import (
    BackwardPrefetch,
    FullyShardedDataParallel,
    MixedPrecision,
    ShardingStrategy,
)
from common.distributed import (
    get_device,
    get_global_rank,
    get_local_rank,
    meta_param_init_fn,
    meta_non_persistent_buffer_init_fn,
    init_torch,
)
from common.distributed.advanced import (
    init_unified_parallel,
    get_unified_parallel_world_size,
    get_sequence_parallel_rank,
    init_model_shard_cpu_group,
)
from common.logger import get_logger
from common.config import create_object
from common.distributed import get_device, get_global_rank
from torchvision.transforms import Compose, Normalize, ToTensor
from humo.models.wan_modules.t5 import T5EncoderModel
from humo.models.wan_modules.vae import WanVAE
from humo.models.utils.utils import tensor_to_video, prepare_json_dataset
from contextlib import contextmanager
import torch.cuda.amp as amp
from humo.models.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from humo.utils.audio_processor_whisper import AudioProcessor
from humo.utils.wav2vec import linear_interpolation_fps


image_transform = Compose([
    ToTensor(),
    Normalize(mean=0.5, std=0.5),
])

SIZE_CONFIGS = {
    '720*1280': (720, 1280),
    '1280*720': (1280, 720),
    '480*832': (480, 832),
    '832*480': (832, 480),
    '1024*1024': (1024, 1024),
}

def clever_format(nums, format="%.2f"):
    from typing import Iterable
    if not isinstance(nums, Iterable):
        nums = [nums]
    clever_nums = []
    for num in nums:
        if num > 1e12:
            clever_nums.append(format % (num / 1e12) + "T")
        elif num > 1e9:
            clever_nums.append(format % (num / 1e9) + "G")
        elif num > 1e6:
            clever_nums.append(format % (num / 1e6) + "M")
        elif num > 1e3:
            clever_nums.append(format % (num / 1e3) + "K")
        else:
            clever_nums.append(format % num + "B")

    clever_nums = clever_nums[0] if len(clever_nums) == 1 else (*clever_nums,)

    return clever_nums


class Generator():
    def __init__(self, config: DictConfig):
        self.config = config.copy()
        OmegaConf.set_readonly(self.config, True)
        self.logger = get_logger(self.__class__.__name__)
        
        init_torch(cudnn_benchmark=False)

    def entrypoint(self):
        self.configure_models()
        self.inference_loop()
    
    def get_fsdp_sharding_config(self, sharding_strategy, device_mesh_config):
        device_mesh = None
        fsdp_strategy = ShardingStrategy[sharding_strategy]
        if (
            fsdp_strategy in [ShardingStrategy._HYBRID_SHARD_ZERO2, ShardingStrategy.HYBRID_SHARD]
            and device_mesh_config is not None
        ):
            device_mesh = init_device_mesh("cuda", tuple(device_mesh_config))
        return device_mesh, fsdp_strategy

    def configure_models(self):
        self.configure_dit_model(device="cpu")
        self.configure_vae_model()
        if self.config.generation.get('extract_audio_feat', False):
            self.configure_wav2vec(device="cpu")
        self.configure_text_model(device="cpu")

        # Initialize fsdp.
        self.configure_dit_fsdp_model()
        self.configure_text_fsdp_model()
    
    def configure_dit_model(self, device=get_device()):

        init_unified_parallel(self.config.dit.sp_size)
        self.sp_size = get_unified_parallel_world_size()
        
        # Create dit model.
        init_device = "meta"
        with torch.device(init_device):
            self.dit = create_object(self.config.dit.model)
        self.logger.info(f"Load DiT model on {init_device}.")
        self.dit.eval().requires_grad_(False)

        # Load dit checkpoint.
        path = self.config.dit.checkpoint_dir
        if path.endswith(".pth"):
            state = torch.load(path, map_location=device, mmap=True)
            missing_keys, unexpected_keys = self.dit.load_state_dict(state, strict=False, assign=True)
            self.logger.info(
                f"dit loaded from {path}. "
                f"Missing keys: {len(missing_keys)}, "
                f"Unexpected keys: {len(unexpected_keys)}"
            )
        else:
            from safetensors.torch import load_file
            import json
            def load_custom_sharded_weights(model_dir, base_name, device=device):
                index_path = f"{model_dir}/{base_name}.safetensors.index.json"
                with open(index_path, "r") as f:
                    index = json.load(f)
                weight_map = index["weight_map"]
                shard_files = set(weight_map.values())
                state_dict = {}
                for shard_file in shard_files:
                    shard_path = f"{model_dir}/{shard_file}"
                    shard_state = load_file(shard_path)
                    shard_state = {k: v.to(device) for k, v in shard_state.items()}
                    state_dict.update(shard_state)
                return state_dict
            state = load_custom_sharded_weights(path, 'humo', device)
            self.dit.load_state_dict(state, strict=False, assign=True)
        
        self.dit = meta_non_persistent_buffer_init_fn(self.dit)

        # Print model size.
        params = sum(p.numel() for p in self.dit.parameters())
        self.logger.info(
            f"[RANK:{get_global_rank()}] DiT Parameters: {clever_format(params, '%.3f')}"
        )
    
    def configure_vae_model(self, device=get_device()):
        self.vae_stride = self.config.vae.vae_stride
        self.vae = WanVAE(
            vae_pth=self.config.vae.checkpoint,
            device=device)
        
        if self.config.generation.height == 480:
            self.zero_vae = torch.load(self.config.dit.zero_vae_path)
        elif self.config.generation.height == 720:
            self.zero_vae = torch.load(self.config.dit.zero_vae_720p_path)
        else:
            raise ValueError(f"Unsupported height {self.config.generation.height} for zero-vae.")
    
    def configure_wav2vec(self, device=get_device()):
        audio_separator_model_file = self.config.audio.vocal_separator
        wav2vec_model_path = self.config.audio.wav2vec_model

        self.audio_processor = AudioProcessor(
            16000,
            25,
            wav2vec_model_path,
            "all",
            audio_separator_model_file,
            None,  # not seperate
            os.path.join(self.config.generation.output.dir, "vocals"),
            device=device,
        )

    def configure_text_model(self, device=get_device()):
        self.text_encoder = T5EncoderModel(
            text_len=self.config.dit.model.text_len,
            dtype=torch.bfloat16,
            device=device,
            checkpoint_path=self.config.text.t5_checkpoint,
            tokenizer_path=self.config.text.t5_tokenizer,
            )

    
    def configure_dit_fsdp_model(self):
        from humo.models.wan_modules.model_humo import WanAttentionBlock

        dit_blocks = (WanAttentionBlock,)

        # Init model_shard_cpu_group for saving checkpoint with sharded state_dict.
        init_model_shard_cpu_group(
            self.config.dit.fsdp.sharding_strategy,
            self.config.dit.fsdp.get("device_mesh", None),
        )

        # Assert that dit has wrappable blocks.
        assert any(isinstance(m, dit_blocks) for m in self.dit.modules())

        # Define wrap policy on all dit blocks.
        def custom_auto_wrap_policy(module, recurse, *args, **kwargs):
            return recurse or isinstance(module, dit_blocks)

        # Configure FSDP settings.
        device_mesh, fsdp_strategy = self.get_fsdp_sharding_config(
            self.config.dit.fsdp.sharding_strategy,
            self.config.dit.fsdp.get("device_mesh", None),
        )
        settings = dict(
            auto_wrap_policy=custom_auto_wrap_policy,
            sharding_strategy=fsdp_strategy,
            backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
            device_id=get_local_rank(),
            use_orig_params=False,
            sync_module_states=True,
            forward_prefetch=True,
            limit_all_gathers=False,  # False for ZERO2.
            mixed_precision=MixedPrecision(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.float32,
                buffer_dtype=torch.float32,
            ),
            device_mesh=device_mesh,
            param_init_fn=meta_param_init_fn,
        )

        # Apply FSDP.
        self.dit = FullyShardedDataParallel(self.dit, **settings)
        # self.dit.to(get_device())


    def configure_text_fsdp_model(self):
        # If FSDP is not enabled, put text_encoder to GPU and return.
        if not self.config.text.fsdp.enabled:
            self.text_encoder.to(get_device())
            return

        # from transformers.models.t5.modeling_t5 import T5Block
        from humo.models.wan_modules.t5 import T5SelfAttention

        text_blocks = (torch.nn.Embedding, T5SelfAttention)
        # text_blocks_names = ("QWenBlock", "QWenModel")  # QWen cannot be imported. Use str.

        def custom_auto_wrap_policy(module, recurse, *args, **kwargs):
            return (
                recurse
                or isinstance(module, text_blocks)
            )

        # Apply FSDP.
        text_encoder_dtype = getattr(torch, self.config.text.dtype)
        device_mesh, fsdp_strategy = self.get_fsdp_sharding_config(
            self.config.text.fsdp.sharding_strategy,
            self.config.text.fsdp.get("device_mesh", None),
        )
        self.text_encoder = FullyShardedDataParallel(
            module=self.text_encoder,
            auto_wrap_policy=custom_auto_wrap_policy,
            sharding_strategy=fsdp_strategy,
            backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
            device_id=get_local_rank(),
            use_orig_params=False,
            sync_module_states=False,
            forward_prefetch=True,
            limit_all_gathers=True,
            mixed_precision=MixedPrecision(
                param_dtype=text_encoder_dtype,
                reduce_dtype=text_encoder_dtype,
                buffer_dtype=text_encoder_dtype,
            ),
            device_mesh=device_mesh,
        )
        self.text_encoder.to(get_device()).requires_grad_(False)


    def load_image_latent_ref_id(self, path: str, size, device):
        # Load size.
        h, w = size[1], size[0]

        # Load image.
        if len(path) > 1 and not isinstance(path, str):
            ref_vae_latents = []
            for image_path in path:
                with Image.open(image_path) as img:
                    img = img.convert("RGB")

                    # Calculate the required size to keep aspect ratio and fill the rest with padding.
                    img_ratio = img.width / img.height
                    target_ratio = w / h
                    
                    if img_ratio > target_ratio:  # Image is wider than target
                        new_width = w
                        new_height = int(new_width / img_ratio)
                    else:  # Image is taller than target
                        new_height = h
                        new_width = int(new_height * img_ratio)
                    
                    # img = img.resize((new_width, new_height), Image.ANTIALIAS)
                    img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

                    # Create a new image with the target size and place the resized image in the center
                    delta_w = w - img.size[0]
                    delta_h = h - img.size[1]
                    padding = (delta_w // 2, delta_h // 2, delta_w - (delta_w // 2), delta_h - (delta_h // 2))
                    new_img = ImageOps.expand(img, padding, fill=(255, 255, 255))

                    # Transform to tensor and normalize.
                    transform = Compose(
                        [
                            ToTensor(),
                            Normalize(0.5, 0.5),
                        ]
                    )
                    new_img = transform(new_img)
                    # img_vae_latent = self.vae_encode([new_img.unsqueeze(1)])[0]
                    img_vae_latent = self.vae.encode([new_img.unsqueeze(1)], device)
                    ref_vae_latents.append(img_vae_latent[0])

            return [torch.cat(ref_vae_latents, dim=1)]
        else:
            if not isinstance(path, str):
                path = path[0]
            with Image.open(path) as img:
                img = img.convert("RGB")

                # Calculate the required size to keep aspect ratio and fill the rest with padding.
                img_ratio = img.width / img.height
                target_ratio = w / h
                
                if img_ratio > target_ratio:  # Image is wider than target
                    new_width = w
                    new_height = int(new_width / img_ratio)
                else:  # Image is taller than target
                    new_height = h
                    new_width = int(new_height * img_ratio)
                
                # img = img.resize((new_width, new_height), Image.ANTIALIAS)
                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

                # Create a new image with the target size and place the resized image in the center
                delta_w = w - img.size[0]
                delta_h = h - img.size[1]
                padding = (delta_w // 2, delta_h // 2, delta_w - (delta_w // 2), delta_h - (delta_h // 2))
                new_img = ImageOps.expand(img, padding, fill=(255, 255, 255))

                # Transform to tensor and normalize.
                transform = Compose(
                    [
                        ToTensor(),
                        Normalize(0.5, 0.5),
                    ]
                )
                new_img = transform(new_img)
                img_vae_latent = self.vae.encode([new_img.unsqueeze(1)], device)

            # Vae encode.
            return img_vae_latent
    
    def get_audio_emb_window(self, audio_emb, frame_num, frame0_idx, audio_shift=2):
        zero_audio_embed = torch.zeros((audio_emb.shape[1], audio_emb.shape[2]), dtype=audio_emb.dtype, device=audio_emb.device)
        zero_audio_embed_3 = torch.zeros((3, audio_emb.shape[1], audio_emb.shape[2]), dtype=audio_emb.dtype, device=audio_emb.device)  # device=audio_emb.device
        iter_ = 1 + (frame_num - 1) // 4
        audio_emb_wind = []
        for lt_i in range(iter_):
            if lt_i == 0:
                st = frame0_idx + lt_i - 2
                ed = frame0_idx + lt_i + 3
                wind_feat = torch.stack([
                    audio_emb[i] if (0 <= i < audio_emb.shape[0]) else zero_audio_embed
                    for i in range(st, ed)
                ], dim=0)
                wind_feat = torch.cat((zero_audio_embed_3, wind_feat), dim=0)
            else:
                st = frame0_idx + 1 + 4 * (lt_i - 1) - audio_shift
                ed = frame0_idx + 1 + 4 * lt_i + audio_shift
                wind_feat = torch.stack([
                    audio_emb[i] if (0 <= i < audio_emb.shape[0]) else zero_audio_embed
                    for i in range(st, ed)
                ], dim=0)
            audio_emb_wind.append(wind_feat)
        audio_emb_wind = torch.stack(audio_emb_wind, dim=0)

        return audio_emb_wind, ed - audio_shift
    
    def audio_emb_enc(self, audio_emb, wav_enc_type="whisper"):
        if wav_enc_type == "wav2vec":
            feat_merge = audio_emb
        elif wav_enc_type == "whisper":
            feat0 = linear_interpolation_fps(audio_emb[:, :, 0: 8].mean(dim=2), 50, 25)
            feat1 = linear_interpolation_fps(audio_emb[:, :, 8: 16].mean(dim=2), 50, 25)
            feat2 = linear_interpolation_fps(audio_emb[:, :, 16: 24].mean(dim=2), 50, 25)
            feat3 = linear_interpolation_fps(audio_emb[:, :, 24: 32].mean(dim=2), 50, 25)
            feat4 = linear_interpolation_fps(audio_emb[:, :, 32], 50, 25)
            feat_merge = torch.stack([feat0, feat1, feat2, feat3, feat4], dim=2)[0]
        else:
            raise ValueError(f"Unsupported wav_enc_type: {wav_enc_type}")
        
        return feat_merge
    
    def parse_output(self, output):
        latent = output[0]
        mask = output[1] if len(output) > 1 else None
        return latent, mask
    
    def forward_tia(self, latents, timestep, t, step_change, arg_tia, arg_ti, arg_i, arg_null):
        pos_tia, _ = self.parse_output(self.dit(
            latents, t=timestep, **arg_tia
            ))
        torch.cuda.empty_cache()

        pos_ti, _ = self.parse_output(self.dit(
            latents, t=timestep, **arg_ti
            ))
        torch.cuda.empty_cache()

        if t > step_change:
            neg, _ = self.parse_output(self.dit(
                latents, t=timestep, **arg_i
                ))  # img included in null, same with official Wan-2.1
            torch.cuda.empty_cache()

            noise_pred = self.config.generation.scale_a * (pos_tia - pos_ti) + \
                    self.config.generation.scale_t * (pos_ti - neg) + \
                    neg
        else:
            neg, _ = self.parse_output(self.dit(
                latents, t=timestep, **arg_null
                ))  # img not included in null
            torch.cuda.empty_cache()

            noise_pred = self.config.generation.scale_a * (pos_tia - pos_ti) + \
                    (self.config.generation.scale_t - 2.0) * (pos_ti - neg) + \
                    neg
        return noise_pred
    
    def forward_ta(self, latents, timestep, arg_ta, arg_t, arg_null):
        pos_ta, _ = self.parse_output(self.dit(
            latents, t=timestep, **arg_ta
            ))
        torch.cuda.empty_cache()

        pos_t, _ = self.parse_output(self.dit(
            latents, t=timestep, **arg_t
            ))
        torch.cuda.empty_cache()

        neg, _ = self.parse_output(self.dit(
                latents, t=timestep, **arg_null
                ))
        torch.cuda.empty_cache()
            
        noise_pred = self.config.generation.scale_a * (pos_ta - pos_t) + \
                self.config.generation.scale_t * (pos_t - neg) + \
                neg
        return noise_pred
                    
    @torch.no_grad()
    def inference(self,
                 input_prompt,
                 img_path,
                 audio_path,
                 size=(1280, 720),
                 frame_num=81,
                 shift=5.0,
                 sample_solver='unipc',
                 sampling_steps=50,
                 n_prompt="",
                 seed=-1,
                 offload_model=True,
                 device = get_device(),
        ):

        self.vae.model.to(device=device)
        if img_path is not None:
            latents_ref = self.load_image_latent_ref_id(img_path, size, device)
        else:
            latents_ref = [torch.zeros(16, 1, size[1]//8, size[0]//8).to(device)]
            
        self.vae.model.to(device="cpu")
        latents_ref_neg = [torch.zeros_like(latent_ref) for latent_ref in latents_ref]
        
        # audio
        if audio_path is not None:
            if self.config.generation.extract_audio_feat:
                self.audio_processor.whisper.to(device=device)
                audio_emb, audio_length = self.audio_processor.preprocess(audio_path)
                self.audio_processor.whisper.to(device='cpu')
            else:
                audio_emb_path = audio_path.replace(".wav", ".pt")
                audio_emb = torch.load(audio_emb_path).to(device=device)
                audio_emb = self.audio_emb_enc(audio_emb, wav_enc_type="whisper")
                self.logger.info("使用预先提取好的音频特征: %s", audio_emb_path)
        else:
            audio_emb = torch.zeros(frame_num, 5, 1280).to(device)
            
        frame_num = frame_num if frame_num != -1 else audio_length
        frame_num = 4 * ((frame_num - 1) // 4) + 1
        audio_emb, _ = self.get_audio_emb_window(audio_emb, frame_num, frame0_idx=0)
        zero_audio_pad = torch.zeros(latents_ref[0].shape[1], *audio_emb.shape[1:]).to(audio_emb.device)
        audio_emb = torch.cat([audio_emb, zero_audio_pad], dim=0)
        audio_emb = [audio_emb.to(device)]
        audio_emb_neg = [torch.zeros_like(audio_emb[0])]
        
        # preprocess
        self.patch_size = self.config.dit.model.patch_size
        F = frame_num
        target_shape = (self.vae.model.z_dim, (F - 1) // self.vae_stride[0] + 1 + latents_ref[0].shape[1],
                        size[1] // self.vae_stride[1],
                        size[0] // self.vae_stride[2])

        seq_len = math.ceil((target_shape[2] * target_shape[3]) /
                            (self.patch_size[1] * self.patch_size[2]) *
                            target_shape[1] / self.sp_size) * self.sp_size

        if n_prompt == "":
            n_prompt = self.config.generation.sample_neg_prompt
        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=device)
        seed_g.manual_seed(seed)

        self.text_encoder.model.to(device)
        context = self.text_encoder([input_prompt], device)
        context_null = self.text_encoder([n_prompt], device)
        self.text_encoder.model.cpu()

        noise = [
            torch.randn(
                target_shape[0],
                target_shape[1], # - latents_ref[0].shape[1],
                target_shape[2],
                target_shape[3],
                dtype=torch.float32,
                device=device,
                generator=seed_g)
        ]

        @contextmanager
        def noop_no_sync():
            yield

        no_sync = getattr(self.dit, 'no_sync', noop_no_sync)
        step_change = self.config.generation.step_change # 980

        # evaluation mode
        with amp.autocast(dtype=torch.bfloat16), torch.no_grad(), no_sync():

            if sample_solver == 'unipc':
                sample_scheduler = FlowUniPCMultistepScheduler(
                    num_train_timesteps=1000,
                    shift=1,
                    use_dynamic_shifting=False)
                sample_scheduler.set_timesteps(
                    sampling_steps, device=device, shift=shift)
                timesteps = sample_scheduler.timesteps

            # sample videos
            latents = noise

            msk = torch.ones(4, target_shape[1], target_shape[2], target_shape[3], device=get_device())
            msk[:,:-latents_ref[0].shape[1]] = 0

            zero_vae = self.zero_vae[:, :(target_shape[1]-latents_ref[0].shape[1])].to(
                device=get_device(), dtype=latents_ref[0].dtype)
            y_c = torch.cat([
                zero_vae,
                latents_ref[0]
                ], dim=1)
            y_c = [torch.concat([msk, y_c])]

            y_null = self.zero_vae[:, :target_shape[1]].to(
                device=get_device(), dtype=latents_ref[0].dtype)
            y_null = [torch.concat([msk, y_null])]

            arg_null = {'seq_len': seq_len, 'audio': audio_emb_neg, 'y': y_null, 'context': context_null}
            arg_t = {'seq_len': seq_len, 'audio': audio_emb_neg, 'y': y_null, 'context': context}
            arg_i = {'seq_len': seq_len, 'audio': audio_emb_neg, 'y': y_c, 'context': context_null}
            arg_ti = {'seq_len': seq_len, 'audio': audio_emb_neg, 'y': y_c, 'context': context}
            arg_ta = {'seq_len': seq_len, 'audio': audio_emb, 'y': y_null, 'context': context}
            arg_tia = {'seq_len': seq_len, 'audio': audio_emb, 'y': y_c, 'context': context}
            
            torch.cuda.empty_cache()
            self.dit.to(device=get_device())
            for _, t in enumerate(tqdm(timesteps)):
                timestep = [t]
                timestep = torch.stack(timestep)

                if self.config.generation.mode == "TIA":
                    noise_pred = self.forward_tia(latents, timestep, t, step_change, 
                                                  arg_tia, arg_ti, arg_i, arg_null)
                elif self.config.generation.mode == "TA":
                    noise_pred = self.forward_ta(latents, timestep, arg_ta, arg_t, arg_null)
                else:
                    raise ValueError(f"Unsupported generation mode: {self.config.generation.mode}")

                temp_x0 = sample_scheduler.step(
                    noise_pred.unsqueeze(0),
                    t,
                    latents[0].unsqueeze(0),
                    return_dict=False,
                    generator=seed_g)[0]
                latents = [temp_x0.squeeze(0)]

                del timestep
                torch.cuda.empty_cache()

            x0 = latents
            x0 = [x0_[:,:-latents_ref[0].shape[1]] for x0_ in x0]

            # if offload_model:
            self.dit.cpu()
            torch.cuda.empty_cache()
            # if get_local_rank() == 0:
            self.vae.model.to(device=device)
            videos = self.vae.decode(x0)
            self.vae.model.to(device="cpu")

        del noise, latents, noise_pred
        del audio_emb, audio_emb_neg, latents_ref, latents_ref_neg, context, context_null
        del x0, temp_x0
        del sample_scheduler
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()

        return videos[0] # if get_local_rank() == 0 else None


    def inference_loop(self):
        gen_config = self.config.generation
        pos_prompts = self.prepare_positive_prompts()
        
        # Create output dir.
        os.makedirs(gen_config.output.dir, exist_ok=True)

        # Start generation.
        for prompt in pos_prompts:
            seed = self.config.generation.seed
            seed = seed if seed is not None else random.randint(0, 100000)

            audio_path = prompt.get("audio", None)
            ref_img_path = prompt.get("ref_img", None)
            itemname = prompt.get("itemname", None)
            if "I" not in self.config.generation.mode:
                ref_img_path = None
            if "A" not in self.config.generation.mode:
                audio_path = None

            video = self.inference(
                prompt.text,
                ref_img_path,
                audio_path,
                size=SIZE_CONFIGS[f"{gen_config.width}*{gen_config.height}"],
                frame_num=gen_config.frames,
                shift=self.config.diffusion.timesteps.sampling.shift,
                sample_solver='unipc',
                sampling_steps=self.config.diffusion.timesteps.sampling.steps,
                seed=seed,
                offload_model=False,
            )

            torch.cuda.empty_cache()
            gc.collect()
            

            # Save samples.
            if get_sequence_parallel_rank() == 0:
                pathname = self.save_sample(
                    sample=video,
                    audio_path=audio_path,
                    itemname=itemname,
                )
                self.logger.info(f"Finished {itemname}, saved to {pathname}.")
            
            del video, prompt
            torch.cuda.empty_cache()
            gc.collect()
            

    def save_sample(self, *, sample: torch.Tensor, audio_path: str, itemname: str):
        gen_config = self.config.generation
        # Prepare file path.
        extension = ".mp4" if sample.ndim == 4 else ".png"
        filename = f"{itemname}_seed{gen_config.seed}"
        filename += extension
        pathname = os.path.join(gen_config.output.dir, filename)
        # Convert sample.
        sample = sample.clip(-1, 1).mul_(0.5).add_(0.5).mul_(255).to("cpu", torch.uint8)
        sample = rearrange(sample, "c t h w -> t h w c")
        # Save file.
        if sample.ndim == 4:
            if audio_path is not None:
                tensor_to_video(
                    sample.numpy(),
                    pathname,
                    audio_path,
                    fps=gen_config.fps)
            else:
                mediapy.write_video(
                path=pathname,
                images=sample.numpy(),
                fps=gen_config.fps,
            )
        else:
            raise ValueError
        return pathname
    

    def prepare_positive_prompts(self):
        pos_prompts = self.config.generation.positive_prompt
        if pos_prompts.endswith(".json"):
            pos_prompts = prepare_json_dataset(pos_prompts)
        else:
            raise NotImplementedError
        assert isinstance(pos_prompts, ListConfig)

        return pos_prompts