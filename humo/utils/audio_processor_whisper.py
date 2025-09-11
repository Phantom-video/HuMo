# pylint: disable=C0301
'''
This module contains the AudioProcessor class and related functions for processing audio data.
It utilizes various libraries and models to perform tasks such as preprocessing, feature extraction,
and audio separation. The class is initialized with configuration parameters and can process
audio files using the provided models.
'''
import math
import os

import librosa
import numpy as np
import torch
from audio_separator.separator import Separator
from einops import rearrange
from transformers import WhisperModel, AutoFeatureExtractor

from humo.utils.wav2vec import linear_interpolation_fps


class AudioProcessor:
    """
    AudioProcessor is a class that handles the processing of audio files.
    It takes care of preprocessing the audio files, extracting features
    using wav2vec models, and separating audio signals if needed.

    :param sample_rate: Sampling rate of the audio file
    :param fps: Frames per second for the extracted features
    :param wav2vec_model_path: Path to the wav2vec model
    :param only_last_features: Whether to only use the last features
    :param audio_separator_model_path: Path to the audio separator model
    :param audio_separator_model_name: Name of the audio separator model
    :param cache_dir: Directory to cache the intermediate results
    :param device: Device to run the processing on
    """
    def __init__(
        self,
        sample_rate,
        fps,
        wav2vec_model_path,
        wav2vec_feature_type,
        audio_separator_model_path:str=None,
        audio_separator_model_name:str=None,
        cache_dir:str='',
        device="cuda:0",
    ) -> None:
        self.sample_rate = sample_rate
        self.fps = fps
        self.device = device

        self.whisper = WhisperModel.from_pretrained(wav2vec_model_path).to(device).eval()
        self.whisper.requires_grad_(False)
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(wav2vec_model_path)

        if audio_separator_model_name is not None:
            try:
                os.makedirs(cache_dir, exist_ok=True)
            except OSError as _:
                print("Fail to create the output cache dir.")
            self.audio_separator = Separator(
                output_dir=cache_dir,
                output_single_stem="vocals",
                model_file_dir=audio_separator_model_path,
            )
            self.audio_separator.load_model(audio_separator_model_name)
            assert self.audio_separator.model_instance is not None, "Fail to load audio separate model."
        else:
            self.audio_separator=None
            print("Use audio directly without vocals seperator.")        


    def get_audio_feature(self, audio_path):
        audio_input, sampling_rate = librosa.load(audio_path, sr=16000)
        assert sampling_rate == 16000

        audio_features = []
        window = 750*640
        for i in range(0, len(audio_input), window):
            audio_feature = self.feature_extractor(audio_input[i:i+window], 
                                            sampling_rate=sampling_rate, 
                                            return_tensors="pt", 
                                            ).input_features
            audio_features.append(audio_feature)
        audio_features = torch.cat(audio_features, dim=-1)
        return audio_features, len(audio_input) // 640


    def preprocess(self, audio_path: str):
        audio_input, audio_len = self.get_audio_feature(audio_path)
        audio_feature = audio_input.to(self.whisper.device).float()
        window = 3000
        audio_prompts = []
        for i in range(0, audio_feature.shape[-1], window):
            audio_prompt = self.whisper.encoder(audio_feature[:,:,i:i+window], output_hidden_states=True).hidden_states
            audio_prompt = torch.stack(audio_prompt, dim=2)
            audio_prompts.append(audio_prompt)

        audio_prompts = torch.cat(audio_prompts, dim=1)
        audio_prompts = audio_prompts[:,:audio_len*2]

        audio_emb = self.audio_emb_enc(audio_prompts, wav_enc_type="whisper")

        return audio_emb, audio_emb.shape[0]
    
    def audio_emb_enc(self, audio_emb, wav_enc_type="whisper"):
        if wav_enc_type == "wav2vec":
            feat_merge = audio_emb
        elif wav_enc_type == "whisper":
            # [1, T, 33, 1280]
            feat0 = linear_interpolation_fps(audio_emb[:, :, 0: 8].mean(dim=2), 50, 25)
            feat1 = linear_interpolation_fps(audio_emb[:, :, 8: 16].mean(dim=2), 50, 25)
            feat2 = linear_interpolation_fps(audio_emb[:, :, 16: 24].mean(dim=2), 50, 25)
            feat3 = linear_interpolation_fps(audio_emb[:, :, 24: 32].mean(dim=2), 50, 25)
            feat4 = linear_interpolation_fps(audio_emb[:, :, 32], 50, 25)
            feat_merge = torch.stack([feat0, feat1, feat2, feat3, feat4], dim=2)[0]  # [T, 5, 1280]
        else:
            raise ValueError(f"Unsupported wav_enc_type: {wav_enc_type}")
        
        return feat_merge
    
    def get_audio_emb_window(self, audio_emb, frame_num, frame0_idx, audio_shift=2):
        zero_audio_embed = torch.zeros((audio_emb.shape[1], audio_emb.shape[2]), dtype=audio_emb.dtype, device=audio_emb.device)
        zero_audio_embed_3 = torch.zeros((3, audio_emb.shape[1], audio_emb.shape[2]), dtype=audio_emb.dtype, device=audio_emb.device)  # device=audio_emb.device
        iter_ = 1 + (frame_num - 1) // 4
        audio_emb_wind = []
        for lt_i in range(iter_):
            if lt_i == 0:  # latent_i
                # 提取第一帧VAElatent，audio左侧补0，标识出
                st = frame0_idx + lt_i - 2
                ed = frame0_idx + lt_i + 3
                wind_feat = torch.stack([
                    audio_emb[i] if (0 <= i < audio_emb.shape[0]) else zero_audio_embed
                    for i in range(st, ed)
                ], dim=0)  # [5, 13, 768]
                wind_feat = torch.cat((zero_audio_embed_3, wind_feat), dim=0)  # [8, 13, 768]
            else:
                st = frame0_idx + 1 + 4 * (lt_i - 1) - audio_shift
                ed = frame0_idx + 1 + 4 * lt_i + audio_shift
                wind_feat = torch.stack([
                    audio_emb[i] if (0 <= i < audio_emb.shape[0]) else zero_audio_embed
                    for i in range(st, ed)
                ], dim=0)  # [8, 13, 768]
            audio_emb_wind.append(wind_feat)
        audio_emb_wind = torch.stack(audio_emb_wind, dim=0)  # [iter_, 8, 13, 768]

        return audio_emb_wind, ed - audio_shift

    def close(self):
        """
        TODO: to be implemented
        """
        return self

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc_val, _exc_tb):
        self.close()
