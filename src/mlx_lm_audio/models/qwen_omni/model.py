import mlx.nn as nn
import mlx.core as mx
from mlx_lm.utils import load_model, load_tokenizer, get_model_path, load_adapters
from dataclasses import dataclass
from typing import List, Optional, Tuple, Type
from mlx_lm.tokenizer_utils import TokenizerWrapper
import numpy as np

from mlx_lm_audio.audio_mel import AudioMel, AudioMelConfig

from .thinker import Thinker
from .audio_tower import AudioTower
from mlx_lm_audio.tokenizer import ExtendedEmbedding, ExtendedTokenizer

@dataclass
class ModelArgs:
    thinker_config: dict
    
    @staticmethod
    def from_dict(cfg: dict) -> "ModelArgs":
        return ModelArgs(
            thinker_config=cfg["thinker_config"]
        )

class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        self.thinker = Thinker(args.thinker_config)
        
    @property
    def layers(self) -> list[nn.Module]:
        return self.thinker.layers
    
    def __call__(self, inputs: mx.array, cache = None) -> mx.array:
        return self.thinker(inputs, cache=cache)
    
    def build_custom_tokenizer(self, tokenizer: TokenizerWrapper) -> ExtendedTokenizer:
        return TokenizerWithAudio(self.thinker.audio_tower, tokenizer, self.thinker.model.embed_tokens)

class TokenizerWithAudio(ExtendedTokenizer):
    def __init__(self, audio_tower: AudioTower, tokenizer: TokenizerWrapper, embeddings: ExtendedEmbedding):
        self._audio_mel = AudioMel(AudioMelConfig.from_dict({
            "feature_size": 128,
            "sampling_rate": 16000,
            "hop_length": 160,
            "n_fft": 400,
        }))
        self._audio_tower = audio_tower
        self._embeddings = embeddings
        self._tokenizer = tokenizer
        
    @property
    def eos_token_id(self) -> int:
        return self._tokenizer.eos_token_id
    
    def clean_up_tokenization_spaces(self) -> int:
        return self._tokenizer.clean_up_tokenization_spaces()

    def encode(self, text: str) -> list[int]:
        return self._tokenizer.encode(text)
    
    def decode(self, tokens: list[int]) -> str:
        return self._tokenizer.decode(tokens)
        
    def encode_audio(self, audio: np.ndarray) -> list[int]:
        mel = self._audio_mel.forward(audio)
        audio_tower = self._audio_tower.forward(mel)
        return self._embeddings.embed_audio_chunk(audio_tower)
    
    def apply_chat_template(self, messages: list[dict], add_generation_prompt: bool = True) -> str:
        return self._tokenizer.apply_chat_template(messages, add_generation_prompt=add_generation_prompt)