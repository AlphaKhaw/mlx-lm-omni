import argparse
import librosa
import numpy as np
import os
from mlx_lm_omni import load, generate
import librosa
from io import BytesIO
from urllib.request import urlopen

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, help='Path to the model')
parser.add_argument('--text-model-id', type=str, help='Override the text model id for ultravox model')

args = parser.parse_args()

if args.model is None:
    args.model = "Qwen/Qwen2.5-Omni-3B"

model, tokenizer = load(args.model, model_config={"text_model_id": args.text_model_id})

audio1, sr1 = librosa.load("audios/what-is-your-name.wav", sr=16000)
audio2, sr2 = librosa.load("audios/noise.wav", sr=16000)
audio3, sr3 = librosa.load("audios/where-are-you-from.wav", sr=16000)

messages = [
    {"role": "system", "content": """You are a expert voice assistant. Explain what you hear in the audio."""},
    {"role": "user", "content": "Audio 1:", "audio": audio1},
    {"role": "assistant", "content": "A person asked: What is your name?"},
    {"role": "user", "content": "Audio 3:", "audio": audio2},
]
prompt = tokenizer.apply_chat_template(
    messages, add_generation_prompt=True
)
text = generate(model, tokenizer, prompt=prompt)
print(text)