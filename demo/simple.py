from mlx_lm_audio import load, generate
import librosa

model, tokenizer = load("Qwen/Qwen2.5-Omni-7B")

audio, sr = librosa.load("demo/what-name.mp3", sr=16000)

messages = [
    {"role": "system", "content": [{"type": "text", "text": "You are a speech recognition model."}]},
    {"role": "user", "content": [
        {"type": "audio", "audio": audio},
        {"type": "text", "text": "Transcribe the English audio into text without any punctuation marks."},
    ]
    },
]
prompt = tokenizer.apply_chat_template(
    messages, add_generation_prompt=True
)

text = generate(model, tokenizer, prompt=prompt, verbose=True)
