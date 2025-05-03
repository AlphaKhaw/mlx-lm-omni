from mlx_lm_audio import load, generate

model, tokenizer = load("Qwen/Qwen2.5-Omni-3B")

prompt = "Write a story about Einstein"

messages = [{"role": "user", "content": prompt}]
prompt = tokenizer.apply_chat_template(
    messages, add_generation_prompt=True
)

text = generate(model, tokenizer, prompt=prompt, verbose=True)
