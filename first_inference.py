
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Model configuration
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
CACHE_DIR = "./model_cache"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    cache_dir=CACHE_DIR
)

# Load model with half-precision weights
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float16,
    cache_dir=CACHE_DIR
)

# Move model to NVIDIA GPU
model = model.to("cuda")
model.eval()

print("Model device:", next(model.parameters()).device)
print("Model dtype:", next(model.parameters()).dtype)

# Customer-support example
messages = [
    {
        "role": "user",
        "content": (
            "Classify this customer support ticket. "
            "Choose one category: NETWORK, ACCOUNT, HARDWARE, SOFTWARE. "
            "Return only the category name.\n"
            "Ticket: My password was reset but I still cannot log in."
        )
    }
]

# Convert conversation into token IDs
inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt"
).to("cuda")

# Generate prediction
with torch.inference_mode():
    outputs = model.generate(
        **inputs,
        max_new_tokens=30,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )

# Remove input tokens from generated sequence
new_tokens = outputs[0][inputs["input_ids"].shape[-1]:]

# Convert token IDs back into readable text
answer = tokenizer.decode(new_tokens, skip_special_tokens=True)

print("Model prediction:", answer)
print(
    "Peak GPU memory allocated (MiB):",
    round(torch.cuda.max_memory_allocated() / (1024 ** 2), 2)
)



