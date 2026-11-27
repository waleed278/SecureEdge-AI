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



# Eight labeled evaluation examples
test_data = [
    ("My Wi-Fi disconnects every few minutes.", "NETWORK"),
    ("Office VPN cannot connect.", "NETWORK"),
    ("I forgot my password.", "ACCOUNT"),
    ("My account is locked.", "ACCOUNT"),
    ("My keyboard has a broken key.", "HARDWARE"),
    ("My laptop fan makes grinding noises.", "HARDWARE"),
    ("Microsoft Word crashes when opened.", "SOFTWARE"),
    ("App installation reports a missing DLL.", "SOFTWARE"),
]

correct = 0

for ticket, expected in test_data:



    messages = [
    {
        "role": "system",
        "content": (
            "You are an IT support ticket classifier.\n"
            "Choose exactly ONE category based on the main issue:\n"
            "NETWORK: Wi-Fi, VPN, internet, or connectivity issues.\n"
            "ACCOUNT: Passwords, login, or account access issues.\n"
            "HARDWARE: Physical devices, keyboards, screens, or fans.\n"
            "SOFTWARE: Application errors, crashes, or installation issues.\n"
            "Output exactly one uppercase category name."
        )
    },
    {
        "role": "user",
        "content": f"Ticket: {ticket}"
    }
]


    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    ).to("cuda")

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=15,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    new_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
    prediction = tokenizer.decode(
        new_tokens,
        skip_special_tokens=True
    ).strip().upper()

    is_correct = prediction == expected

    if is_correct:
        correct += 1

    print(f"Expected: {expected}")
    print(f"Predicted: {prediction}")
    print(f"Correct: {is_correct}")
    print("-" * 35)

accuracy = correct / len(test_data) * 100

print(f"Correct predictions: {correct}/{len(test_data)}")
print(f"Accuracy: {accuracy:.2f}%")
