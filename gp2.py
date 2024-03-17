import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

#.\myenv\scripts\activate.ps1

# Load pre-trained model tokenizer (vocabulary)
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Load pre-trained model
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Encode a text inputs
text = "Summarize the main arguments presented in the debate on renewable energy, and conclude with the most compelling point."
indexed_tokens = tokenizer.encode(text, add_special_tokens=True)
tokens_tensor = torch.tensor([indexed_tokens])

# Generate a text sample
model.eval()
with torch.no_grad():
    outputs = model.generate(
        tokens_tensor, 
        max_length=200, 
        num_return_sequences=1, 
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True, 
        temperature=0.7, 
        top_k=50, 
        top_p=0.95)

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)