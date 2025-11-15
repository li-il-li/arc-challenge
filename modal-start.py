import modal

app = modal.App("modernbert-inference")

image = modal.Image.debian_slim(python_version="3.11").uv_sync()

# Create a volume to cache model weights
volume = modal.Volume.from_name("modernbert-cache", create_if_missing=True)
MODEL_DIR = "/cache"


@app.function(
    image=image,
    gpu="T4",
    volumes={MODEL_DIR: volume},
)
def predict_masked(text: str):
    """Run masked language modeling with ModernBERT-base"""
    from transformers import AutoTokenizer, AutoModelForMaskedLM
    from pprint import pprint
    import torch

    print(f"Loading ModernBERT-base model...")
    tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base", cache_dir=MODEL_DIR)
    model = AutoModelForMaskedLM.from_pretrained("answerdotai/ModernBERT-base", cache_dir=MODEL_DIR)

    print(f"Running inference on: {text}")
    inputs = tokenizer(text, return_tensors="pt")
    
    print(f"Printing out tokenized inputs...")
    pprint(inputs)

    with torch.no_grad():
        outputs = model(**inputs)

    print(f"Printing out model output...")
    pprint(outputs)
    # Get predictions for masked tokens
    mask_token_index = (inputs.input_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)[1]

    if len(mask_token_index) > 0:
        predicted_token_id = outputs.logits[0, mask_token_index].argmax(axis=-1)
        predicted_tokens = tokenizer.decode(predicted_token_id)
        print(f"Predicted tokens: {predicted_tokens}")
        return predicted_tokens
    else:
        print("No [MASK] tokens found in input")
        return None


@app.local_entrypoint()
def main():
    # Example: predict masked word
    text = "The capital of France is [MASK]."
    result = predict_masked.remote(text)
    print(f"\nResult: {result}")
