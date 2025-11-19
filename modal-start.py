import modal

app = modal.App("modernbert-inference")

image = modal.Image.debian_slim(python_version="3.11").uv_sync()

# Create a volume to cache model weights
volume = modal.Volume.from_name("modernbert-cache", create_if_missing=True)
MODEL_DIR = "/cache"

# Parameter
SEED=42
BATCH_SIZE=10
LEARNING_RATE=5e-5
EPOCHS=3

@app.function(
    image=image,
    gpu="T4",
    volumes={MODEL_DIR: volume},
    #scaledown_window=30,
)

    
def predict_masked(text: str):

    def ppp(header: str, ob: object):
        print("=" * 80)
        print(f"{header}:")
        print("-" * 80)
        pprint(ob)

    """Run masked language modeling with ModernBERT-base"""
    from transformers import AutoTokenizer, AutoModelForMaskedLM, PreTrainedModel, PreTrainedTokenizerFast
    from datasets import load_dataset
    from pprint import pprint
    from itertools import islice
    import torch
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Device {device}")

    # Load dataset
    # dataset = load_dataset("wikitext", "wikitext-2-raw-v1", cache_dir=MODEL_DIR)['train'].with_format('torch')
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", cache_dir=MODEL_DIR)
    ppp('Dataset', dataset)

    train_dataloader = DataLoader(dataset['train'].shuffle(seed=SEED).with_format('torch'), BATCH_SIZE)
    test_dataloader = DataLoader(dataset['test'].shuffle(seed=SEED).with_format('torch'), BATCH_SIZE)
    validation_dataloader = DataLoader(dataset['validation'].shuffle(seed=SEED).with_format('torch'), BATCH_SIZE)
    
    # Load tokenizer and instantiate the model
    print(f"Loading ModernBERT-base model...")
    tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base", cache_dir=MODEL_DIR)
    #tokenizer.to(device)
    model: PreTrainedModel = AutoModelForMaskedLM.from_pretrained("answerdotai/ModernBERT-base", cache_dir=MODEL_DIR)
    model.to(device)
    ppp('Model', model)

    # Load optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    # Training Loop
    model.train()
    
    for epoch in range(EPOCHS):
        for i, batch in enumerate(train_dataloader):
            inputs = tokenizer(batch['text'], return_tensors='pt', padding=True, truncation=True, max_length=512) # Up to 8192 tokens
            inputs = inputs.to(device=device)
            ppp('Inputs 1', inputs)
            
            # CREATE MASKS - This is missing!
            labels = inputs['input_ids'].clone()
            rand = torch.rand(inputs['input_ids'].shape)
            mask_arr = (rand < 0.15) * (inputs['input_ids'] != tokenizer.pad_token_id)  # Mask 15% of tokens
            inputs['input_ids'][mask_arr] = tokenizer.mask_token_id

            # Set non-masked positions to -100 (tells loss function to IGNORE them)
            labels[~mask_arr] = -100

            # Forward pass with original tokens as labels
            outputs = model(**inputs, labels=labels)
            loss = outputs.loss
            
            # Backward pass
            optimizer.zero_grad() # Clear old gradients
            loss.backward() # Compute new gradients
            optimizer.step() # Update weigths
            

    # Inference
    # with torch.no_grad():
    #     outputs = model(**inputs)
    #     ppp('Outputs', outputs)

    #print(f"Printing out model output...")
    #pprint(outputs)
    ## Get predictions for masked tokens
    #mask_token_index = (inputs.input_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)[1]

    #if len(mask_token_index) > 0:
    #    predicted_token_id = outputs.logits[0, mask_token_index].argmax(axis=-1)
    #    predicted_tokens = tokenizer.decode(predicted_token_id)
    #    print(f"Predicted tokens: {predicted_tokens}")
    #    return predicted_tokens
    #else:
    #    print("No [MASK] tokens found in input")
    #    return None


@app.local_entrypoint()
def main():
    # Example: predict masked word
    text = "The capital of France is [MASK]."
    result = predict_masked.remote(text)
    print(f"\nResult: {result}")
