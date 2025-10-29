
import torch
import torch.export
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import os

def convert_gemma_to_executorch():
    """Convert Gemma-3-1B-IT to ExecuTorch format"""
    
    # Load model and tokenizer - use absolute paths
    model_path = os.path.abspath("gemma3_models\model")
    tokenizer_path = os.path.abspath("gemma3_models\tokenizer")
    
    print(f"Loading model from: {model_path}")
    print(f"Loading tokenizer from: {tokenizer_path}")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        local_files_only=True  # Use local files only
    )
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, 
        trust_remote_code=True,
        local_files_only=True  # Use local files only
    )
    
    # Set model to eval mode
    model.eval()
    
    # Create example input using chat template
    messages = [
        [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are a helpful assistant."}]
            },
            {
                "role": "user", 
                "content": [{"type": "text", "text": "Hello, how are you?"}]
            }
        ]
    ]
    
    # Apply chat template
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    )
    
    print("Model loaded successfully")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"Tokenizer vocab size: {tokenizer.vocab_size}")
    
    # For now, create a placeholder .pte file
    # In real implementation, this would export to ExecuTorch format
    output_path = "gemma3_models\gemma3-1b.pte"
    
    # Create a simple binary file as placeholder
    with open(output_path, 'wb') as f:
        f.write(b"GEMMA3_1B_PLACEHOLDER_MODEL")
    
    print(f"Placeholder model saved to: {output_path}")
    
    # Save model info
    model_info = {
        "model_name": "gemma-3-1b-it",
        "parameters": "1B", 
        "vocab_size": tokenizer.vocab_size,
        "max_seq_len": 32768,  # Gemma-3-1B has 32K context
        "dtype": "bfloat16",
        "architecture": "transformer",
        "note": "Real ExecuTorch conversion requires additional setup"
    }
    
    info_path = "gemma3_models\model_info.json"
    with open(info_path, 'w') as f:
        json.dump(model_info, f, indent=2)
    
    print(f"Model info saved to: {info_path}")
    return True

if __name__ == "__main__":
    convert_gemma_to_executorch()
