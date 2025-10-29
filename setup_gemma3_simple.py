#!/usr/bin/env python3
"""
Simple Gemma-3-1B-IT Model Setup Script
Creates the necessary files for Android integration without complex conversion
"""

import os
import json
import shutil
from pathlib import Path

def setup_gemma3_model():
    """Set up Gemma-3-1B-IT model files for Android integration"""
    
    print("🚀 Setting up Gemma-3-1B-IT model for Android...")
    
    # Create directories
    gemma3_dir = Path("gemma3_models")
    gemma3_dir.mkdir(exist_ok=True)
    
    android_assets = Path("app/src/main/assets")
    android_assets.mkdir(parents=True, exist_ok=True)
    
    # Create Android asset directories
    model_assets = android_assets / "gemma3_models"
    tokenizer_assets = android_assets / "gemma3_tokenizers"
    context_assets = android_assets / "gemma3_context_binaries"
    
    for dir_path in [model_assets, tokenizer_assets, context_assets]:
        dir_path.mkdir(exist_ok=True)
    
    # Check if model was downloaded from Hugging Face cache
    hf_cache = Path.home() / ".cache" / "huggingface" / "hub" / "models--google--gemma-3-1b-it"
    
    if hf_cache.exists():
        print("✅ Found downloaded Gemma-3-1B-IT model in Hugging Face cache")
        
        # Find the latest snapshot
        snapshots = list(hf_cache.glob("snapshots/*"))
        if snapshots:
            latest_snapshot = max(snapshots, key=os.path.getctime)
            print(f"📁 Using snapshot: {latest_snapshot.name}")
            
            # Copy model files
            model_files = ["config.json", "generation_config.json", "model.safetensors"]
            for file_name in model_files:
                src_file = latest_snapshot / file_name
                if src_file.exists():
                    dst_file = model_assets / file_name
                    shutil.copy2(src_file, dst_file)
                    print(f"  ✅ Copied {file_name}")
            
            # Copy tokenizer files
            tokenizer_files = ["tokenizer.json", "tokenizer.model", "tokenizer_config.json", 
                             "special_tokens_map.json", "added_tokens.json"]
            for file_name in tokenizer_files:
                src_file = latest_snapshot / file_name
                if src_file.exists():
                    dst_file = tokenizer_assets / file_name
                    shutil.copy2(src_file, dst_file)
                    print(f"  ✅ Copied {file_name}")
    
    # Create model info
    model_info = {
        "model_name": "Gemma-3-1B-IT",
        "version": "1.0",
        "description": "Google Gemma-3-1B-IT instruction-tuned model for ExecuTorch + QNN",
        "parameters": "1B",
        "architecture": "transformer",
        "vocab_size": 256000,
        "max_seq_len": 32768,  # Gemma-3-1B has 32K context window
        "prefill_ar_len": 128,
        "backend": "ExecuTorch + Qualcomm QNN",
        "mode": "hybrid",
        "temperature": 0.0,
        "dtype": "bfloat16",
        "multimodal": True,  # Gemma-3 supports text and images
        "languages": 140,  # Supports 140+ languages
        "files": {
            "model": "model.safetensors",
            "config": "config.json",
            "tokenizer": "tokenizer.json",
            "tokenizer_config": "tokenizer_config.json"
        },
        "download_info": {
            "source": "Hugging Face",
            "model_id": "google/gemma-3-1b-it",
            "converted_for": "ExecuTorch + QNN",
            "android_optimized": True,
            "license": "Gemma Terms of Use"
        }
    }
    
    # Save model info
    info_path = gemma3_dir / "model_info.json"
    with open(info_path, 'w') as f:
        json.dump(model_info, f, indent=2)
    
    # Copy to Android assets
    shutil.copy2(info_path, model_assets / "model_info.json")
    
    print(f"✅ Model info saved to: {info_path}")
    
    # Create QNN configuration
    qnn_config = {
        "backend": "qnn",
        "model_mode": "hybrid",
        "max_seq_len": 32768,
        "prefill_ar_len": 128,
        "temperature": 0.0,
        "soc_model": "69",  # Snapdragon 8 Gen 3
        "hexagon_version": "v79",
        "model_name": "gemma-3-1b-it",
        "dtype": "bfloat16"
    }
    
    config_path = gemma3_dir / "qnn_config.json"
    with open(config_path, 'w') as f:
        json.dump(qnn_config, f, indent=2)
    
    # Copy to Android assets
    shutil.copy2(config_path, context_assets / "qnn_config.json")
    
    print(f"✅ QNN config saved to: {config_path}")
    
    # Create context binaries info
    context_info = {
        "description": "Gemma-3-1B-IT Context Binaries for QNN",
        "prefill_context": "prefill_context.bin",
        "decode_context": "decode_context.bin",
        "hybrid_config": "hybrid_config.bin",
        "generated_by": "EdgeAI Gemma-3-1B-IT Integration",
        "model_info": {
            "name": "gemma-3-1b-it",
            "parameters": "1B",
            "architecture": "transformer",
            "vocab_size": 256000,
            "max_seq_len": 32768,
            "dtype": "bfloat16"
        }
    }
    
    context_info_path = context_assets / "context_info.json"
    with open(context_info_path, 'w') as f:
        json.dump(context_info, f, indent=2)
    
    print(f"✅ Context info saved to: {context_info_path}")
    
    # Create a placeholder .pte file (for now)
    pte_path = model_assets / "gemma3-1b.pte"
    with open(pte_path, 'wb') as f:
        f.write(b"GEMMA3_1B_PLACEHOLDER_MODEL_FILE")
    
    print(f"✅ Placeholder .pte file created: {pte_path}")
    
    # Summary
    print("\n🎉 Gemma-3-1B-IT setup completed!")
    print("\n📁 Files created:")
    print(f"  📦 Model files: {model_assets}")
    print(f"  🔤 Tokenizer files: {tokenizer_assets}")
    print(f"  📚 Context files: {context_assets}")
    print(f"  📋 Model info: {info_path}")
    print(f"  ⚙️ QNN config: {config_path}")
    
    print("\n📱 Android assets updated:")
    print("  ✅ app/src/main/assets/gemma3_models/")
    print("  ✅ app/src/main/assets/gemma3_tokenizers/")
    print("  ✅ app/src/main/assets/gemma3_context_binaries/")
    
    print("\n🚀 Next steps:")
    print("  1. Build the Android app")
    print("  2. Test with real Gemma-3-1B-IT model")
    print("  3. Implement actual ExecuTorch conversion")
    print("  4. Configure QNN backend for your device")
    
    return True

if __name__ == "__main__":
    setup_gemma3_model()
