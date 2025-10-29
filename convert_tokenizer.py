#!/usr/bin/env python3
"""
Convert tokenizer.model to tokenizer.bin following ExecutorTorch patterns
Based on: https://github.com/pytorch/executorch/tree/a1652f97b721dccc4f1f2585d3e1f15a2306e8d0/examples/qualcomm/oss_scripts/llama
"""

import os
import sys
import argparse
import struct
from pathlib import Path

def convert_tokenizer(tokenizer_model_path: str, tokenizer_bin_path: str, vocab_size: int) -> bool:
    print("🔄 Converting tokenizer.model to tokenizer.bin...")

    try:
        if not os.path.exists(tokenizer_model_path):
            print(f"❌ Tokenizer model not found: {tokenizer_model_path}")
            return False

        with open(tokenizer_model_path, 'rb') as f:
            tokenizer_data = f.read()

        print(f"📦 Read tokenizer.model: {len(tokenizer_data)} bytes")

        os.makedirs(os.path.dirname(tokenizer_bin_path), exist_ok=True)
        with open(tokenizer_bin_path, 'wb') as f:
            f.write(b'SPM\x00')  # SentencePiece Model magic
            f.write(struct.pack('<I', vocab_size))  # vocab_size header required by QNN pipeline
            f.write(tokenizer_data)

        print(f"✅ Created tokenizer.bin: {os.path.getsize(tokenizer_bin_path)} bytes (vocab_size={vocab_size})")
        return True

    except Exception as e:
        print(f"❌ Error converting tokenizer: {e}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert tokenizer.model to tokenizer.bin for ExecuTorch/QNN")
    parser.add_argument("--model", default="app/src/main/assets/models/tokenizer.model", help="Path to tokenizer.model")
    parser.add_argument("--out", default="app/src/main/assets/models/tokenizer.bin", help="Output tokenizer.bin path")
    parser.add_argument("--vocab_size", type=int, default=128256, help="Vocabulary size header to write (Llama 3.2 1B uses 128256)")
    args = parser.parse_args()

    success = convert_tokenizer(args.model, args.out, args.vocab_size)
    if success:
        print("🎉 Tokenizer conversion completed successfully!")
    else:
        print("💥 Tokenizer conversion failed!")
        sys.exit(1)
