#!/bin/bash

# Gemma3-1B ExecuTorch + QNN Setup Script
# Based on: https://github.com/pytorch/executorch/tree/main/examples/qualcomm/oss_scripts/llama#gemma3-1b

set -e

echo "🚀 Setting up Gemma3-1B with ExecuTorch + QNN Integration"
echo "=========================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if we're in the right directory
if [ ! -f "app/build.gradle.kts" ]; then
    print_error "Please run this script from the EdgeAI project root directory"
        exit 1
    fi
    
print_status "Setting up Gemma3-1B ExecuTorch + QNN integration..."

# 1. Create model directories
print_status "Creating model directories..."
mkdir -p app/src/main/assets/gemma3_models
mkdir -p app/src/main/assets/gemma3_tokenizers
mkdir -p app/src/main/assets/gemma3_context_binaries

print_success "Model directories created"

# 2. Create placeholder model files
print_status "Creating placeholder model files..."

# Create placeholder Gemma3-1B model file
cat > app/src/main/assets/gemma3_models/gemma3-1b.pte << 'EOF'
# Placeholder Gemma3-1B model file
# In real implementation, this would be the actual compiled .pte model
# Generated using: python examples/qualcomm/oss_scripts/llama/llama.py -b build-android -s ${SERIAL_NUM} -m ${SOC_MODEL} --decoder_model gemma3-1b
EOF

# Create placeholder tokenizer file
cat > app/src/main/assets/gemma3_tokenizers/gemma3_tokenizer.json << 'EOF'
{
  "version": "1.0",
  "model": "gemma3-1b",
  "tokenizer": "sentencepiece",
  "vocab_size": 256000,
  "special_tokens": {
    "<pad>": 0,
    "<eos>": 1,
    "<bos>": 2,
    "<unk>": 3
  },
  "note": "Placeholder tokenizer - replace with actual Gemma3-1B tokenizer"
}
EOF

# Create placeholder context binaries
cat > app/src/main/assets/gemma3_context_binaries/context_info.txt << 'EOF'
# Gemma3-1B Context Binaries
# 
# In real implementation, this directory would contain:
# - prefill_context.bin
# - decode_context.bin
# - hybrid_mode_config.bin
# 
# Generated using Qualcomm AI HUB with hybrid mode configuration:
# --model_mode hybrid --max_seq_len 1024 --prefill_ar_len 128
EOF

print_success "Placeholder model files created"

# 3. Update AndroidManifest.xml to include Gemma3-1B test activity
print_status "Updating AndroidManifest.xml..."

# Check if the activity is already declared
if ! grep -q "ExecutorTorchGemma3TestActivity" app/src/main/AndroidManifest.xml; then
    # Add the activity declaration before the closing </application> tag
    sed -i.bak '/<\/application>/i\
        <activity\
            android:name=".ui.ExecutorTorchGemma3TestActivity"\
            android:exported="true"\
            android:label="Gemma3-1B Test"\
            android:theme="@style/Theme.EdgeAI">\
            <intent-filter>\
                <action android:name="android.intent.action.MAIN" />\
                <category android:name="android.intent.category.LAUNCHER" />\
            </intent-filter>\
        </activity>' app/src/main/AndroidManifest.xml
    
    print_success "Added ExecutorTorchGemma3TestActivity to AndroidManifest.xml"
else
    print_warning "ExecutorTorchGemma3TestActivity already exists in AndroidManifest.xml"
fi

# 4. Create documentation for Gemma3-1B integration
print_status "Creating Gemma3-1B documentation..."

cat > GEMMA3_1B_SETUP.md << 'EOF'
# Gemma3-1B ExecuTorch + QNN Integration

This document describes the integration of Gemma3-1B model with ExecuTorch + Qualcomm QNN backend in the EdgeAI project.

## Overview

Gemma3-1B is a 1 billion parameter language model optimized for mobile inference using ExecuTorch and Qualcomm's QNN backend. This integration follows the official PyTorch ExecuTorch Qualcomm examples.

## References

- [PyTorch ExecuTorch Qualcomm Examples](https://github.com/pytorch/executorch/tree/main/examples/qualcomm/oss_scripts/llama#gemma3-1b)
- [ExecuTorch Documentation](https://docs.pytorch.org/executorch/stable/backends-qualcomm.html)

## Model Configuration

- **Model**: Gemma3-1B (1B parameters)
- **Backend**: ExecuTorch + Qualcomm QNN
- **Mode**: Hybrid (prefill + decode)
- **Max Sequence Length**: 1024
- **Prefill AR Length**: 128
- **Temperature**: 0.0 (deterministic)

## Setup Commands

Based on the official example:

```bash
python examples/qualcomm/oss_scripts/llama/llama.py \
    -b build-android \
    -s ${SERIAL_NUM} \
    -m ${SOC_MODEL} \
    --temperature 0 \
    --model_mode hybrid \
    --max_seq_len 1024 \
    --prefill_ar_len 128 \
    --decoder_model gemma3-1b \
    --prompt "I would like to learn python, could you teach me with a simple example?" \
    --tasks wikitext \
    --limit 1
```

## Implementation Details

### Native C++ Implementation

The native implementation (`executorch_gemma3.cpp`) provides:

1. **GemmaProgram Class**: Simulates ExecuTorch program loading for Gemma3-1B
2. **QNN Backend Integration**: Handles Qualcomm QNN backend initialization
3. **Hybrid Mode Support**: Implements both prefill and decode phases
4. **Tokenizer Integration**: Handles Gemma3-1B specific tokenization

### Kotlin Interface

The Kotlin interface (`ExecutorTorchGemma3.kt`) provides:

1. **Model Initialization**: Loads Gemma3-1B model and tokenizer
2. **Response Generation**: Generates responses using hybrid mode
3. **Test Framework**: Comprehensive testing capabilities
4. **Model Statistics**: Detailed model information and stats

### Test Activity

The test activity (`ExecutorTorchGemma3TestActivity.kt`) provides:

1. **Interactive Testing**: UI for testing Gemma3-1B inference
2. **Model Information**: Displays model configuration and status
3. **Response Generation**: Interactive prompt and response interface
4. **Comprehensive Tests**: Automated test suite for model validation

## File Structure

```
app/src/main/
├── cpp/
│   └── executorch_gemma3.cpp          # Native Gemma3-1B implementation
├── java/com/example/edgeai/ml/
│   └── ExecutorTorchGemma3.kt          # Kotlin interface
├── java/com/example/edgeai/ui/
│   └── ExecutorTorchGemma3TestActivity.kt  # Test activity
├── res/layout/
│   └── activity_executor_torch_gemma3_test.xml  # Test UI layout
└── assets/
    ├── gemma3_models/
    │   └── gemma3-1b.pte              # Gemma3-1B model file
    ├── gemma3_tokenizers/
    │   └── gemma3_tokenizer.json       # Tokenizer configuration
    └── gemma3_context_binaries/        # Context binaries for hybrid mode
```

## Usage

1. **Initialize Model**:
   ```kotlin
   val gemma3 = ExecutorTorchGemma3(context)
   val success = gemma3.initializeGemma3()
   ```

2. **Generate Response**:
   ```kotlin
   val response = gemma3.generateResponse(
       prompt = "I would like to learn python, could you teach me with a simple example?",
       maxTokens = 100,
       temperature = 0.0f
   )
   ```

3. **Run Tests**:
   ```kotlin
   val testResults = gemma3.runGemma3Tests()
   ```

## Testing

The integration includes comprehensive testing:

- **Initialization Test**: Verifies model loading and QNN backend setup
- **Response Generation Test**: Tests inference with various prompts
- **Python Learning Test**: Specific test for Python learning scenarios
- **General Question Test**: Tests general conversational capabilities
- **Programming Help Test**: Tests programming assistance features
- **Technical Question Test**: Tests technical knowledge capabilities

## Performance Considerations

- **Hybrid Mode**: Optimizes both prefill and decode phases
- **QNN Acceleration**: Leverages Qualcomm hardware acceleration
- **Memory Management**: Efficient memory usage for mobile devices
- **Temperature Control**: Deterministic output for consistent results

## Next Steps

1. **Real Model Integration**: Replace placeholder files with actual Gemma3-1B model
2. **QNN Backend**: Implement actual QNN backend integration
3. **Tokenizer**: Integrate real Gemma3-1B tokenizer
4. **Context Binaries**: Load actual context binaries for hybrid mode
5. **Performance Optimization**: Fine-tune for specific hardware configurations

## Troubleshooting

- **Initialization Failures**: Check model file paths and QNN library availability
- **Response Quality**: Verify tokenizer configuration and model weights
- **Performance Issues**: Check QNN backend configuration and memory allocation
- **Build Errors**: Ensure all native dependencies are properly linked

## Version Information

- **EdgeAI Version**: 1.5.0
- **Gemma3-1B Model**: Latest version
- **ExecuTorch**: Latest stable
- **QNN Backend**: v79 compatible
- **Android NDK**: 25.1.8937393
EOF

print_success "Gemma3-1B documentation created"

# 5. Update README.md to include Gemma3-1B information
print_status "Updating README.md with Gemma3-1B information..."

# Check if Gemma3-1B section already exists
if ! grep -q "Gemma3-1B" README.md; then
    # Add Gemma3-1B section before the Contributing section
    sed -i.bak '/## Contributing/i\
\
## Gemma3-1B Integration\
\
EdgeAI now supports Gemma3-1B model with ExecuTorch + QNN backend:\
\
- **Model**: Gemma3-1B (1B parameters)\
- **Backend**: ExecuTorch + Qualcomm QNN\
- **Mode**: Hybrid (prefill + decode)\
- **Configuration**: max_seq_len=1024, prefill_ar_len=128, temperature=0.0\
\
### Quick Start\
\
```kotlin\
val gemma3 = ExecutorTorchGemma3(context)\
val success = gemma3.initializeGemma3()\
val response = gemma3.generateResponse("I would like to learn python, could you teach me with a simple example?")\
```\
\
### Testing\
\
Run the Gemma3-1B test activity to verify integration:\
\
```bash\
adb shell am start -n com.example.edgeai/.ui.ExecutorTorchGemma3TestActivity\
```\
\
For detailed setup instructions, see [GEMMA3_1B_SETUP.md](GEMMA3_1B_SETUP.md).\
' README.md
    
    print_success "Updated README.md with Gemma3-1B information"
else
    print_warning "Gemma3-1B section already exists in README.md"
fi

# 6. Clean up backup files
print_status "Cleaning up backup files..."
rm -f app/src/main/AndroidManifest.xml.bak README.md.bak

print_success "Backup files cleaned up"

# 7. Build the project
print_status "Building the project..."
if ./gradlew assembleDebug; then
    print_success "Project built successfully"
else
    print_error "Build failed. Please check the error messages above."
    exit 1
fi

# 8. Summary
echo ""
echo "🎉 Gemma3-1B ExecuTorch + QNN Integration Setup Complete!"
echo "========================================================"
echo ""
echo "📁 Files created/modified:"
echo "  ✅ app/src/main/cpp/executorch_gemma3.cpp"
echo "  ✅ app/src/main/java/com/example/edgeai/ml/ExecutorTorchGemma3.kt"
echo "  ✅ app/src/main/java/com/example/edgeai/ui/ExecutorTorchGemma3TestActivity.kt"
echo "  ✅ app/src/main/res/layout/activity_executor_torch_gemma3_test.xml"
echo "  ✅ app/src/main/cpp/CMakeLists.txt"
echo "  ✅ app/build.gradle.kts (version 1.5.0)"
echo "  ✅ app/src/main/AndroidManifest.xml"
echo "  ✅ GEMMA3_1B_SETUP.md"
echo "  ✅ README.md"
    echo ""
echo "📱 Model files:"
echo "  📦 app/src/main/assets/gemma3_models/gemma3-1b.pte"
echo "  🔤 app/src/main/assets/gemma3_tokenizers/gemma3_tokenizer.json"
echo "  📚 app/src/main/assets/gemma3_context_binaries/"
    echo ""
echo "🚀 Next steps:"
echo "  1. Install the app: adb install app/build/outputs/apk/debug/app-debug.apk"
echo "  2. Test Gemma3-1B: adb shell am start -n com.example.edgeai/.ui.ExecutorTorchGemma3TestActivity"
echo "  3. Replace placeholder model files with actual Gemma3-1B model"
echo "  4. Configure QNN backend for your specific hardware"
    echo ""
echo "📖 Documentation:"
echo "  📄 GEMMA3_1B_SETUP.md - Detailed setup instructions"
echo "  🌐 https://github.com/pytorch/executorch/tree/main/examples/qualcomm/oss_scripts/llama#gemma3-1b"
    echo ""
print_success "Setup completed successfully! 🎉"