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

`ash
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
`

## Implementation Details

### Native C++ Implementation

The native implementation (executorch_gemma3.cpp) provides:

1. **GemmaProgram Class**: Simulates ExecuTorch program loading for Gemma3-1B
2. **QNN Backend Integration**: Handles Qualcomm QNN backend initialization
3. **Hybrid Mode Support**: Implements both prefill and decode phases
4. **Tokenizer Integration**: Handles Gemma3-1B specific tokenization

### Kotlin Interface

The Kotlin interface (ExecutorTorchGemma3.kt) provides:

1. **Model Initialization**: Loads Gemma3-1B model and tokenizer
2. **Response Generation**: Generates responses using hybrid mode
3. **Test Framework**: Comprehensive testing capabilities
4. **Model Statistics**: Detailed model information and stats

### Test Activity

The test activity (ExecutorTorchGemma3TestActivity.kt) provides:

1. **Interactive Testing**: UI for testing Gemma3-1B inference
2. **Model Information**: Displays model configuration and status
3. **Response Generation**: Interactive prompt and response interface
4. **Comprehensive Tests**: Automated test suite for model validation

## File Structure

`
app/src/main/
â”œâ”€â”€ cpp/
â”‚   â””â”€â”€ executorch_gemma3.cpp          # Native Gemma3-1B implementation
â”œâ”€â”€ java/com/example/edgeai/ml/
â”‚   â””â”€â”€ ExecutorTorchGemma3.kt          # Kotlin interface
â”œâ”€â”€ java/com/example/edgeai/ui/
â”‚   â””â”€â”€ ExecutorTorchGemma3TestActivity.kt  # Test activity
â”œâ”€â”€ res/layout/
â”‚   â””â”€â”€ activity_executor_torch_gemma3_test.xml  # Test UI layout
â””â”€â”€ assets/
    â”œâ”€â”€ gemma3_models/
    â”‚   â””â”€â”€ gemma3-1b.pte              # Gemma3-1B model file
    â”œâ”€â”€ gemma3_tokenizers/
    â”‚   â””â”€â”€ gemma3_tokenizer.json       # Tokenizer configuration
    â””â”€â”€ gemma3_context_binaries/        # Context binaries for hybrid mode
`

## Usage

1. **Initialize Model**:
   `kotlin
   val gemma3 = ExecutorTorchGemma3(context)
   val success = gemma3.initializeGemma3()
   `

2. **Generate Response**:
   `kotlin
   val response = gemma3.generateResponse(
       prompt = "I would like to learn python, could you teach me with a simple example?",
       maxTokens = 100,
       temperature = 0.0f
   )
   `

3. **Run Tests**:
   `kotlin
   val testResults = gemma3.runGemma3Tests()
   `

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
