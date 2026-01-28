# EdgeAI - CLIP with ExecuTorch + QNN

[![Version](https://img.shields.io/badge/version-1.4.0-blue.svg)](https://github.com/carrycooldude/EdgeAIApp-ExecuTorch)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Android](https://img.shields.io/badge/platform-Android-green.svg)](https://developer.android.com)
[![ExecuTorch](https://img.shields.io/badge/ExecuTorch-0.7.0-orange.svg)](https://github.com/pytorch/executorch)
[![QNN](https://img.shields.io/badge/QNN-v79-red.svg)](https://developer.qualcomm.com/software/ai-stack)

> **On-device CLIP model on Android with ExecuTorch + Qualcomm QNN backend for zero-shot image classification and vision-language tasks**

## **What's New in v1.4.0**

- 🆕 **CLIP Model Support**: Full integration of OpenAI's CLIP for vision-language understanding
- 🖼️ **Zero-Shot Classification**: Image classification with natural language queries
- 🚀 **Hardware Acceleration**: Optimized inference with Qualcomm QNN backend
- 📸 **Camera Integration**: Capture images directly from camera for real-time inference
- 🔧 **Comprehensive Documentation**: Detailed setup guides and troubleshooting
- 🎯 **Production Ready**: Robust error handling and memory management


##  **Table of Contents**

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [Documentation](#documentation)
- [Setup Guide](#setup-guide)
- [Technical Details](#technical-details)
- [Contributing](#contributing)
- [License](#license)

##  **Overview**

EdgeAI is an Android application showcasing **on-device CLIP model inference** using ExecuTorch with Qualcomm QNN backend. This implementation demonstrates real multimodal AI inference with actual trained models and hardware acceleration for vision-language tasks.

### **Supported Models**

| **Model** | **Size** | **Use Case** | **Status** |
|-----------|----------|--------------|------------|
| **CLIP** | ~400MB | Zero-shot image classification, Image-text matching, Visual Q&A | ✅ Full support |

### **Key Capabilities**

- ✅ **Real Model Inference**: Actual trained CLIP model, not simulations
- ✅ **Hardware Acceleration**: Qualcomm HTP/DSP via QNN backend
- ✅ **Zero-Shot Learning**: Classify images without predefined categories
- ✅ **Vision-Language Understanding**: Match images with natural language descriptions
- ✅ **Production Ready**: Proper error handling and resource management

## **Features**

### **Core Features**
- 🤖 **CLIP Vision-Language Model**: OpenAI's CLIP for multimodal understanding
- �️ **Zero-Shot Classification**: Classify images using natural language without training
- 📸 **Camera Integration**: Capture photos directly from device camera
- � **Image-Text Matching**: Match images with text descriptions and queries
- ⚡ **Hardware Acceleration**: Qualcomm HTP/DSP acceleration via QNN
- 📱 **Android Native**: Optimized for mobile devices
- 🎯 **Real-time Inference**: Fast vision-language processing

### **Technical Features**
- ⚙️ **Context Binary Support**: v79/SoC Model-69 compatibility
- 🚀 **Optimized Performance**: ExecuTorch optimizations + QNN acceleration
- 💾 **Efficient Model Loading**: External storage for large models
- ⚡ **Real-time Inference**: Fast multimodal response generation
- 🛠️ **Developer Friendly**: Clean API and comprehensive documentation

## **Architecture**

### **High-Level Architecture**

```
+-----------------+     +-------------------+     +-----------------+
|   Android App   |     |   ExecuTorch     |     |   Qualcomm QNN  |
|                 |     |                   |     |                 |
|  +-----------+  | <-> |  +-------------+ | <-> |  +-----------+  |
|  | Kotlin UI |  |     |  | Runtime     | |     |  | HTP/DSP   |  |
|  | Camera    |  |     |  | (.pte model)| |     |  | Backend   |  |
|  +-----------+  | <-> |  +-------------+ | <-> |  +-----------+  |
|  +-----------+  |     |  | CLIP        | |     |  | Context   |  |
|  | JNI Layer |  |     |  | Text/Image  | |     |  | Binaries  |  |
|  +-----------+  |     |  | Encoders    | |     |  +-----------+  |
|                 |     |  +-------------+ |     |                 |
+-----------------+     +-------------------+     +-----------------+
```

### **Implementation Layers**

1. **Android UI Layer**: Kotlin-based user interface with camera integration
2. **JNI Bridge**: Communication between Kotlin and C++
3. **ExecuTorch Runtime**: CLIP model execution and management
4. **QNN Backend**: Hardware acceleration layer
5. **Model Layer**: CLIP vision and text encoders with real weights

## **Quick Start**

### **Prerequisites**

- Android Studio Arctic Fox or later
- Android NDK r25 or later
- Qualcomm device with HTP/DSP support (e.g., Snapdragon 8 Gen 2/3, Snapdragon 8 Elite)
- ExecuTorch 0.7.0+
- QNN SDK v79+

### **Installation**

1. **Clone the repository**
   ```bash
   git clone https://github.com/carrycooldude/EdgeAIApp-ExecuTorch.git
   cd EdgeAIApp-ExecuTorch
   ```

2. **Download CLIP Model**
   ```bash
   # Download the CLIP model using the provided script
   python download_clip_model.py
   ```

3. **Build and install**
   ```bash
   .\gradlew assembleDebug
   adb install app\build\outputs\apk\debug\app-debug.apk
   ```

4. **Grant permissions**
   - Allow camera and storage permissions when prompted

### **Usage**

1. Launch the EdgeAI app on your device
2. Tap "Take Photo" to capture an image or "Select Image" from gallery
3. Enter a question or description about the image (e.g., "What is in this image?")
4. Tap "Analyze Image" to run CLIP inference
5. View zero-shot classification results and similarity scores!

## **Documentation**

### **Technical Documentation**
- [Real ExecuTorch + QNN Integration](docs/technical/REAL_EXECUTORCH_QNN_INTEGRATION.md)
- [Implementation Analysis](docs/technical/IMPLEMENTATION_ANALYSIS.md)
- [Architecture Overview](docs/technical/ARCHITECTURE.md)

### **Setup Guides**
- [Qualcomm AI HUB Setup](docs/setup/QUALCOMM_AIHUB_SETUP.md)
- [ExecuTorch Configuration](docs/setup/EXECUTORCH_SETUP.md)
- [Android Development Setup](docs/setup/ANDROID_SETUP.md)

### **Release Notes**
- [v1.3.0 Release Notes](docs/releases/RELEASE_NOTES_v1.3.0.md)
- [v1.2.0 Release Notes](docs/releases/RELEASE_NOTES_v1.2.0.md)
- [v1.1.0 Release Notes](docs/releases/RELEASE_NOTES_v1.1.0.md)
- [v1.0.0 Release Notes](docs/releases/RELEASE_NOTES_v1.0.0.md)

##  **Setup Guide**

### **1. ExecuTorch Setup**

```bash
# Clone ExecuTorch
git clone https://github.com/pytorch/executorch.git
cd executorch

# Install dependencies
pip install -e .
pip install torch torchvision torchaudio
```

### **2. CLIP Model Download**

```bash
# Use the provided download script
python download_clip_model.py

# Or manually download from Hugging Face
# The CLIP model will be automatically converted to ExecuTorch format
```

### **3. Qualcomm QNN Setup**

```bash
# Download QNN SDK from Qualcomm
# Extract and set environment variables
export QNN_SDK_ROOT=/path/to/qnn-sdk
export LD_LIBRARY_PATH=$QNN_SDK_ROOT/lib/aarch64-android:$LD_LIBRARY_PATH
```

### **4. Model Compilation for QNN**

```bash
# Export CLIP model to ExecuTorch format with QNN backend
python -m executorch.examples.models.clip \
    --export \
    --model_name clip-vit-base-patch32 \
    --backend qnn
```

## **Technical Details**

### **Model Specifications**

- **Model**: OpenAI CLIP (ViT-B/32)
- **Vision Encoder**: Vision Transformer Base
- **Text Encoder**: Transformer-based text encoder
- **Patch Size**: 32x32
- **Image Resolution**: 224x224
- **Embedding Dimension**: 512
- **Vocabulary Size**: 49,408
- **Context Length**: 77 tokens

### **Hardware Requirements**

- **CPU**: ARM64-v8a (aarch64)
- **Accelerator**: Qualcomm HTP/DSP
- **Context Version**: v79
- **SoC Model**: 69 (Snapdragon 8 Gen 2/3/Elite)
- **Architecture**: aarch64-android
- **Minimum RAM**: 4GB
- **Recommended RAM**: 6GB+

### **Performance Metrics**

- **Inference Speed**: ~100-150ms per image-text pair
- **Memory Usage**: ~800MB RAM
- **Model Size**: ~400MB
- **Power Efficiency**: Optimized for mobile with QNN acceleration

## **Development**

### **Project Structure**

```
EdgeAI/
|-- app/                          # Android application
|   |-- src/main/
|   |   |-- cpp/                  # Native C++ implementation
|   |   |   |-- executorch_clip_proper.cpp  # CLIP ExecuTorch + QNN integration
|   |   |   |-- CMakeLists.txt    # Build configuration
|   |   |   `-- ...
|   |   |-- java/                 # Kotlin/Java code
|   |   |   |-- MainActivity.kt   # CLIP UI and inference
|   |   |   `-- ml/ExecutorTorchCLIP.kt  # CLIP model wrapper
|   |   `-- assets/               # Model files and resources
|-- docs/                         # Documentation
|   |-- technical/                # Technical documentation
|   |-- setup/                    # Setup guides
|   `-- releases/                 # Release notes
|-- scripts/                      # Build and setup scripts
|-- download_clip_model.py        # CLIP model download script
`-- README.md                     # This file
```

### **Building from Source**

```bash
# Debug build
.\gradlew assembleDebug

# Release build
.\gradlew assembleRelease

# Clean build
.\gradlew clean
```

### **Testing**

```bash
# Run tests
.\gradlew test

# Run Android tests
.\gradlew connectedAndroidTest
```

## **Contributing**

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### **Development Workflow**

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

### **Code Style**

- Follow Android Kotlin style guide
- Use meaningful variable names
- Add comments for complex logic
- Maintain consistent formatting

## **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## **Acknowledgments**

- [ExecuTorch](https://github.com/pytorch/executorch) - PyTorch's mobile inference framework
- [Qualcomm AI Stack](https://developer.qualcomm.com/software/ai-stack) - AI acceleration platform
- [OpenAI CLIP](https://github.com/openai/CLIP) - Contrastive Language-Image Pre-training model
- [Android NDK](https://developer.android.com/ndk) - Native development kit

## **Support**

- 📧 Email: rawatkari554@gmail.com
- 🐛 Issues: [GitHub Issues](https://github.com/carrycooldude/EdgeAIApp-ExecuTorch/issues)
- 💬 Discussions: [GitHub Discussions](https://github.com/carrycooldude/EdgeAIApp-ExecuTorch/discussions)

---

**Made with ❤️ for the AI community**