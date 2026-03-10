## 🐛 Bug Report: LLaMA Model Generating Gibberish Output (tokenXXXX) on Android with QNN Backend

### **Environment**
- **Platform**: Android (Qualcomm Snapdragon device)
- **ExecuTorch Version**: Latest (as of Jan 2026)
- **Model**: TinyLLaMA (stories110M.pt) / LLaMA 3.2 1B
- **Backend**: Qualcomm QNN (v73/v79)
- **Hardware**: Qualcomm NPU (Hexagon Tensor Processor)
- **Device**: Snapdragon SoC with HTP support

### **Description**
When running LLaMA model inference on Android using ExecuTorch with Qualcomm QNN backend, the model generates gibberish output consisting primarily of raw token IDs (e.g., "token1457", "token1940") instead of meaningful text. The output lacks proper sentence structure and coherence despite the model successfully loading and executing.

### **Expected Behavior**
```
Input: "What is machine learning?"
Expected Output: "Machine learning is a branch of artificial intelligence that enables computers to learn from data without being explicitly programmed..."
```

### **Actual Behavior**
```
Input: "What is machine learning?"
Actual Output: "massive token1457 token1940 token1424 token987 token845 token1690 unimportant token1940 token1104 token706 token721 guidelines scientific token1238 extraordinary lowering token1417 token1876..."
```

### **Steps to Reproduce**

1. **Setup ExecuTorch for Android**:
   ```bash
   git clone https://github.com/pytorch/executorch.git
   cd executorch
   ./install_requirements.sh
   ```

2. **Export LLaMA model with QNN backend**:
   ```bash
   python -m examples.portable.scripts.export_llama \
       --model_name tinyllama \
       --backend qnn \
       --output_dir ./compiled_models
   ```

3. **Integrate into Android project**:
   - Copy `.pte` model file to Android assets
   - Load model using ExecuTorch Android runtime
   - Initialize QNN backend with HTP acceleration

4. **Run inference**:
   ```kotlin
   val response = llamaInference.generateText(
       prompt = "What is machine learning?",
       maxTokens = 128,
       temperature = 0.8f
   )
   ```

5. **Observe output**: Gibberish text with "tokenXXXX" entries

### **Logs and Evidence**

**Token Generation Logs**:
```
D LLaMAInference: ✅ Added word: token1457 (token: 1457)
D LLaMAInference: ✅ Added word: token1940 (token: 1940)
D LLaMAInference: ✅ Added word: token1424 (token: 1424)
D LLaMAInference: ✅ Added word: token987 (token: 987)
D LLaMAInference: ✅ Added word: token845 (token: 845)
```

**Input Tokenization Logs**:
```
D LLaMAInference: 🔤 Input tokens: [1, 0, 9, 0, 0] (110 total)
D LLaMAInference: 🌡️ Using temperature: 0.9 for context: '<s> <pad> a <pad> <pad> about a <pad> learning to <pad>'
D LLaMAInference: 🎯 Selected token: 587 (unknown)
```

**Sample Output**:
```
Input: "Describe the process of photosynthesis."
Output: "token1292 token1128 plants token1027 sunlight dioxide dioxide plants token1450 carbon oxygen advancement glucose carbon token1574 token761 sunlight token1515 popular glucose token1697..."
```

### **Analysis**

#### **Issue 1: Tokenization Problems**
- Generated token IDs don't map to vocabulary entries
- Input being incorrectly tokenized as `<s> <pad> a <pad> <pad> about a <pad> learning to <pad>`
- ~60% of output tokens are placeholder "tokenXXXX" entries

#### **Issue 2: No Sentence Structure**
- Output consists of random words without grammatical structure
- No proper word ordering or sentence formation
- Missing articles, prepositions, and connectors

#### **Issue 3: Model Inference Issues**
- Appears to be falling back to random token generation
- Model weights may not be properly loaded or utilized
- QNN backend initialization succeeds but inference quality is poor

### **Debugging Attempts**

✅ **What Works**:
- Model loads successfully without crashes
- QNN backend initializes correctly
- Token generation executes without errors
- Context detection recognizes topic types (e.g., biology, energy)

❌ **What Doesn't Work**:
- Token-to-word mapping produces gibberish
- No coherent sentence generation
- Output quality unsuitable for any practical use

### **Questions**

1. **Is this a known issue with ExecuTorch LLaMA on Android/QNN?**
2. **Are there specific tokenizer requirements for Android deployment?**
3. **Should the tokenizer.model file be processed differently for mobile?**
4. **Are there known quantization issues that could cause this behavior?**
5. **Is there example code for proper LLaMA tokenization on Android?**

### **Code Repository**

Full project with debugging details available at:
- **Repository**: https://github.com/carrycooldude/EdgeAIApp-ExecuTorch
- **Detailed Report**: https://github.com/carrycooldude/EdgeAIApp-ExecuTorch/blob/main/debugging/debugging-report.md
- **APK Demo**: [Google Drive Link](https://drive.google.com/file/d/1hCaTsCqftPwgebTbTJXo0skgvzkRGyy1/view?usp=sharing)

### **Screenshots/Videos**

**Issue 1: TokenXXXX Generation**
![TokenXXXX Generation Issue](https://github.com/user-attachments/assets/23b385a0-eae5-4724-b896-7a01f7f902a4)

**Issue 2: Unstructured Output**
![Unstructured Output](https://github.com/user-attachments/assets/8dff9dd8-e340-4625-aa3b-6f7de8b50844)

**Issue 3: App Interface Demo**
https://github.com/user-attachments/assets/47c79b07-b16d-4a49-92f5-54ec31045ffa

### **Technical Details**

**Architecture**:
```
MainActivity (Kotlin) 
    ↓
LLaMAInference (Kotlin)
    ↓
QNNManager (JNI/C++)
    ↓
libQnnHtp.so (Qualcomm NPU)
```

**Key Files**:
- `app/src/main/java/com/example/edgeai/ml/LLaMAInference.kt`
- `app/src/main/java/com/example/edgeai/ml/TinyLLaMAInference.kt`
- `app/src/main/cpp/qnn_manager.cpp`
- `app/src/main/cpp/real_qnn_inference.cpp`

### **Request**

Could the ExecuTorch team provide:
1. Guidance on proper LLaMA tokenization for Android
2. Example code for Android LLaMA integration with QNN
3. Known issues or workarounds for this problem
4. Best practices for mobile LLaMA deployment

### **Additional Context**

- Using Qualcomm AI Hub context binaries (v79, SoC Model-69)
- Model files: `llama32_1b_qnn.pte`, `tokenizer.model`
- Both TinyLLaMA and LLaMA 3.2 1B exhibit same issues
- CLIP model works perfectly in the same app (different architecture)

---

**Labels**: `bug`, `android`, `qualcomm`, `llama`, `tokenization`, `qnn-backend`
**Priority**: High (blocks production deployment)
