# Quick Testing Guide for LLaMA 3.2 1B Integration

## Current Build Status

Building the app with LLaMA 3.2 1B integration...

## What Was Implemented

### 1. **Model Export Pipeline**
- `download_llama32_model.py` - Downloads LLaMA 3.2 1B from Hugging Face
- `export_llama32_qnn.py` - Exports to ExecuTorch .pte format with QNN backend
- `export_llama32_qnn.bat` - Windows automation script

### 2. **Native C++ Layer**
- `app/src/main/cpp/executorch_llama32_qnn.cpp` - Complete inference engine
- Tokenizer, sampling, JNI bindings
- Updated CMakeLists.txt

### 3. **Kotlin Android Layer**
- `ExecutorTorchLlama32.kt` - Kotlin wrapper with coroutines
- `MainViewModel.kt` - Multi-model state management
- `MainActivityEnhanced.kt` - UI with model selection

## Quick Test Steps

### Option 1: Test with Placeholder (No Model Export Needed)

The implementation includes placeholder code that will work without actual model files:

```bash
# 1. Build (running now...)
./gradlew assembleDebug

# 2. Install
adb install app/build/outputs/apk/debug/app-debug.apk

# 3. Test
# - Open app
# - Select "LLaMA 3.2 1B" from spinner
# - Enter a prompt: "What is machine learning?"
# - Tap "Generate"
# - You'll see simulated output (no real model needed)
```

### Option 2: Test with Real Model

To test with the actual LLaMA 3.2 1B model:

```bash
# 1. Export model (requires ExecuTorch + QNN setup)
python export_llama32_qnn.py --model-dir models/llama-3.2-1b --quantize 16a4w

# 2. Push to device
adb shell mkdir -p /sdcard/EdgeAI/models/llama32
adb push output/llama32_qnn/llama32_1b_qnn.pte /sdcard/EdgeAI/models/llama32/
adb push output/llama32_qnn/tokenizer/tokenizer.model /sdcard/EdgeAI/models/llama32/

# 3. Grant permissions
adb shell pm grant com.example.edgeai android.permission.READ_EXTERNAL_STORAGE
adb shell pm grant com.example.edgeai android.permission.WRITE_EXTERNAL_STORAGE

# 4. Install and run
adb install app/build/outputs/apk/debug/app-debug.apk
```

## Testing Checklist

### Build Verification
- [ ] App builds without errors
- [ ] Native library includes llama32 symbols
- [ ] No missing dependencies

### Functionality Testing
- [ ] App launches successfully
- [ ] Model selection spinner works
- [ ] LLaMA 3.2 UI shows correctly
- [ ] Can enter prompts
- [ ] Generate button works (simulated or real)
- [ ] Response displays in TextView
- [ ] CLIP model still works

### Performance Testing
- [ ] Monitor logcat for initialization logs
- [ ] Check memory usage
- [ ] Verify inference speed (if using real model)
- [ ] No crashes or memory leaks

## Monitoring Logs

```bash
# Watch all EdgeAI logs
adb logcat | grep -E "(ExecutorTorchLlama32|MainViewModel|EdgeAI)"

# Watch LLaMA 32 specific logs
adb logcat | grep "ExecutorTorchLlama32"

# Clear logs and watch
adb logcat -c && adb logcat | grep "EdgeAI"
```

## Expected Log Output

### On App Launch:
```
I MainViewModel: 🚀 Initializing EdgeAI models...
I MainViewModel: Initializing LLaMA 3.2 1B...
I ExecutorTorchLlama32: 🚀 Initializing LLaMA 3.2 1B with ExecuTorch + QNN
I ExecutorTorchLlama32: Step 1: Loading tokenizer...
I ExecutorTorchLlama32: ✅ Tokenizer initialized (vocab_size=128256)
I ExecutorTorchLlama32: Step 2: Loading ExecuTorch model...
I ExecutorTorchLlama32: ✅ ExecuTorch model loaded
I ExecutorTorchLlama32: ✅ LLaMA 3.2 1B initialization complete!
```

### On Text Generation:
```
I ExecutorTorchLlama32: 🎯 Generating response for prompt: "What is machine learning?"
I ExecutorTorchLlama32: 📝 Tokenized input: X tokens
I ExecutorTorchLlama32: ✅ Generation complete!
I ExecutorTorchLlama32:    Generated X tokens in Yms (Z tokens/sec)
```

## Troubleshooting Quick Fixes

### Build Errors

**Issue**: "Cannot find ExecutorTorchLlama32"
```bash
# Check file exists
ls app/src/main/java/com/example/edgeai/ml/ExecutorTorchLlama32.kt

# Clean and rebuild
./gradlew clean build
```

**Issue**: CMake errors
```bash
# Check CMakeLists.txt includes the new file
cat app/src/main/cpp/CMakeLists.txt | grep llama32
```

### Runtime Errors

**Issue**: "Model not initialized"
- Check if model files exist on device
- Verify permissions granted
- Check logcat for initialization errors

**Issue**: App crashes on launch
- Check logcat for stack trace
- Verify native library loaded correctly

## Next Steps After Successful Build

1. **Test Placeholder Mode First**
   - Verify UI works
   - Check button interactions
   - Confirm no crashes

2. **Export Real Model**
   - Set up ExecuTorch environment
   - Run export scripts
   - Test with actual model

3. **Performance Benchmarking**
   - Measure tokens/sec
   - Monitor memory usage
   - Test on different devices

4. **UI Refinements**
   - Add streaming text display
   - Improve progress indicators
   - Add model switching

## Files Created Summary

```
EdgeAI/
├── download_llama32_model.py          ✅
├── export_llama32_qnn.py              ✅
├── export_llama32_qnn.bat             ✅
├── LLAMA32_SETUP.md                   ✅
├── app/src/main/
│   ├── cpp/
│   │   ├── executorch_llama32_qnn.cpp ✅
│   │   └── CMakeLists.txt             ✅ (updated)
│   └── java/com/example/edgeai/
│       ├── ml/ExecutorTorchLlama32.kt ✅
│       ├── MainActivityEnhanced.kt    ✅
│       └── ui/theme/MainViewModel.kt  ✅ (updated)
```

## Documentation

- Full setup guide: [LLAMA32_SETUP.md](LLAMA32_SETUP.md)
- Implementation details: [walkthrough.md](C:\Users\rawat\.gemini\antigravity\brain\fbb58df3-b3b8-4d46-bf80-1c427d2a4ccd\walkthrough.md)
- Task checklist: [task.md](C:\Users\rawat\.gemini\antigravity\brain\fbb58df3-b3b8-4d46-bf80-1c427d2a4ccd\task.md)
