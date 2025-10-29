package com.example.edgeai.ml

import android.content.Context
import android.util.Log
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import org.json.JSONObject
import org.json.JSONArray

/**
 * ExecutorTorch Gemma3-1B Inference Engine for EdgeAI
 * Handles Gemma3-1B model loading and inference using ExecutorTorch Qualcomm QNN backend
 * Based on official ExecutorTorch Qualcomm integration patterns for Gemma3-1B
 * 
 * References:
 * - https://github.com/pytorch/executorch/tree/main/examples/qualcomm/oss_scripts/llama#gemma3-1b
 * - https://docs.pytorch.org/executorch/stable/backends-qualcomm.html
 * 
 * Model Configuration:
 * - Model: Gemma3-1B with hybrid mode (prefill + decode)
 * - Tokenizer: Gemma3-1B specific tokenizer
 * - Params: Mobile-optimized for Samsung S25 Ultra
 * - Backend: Qualcomm AI Engine Direct (QNN) via ExecutorTorch
 * - Mode: Hybrid (prefill_ar_len=128, max_seq_len=1024)
 */
class ExecutorTorchGemma3(private val context: Context) {

    companion object {
        private const val TAG = "ExecutorTorchGemma3"
        
        // Load native library
        init {
            try {
                System.loadLibrary("edgeai")
                Log.i(TAG, "✅ Native library loaded successfully")
            } catch (e: UnsatisfiedLinkError) {
                Log.e(TAG, "❌ Failed to load native library", e)
            }
        }
    }

    // JNI declarations for Gemma3-1B ExecuTorch integration
    external fun nativeInitializeGemma3(
        modelPath: String,
        tokenizerPath: String,
        contextBinariesPath: String
    ): Boolean

    external fun nativeGenerateGemma3Response(
        prompt: String,
        maxTokens: Int,
        temperature: Float
    ): String

    external fun nativeIsGemma3Initialized(): Boolean
    
    external fun nativeGetGemma3ModelInfo(): String

    /**
     * Initialize Gemma3-1B model with ExecuTorch + QNN backend
     * 
     * @param modelPath Path to Gemma3-1B .pte model file
     * @param tokenizerPath Path to Gemma3-1B tokenizer
     * @param contextBinariesPath Path to context binaries for hybrid mode
     * @return true if initialization successful
     */
    fun initializeGemma3(
        modelPath: String = "gemma3_models",
        tokenizerPath: String = "gemma3_tokenizers", 
        contextBinariesPath: String = "gemma3_context_binaries"
    ): Boolean {
        Log.i(TAG, "🚀 Initializing Gemma3-1B with ExecuTorch + QNN")
        
        try {
            // Copy model files to internal storage if needed
            val internalModelPath = copyModelDirectoryToInternalStorage(modelPath, "gemma3_models")
            val internalTokenizerPath = copyModelDirectoryToInternalStorage(tokenizerPath, "gemma3_tokenizers")
            val internalContextPath = copyModelDirectoryToInternalStorage(contextBinariesPath, "gemma3_context_binaries")
            
            Log.i(TAG, "📁 Model paths:")
            Log.i(TAG, "   - Model: $internalModelPath")
            Log.i(TAG, "   - Tokenizer: $internalTokenizerPath")
            Log.i(TAG, "   - Context Binaries: $internalContextPath")
            
            // Initialize native Gemma3-1B inference
            val success = nativeInitializeGemma3(internalModelPath, internalTokenizerPath, internalContextPath)
            
            if (success) {
                Log.i(TAG, "✅ Gemma3-1B ExecuTorch + QNN initialization successful")
                Log.i(TAG, "📊 Model Configuration:")
                Log.i(TAG, "   - Model: Gemma3-1B (1B parameters)")
                Log.i(TAG, "   - Backend: ExecuTorch + Qualcomm QNN")
                Log.i(TAG, "   - Mode: Hybrid (prefill + decode)")
                Log.i(TAG, "   - Max Sequence Length: 1024")
                Log.i(TAG, "   - Prefill AR Length: 128")
                Log.i(TAG, "   - Temperature: 0.0")
                
                // Log model info
                val modelInfo = nativeGetGemma3ModelInfo()
                Log.i(TAG, "📋 Model Info:\n$modelInfo")
                
            } else {
                Log.e(TAG, "❌ Gemma3-1B initialization failed")
            }
            
            return success

        } catch (e: Exception) {
            Log.e(TAG, "❌ Exception during Gemma3-1B initialization", e)
            return false
        }
    }

    /**
     * Generate response using Gemma3-1B model
     * 
     * @param prompt Input prompt text
     * @param maxTokens Maximum number of tokens to generate
     * @param temperature Sampling temperature (0.0 = deterministic)
     * @return Generated response text
     */
    fun generateResponse(
        prompt: String,
        maxTokens: Int = 100,
        temperature: Float = 0.0f
    ): String {
        Log.i(TAG, "🤖 Generating response with Gemma3-1B")
        Log.i(TAG, "📝 Prompt: $prompt")
        Log.i(TAG, "🎯 Max tokens: $maxTokens, Temperature: $temperature")
        
        try {
            if (!nativeIsGemma3Initialized()) {
                Log.w(TAG, "⚠️ Gemma3-1B not initialized, initializing now...")
                if (!initializeGemma3()) {
                    return "Error: Failed to initialize Gemma3-1B model"
                }
            }
            
            // Generate response using native Gemma3-1B inference
            val response = nativeGenerateGemma3Response(prompt, maxTokens, temperature)
            
            Log.i(TAG, "✅ Gemma3-1B response generated successfully")
            Log.i(TAG, "📄 Response: $response")
            
            return response
            
        } catch (e: Exception) {
            Log.e(TAG, "❌ Exception during Gemma3-1B inference", e)
            return "Error: Failed to generate response with Gemma3-1B"
        }
    }

    /**
     * Check if Gemma3-1B model is initialized
     * 
     * @return true if model is ready for inference
     */
    fun isInitialized(): Boolean {
        return try {
            nativeIsGemma3Initialized()
        } catch (e: Exception) {
            Log.e(TAG, "❌ Exception checking Gemma3-1B initialization status", e)
            false
        }
    }

    /**
     * Get detailed model information
     * 
     * @return Model information string
     */
    fun getModelInfo(): String {
        return try {
            if (nativeIsGemma3Initialized()) {
                nativeGetGemma3ModelInfo()
            } else {
                "Gemma3-1B: Not initialized"
            }
        } catch (e: Exception) {
            Log.e(TAG, "❌ Exception getting Gemma3-1B model info", e)
            "Error: Failed to get Gemma3-1B model information"
        }
    }

    /**
     * Copy model directory to internal storage
     * 
     * @param sourcePath Source directory path
     * @param targetDirName Target directory name
     * @return Internal storage path
     */
    private fun copyModelDirectoryToInternalStorage(sourcePath: String, targetDirName: String): String {
        val internalDir = File(context.filesDir, targetDirName)
        if (!internalDir.exists()) {
            internalDir.mkdirs()
        }
        
        // Copy all files from assets to internal storage
        try {
            val assetManager = context.assets
            val assetFiles = assetManager.list(sourcePath) ?: emptyArray()
            
            for (fileName in assetFiles) {
                val inputStream = assetManager.open("$sourcePath/$fileName")
                val outputFile = File(internalDir, fileName)
                val outputStream = FileOutputStream(outputFile)
                
                inputStream.copyTo(outputStream)
                inputStream.close()
                outputStream.close()
                
                Log.i(TAG, "📁 Copied $fileName to ${outputFile.absolutePath}")
            }
            
            Log.i(TAG, "✅ Directory copied: $sourcePath -> ${internalDir.absolutePath}")
            
        } catch (e: IOException) {
            Log.e(TAG, "❌ Failed to copy directory", e)
        }
        
        return internalDir.absolutePath
    }

    /**
     * Test Gemma3-1B inference with sample prompts
     * 
     * @return Test results
     */
    fun runGemma3Tests(): Map<String, String> {
        Log.i(TAG, "🧪 Running Gemma3-1B inference tests")
        
        val testResults = mutableMapOf<String, String>()
        
        val testPrompts = mapOf(
            "Python Learning" to "I would like to learn python, could you teach me with a simple example?",
            "General Question" to "Hello! How are you today?",
            "Programming Help" to "Can you help me write a simple function in Python?",
            "Technical Question" to "What is machine learning?"
        )
        
        for ((testName, prompt) in testPrompts) {
            try {
                Log.i(TAG, "🔬 Testing: $testName")
                val response = generateResponse(prompt, 50, 0.0f)
                testResults[testName] = response
                Log.i(TAG, "✅ $testName test completed")
                
            } catch (e: Exception) {
                Log.e(TAG, "❌ $testName test failed", e)
                testResults[testName] = "Error: ${e.message}"
            }
        }
        
        Log.i(TAG, "🏁 Gemma3-1B tests completed")
        return testResults
    }

    /**
     * Get Gemma3-1B model statistics
     * 
     * @return Model statistics
     */
    fun getModelStats(): Map<String, Any> {
        return mapOf(
            "model_name" to "Gemma3-1B",
            "parameters" to "1B",
            "backend" to "ExecuTorch + Qualcomm QNN",
            "mode" to "Hybrid (prefill + decode)",
            "max_seq_len" to 1024,
            "prefill_ar_len" to 128,
            "temperature" to 0.0f,
            "initialized" to isInitialized(),
            "model_info" to getModelInfo()
        )
    }
}