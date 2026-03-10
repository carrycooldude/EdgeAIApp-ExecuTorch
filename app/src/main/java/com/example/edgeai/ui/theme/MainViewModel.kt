// Multi-Model ViewModel with LLaMA 3.2, Gemma3, and CLIP support
package com.example.edgeai.ui.theme

import android.content.Context
import android.graphics.Bitmap
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.example.edgeai.ml.ExecutorTorchCLIP
import com.example.edgeai.ml.ExecutorTorchLlama32
import kotlinx.coroutines.launch
import android.util.Log

// Model types
enum class ModelType {
    CLIP,
    LLAMA_32,
    GEMMA3
}

// UI State for multi-model support
data class MultiModelUiState(
    // Current model
    val currentModel: ModelType = ModelType.LLAMA_32,
    
    // LLaMA 3.2 state
    val llama32Prompt: String = "",
    val llama32Response: String = "",
    val llama32IsGenerating: Boolean = false,
    val llama32Initialized: Boolean = false,
    
    // CLIP state
    val clipImage: Bitmap? = null,
    val clipQuestion: String = "What is in this image?",
    val clipAnswer: String = "",
    val clipIsProcessing: Boolean = false,
    val clipInitialized: Boolean = false,
    
    // General state
    val initializationMessage: String = "Initializing models...",
    val errorMessage: String? = null
)

class MainViewModel : ViewModel() {
    var uiState by mutableStateOf(MultiModelUiState())
        private set
    
    private var clipInference: ExecutorTorchCLIP? = null
    private var llama32Inference: ExecutorTorchLlama32? = null
    
    companion object {
        private const val TAG = "MainViewModel"
    }
    
    fun initialize(context: Context) {
        Log.i(TAG, "🚀 Initializing EdgeAI models...")
        
        // Initialize models in background
        viewModelScope.launch {
            // Initialize LLaMA 3.2
            initializeLlama32(context)
            
            // Initialize CLIP
            initializeCLIP(context)
            
            updateInitMessage()
        }
    }
    
    private suspend fun initializeLlama32(context: Context) {
        try {
            Log.i(TAG, "Initializing LLaMA 3.2 1B...")
            llama32Inference = ExecutorTorchLlama32(context)
            val success = llama32Inference?.initialize() ?: false
            
            uiState = uiState.copy(
                llama32Initialized = success
            )
            
            if (success) {
                Log.i(TAG, "✅ LLaMA 3.2 1B initialized successfully")
            } else {
                Log.e(TAG, "❌ LLaMA 3.2 1B initialization failed")
            }
        } catch (e: Exception) {
            Log.e(TAG, "❌ LLaMA 3.2 initialization exception: ${e.message}", e)
            uiState = uiState.copy(
                llama32Initialized = false,
                errorMessage = "LLaMA 3.2 init failed: ${e.message}"
            )
        }
    }
    
    private suspend fun initializeCLIP(context: Context) {
        try {
            Log.i(TAG, "Initializing CLIP...")
            clipInference = ExecutorTorchCLIP(context)
            val success = clipInference?.initialize() ?: false
            
            uiState = uiState.copy(
                clipInitialized = success
            )
            
            if (success) {
                Log.i(TAG, "✅ CLIP initialized successfully")
            } else {
                Log.e(TAG, "❌ CLIP initialization failed")
            }
        } catch (e: Exception) {
            Log.e(TAG, "❌ CLIP initialization exception: ${e.message}", e)
            uiState = uiState.copy(
                clipInitialized = false,
                errorMessage = "CLIP init failed: ${e.message}"
            )
        }
    }
    
    private fun updateInitMessage() {
        val messages = mutableListOf<String>()
        
        if (uiState.llama32Initialized) {
            messages.add("✅ LLaMA 3.2 1B")
        }
        if (uiState.clipInitialized) {
            messages.add("✅ CLIP")
        }
        
        val message = if (messages.isEmpty()) {
            "⚠️ No models initialized. Check logs."
        } else {
            "Models ready:\n${messages.joinToString("\n")}"
        }
        
        uiState = uiState.copy(initializationMessage = message)
    }
    
    // Model selection
    fun selectModel(model: ModelType) {
        uiState = uiState.copy(currentModel = model)
        Log.i(TAG, "Selected model: $model")
    }
    
    // === LLaMA 3.2 Functions ===
    
    fun updateLlama32Prompt(prompt: String) {
        uiState = uiState.copy(llama32Prompt = prompt)
    }
    
    fun generateTextWithLlama32(
        maxTokens: Int = 128,
        temperature: Float = 0.8f
    ) {
        if (!uiState.llama32Initialized || llama32Inference == null) {
            uiState = uiState.copy(
                llama32Response = "❌ LLaMA 3.2 not initialized. Please check model files.",
                errorMessage = "LLaMA 3.2 not available"
            )
            return
        }
        
        val prompt = uiState.llama32Prompt.trim()
        if (prompt.isEmpty()) {
            uiState = uiState.copy(
                llama32Response = "⚠️ Please enter a prompt",
                errorMessage = "Empty prompt"
            )
            return
        }
        
        Log.i(TAG, "🎯 Generating text with LLaMA 3.2...")
        uiState = uiState.copy(
            llama32IsGenerating = true,
            llama32Response = "Generating...",
            errorMessage = null
        )
        
        viewModelScope.launch {
            try {
                val startTime = System.currentTimeMillis()
                
                val response = llama32Inference?.generateText(
                    prompt = prompt,
                    maxTokens = maxTokens,
                    temperature = temperature
                ) ?: ""
                
                val duration = System.currentTimeMillis() - startTime
                val tokensPerSec = if (duration > 0) (maxTokens * 1000.0f) / duration else 0f
                
                val formattedResponse = buildString {
                    appendLine(response)
                    appendLine()
                    appendLine("---")
                    appendLine("⏱️ Time: ${duration}ms (~%.1f tokens/sec)".format(tokensPerSec))
                    appendLine("🔧 Params: max_tokens=$maxTokens, temp=$temperature")
                }
                
                uiState = uiState.copy(
                    llama32Response = formattedResponse,
                    llama32IsGenerating = false
                )
                
                Log.i(TAG, "✅ Text generation complete (${duration}ms)")
                
            } catch (e: Exception) {
                Log.e(TAG, "❌ Text generation failed: ${e.message}", e)
                uiState = uiState.copy(
                    llama32Response = "❌ Generation failed: ${e.message}",
                    llama32IsGenerating = false,
                    errorMessage = e.message
                )
            }
        }
    }
    
    fun generateTextStreamingLlama32(
        maxTokens: Int = 128,
        temperature: Float = 0.8f
    ) {
        if (!uiState.llama32Initialized || llama32Inference == null) {
            return
        }
        
        val prompt = uiState.llama32Prompt.trim()
        if (prompt.isEmpty()) return
        
        uiState = uiState.copy(
            llama32IsGenerating = true,
            llama32Response = "",
            errorMessage = null
        )
        
        viewModelScope.launch {
            try {
                llama32Inference?.generateTextStreaming(
                    prompt = prompt,
                    maxTokens = maxTokens,
                    temperature = temperature,
                    onToken = { token ->
                        // Update UI with each token
                        uiState = uiState.copy(
                            llama32Response = uiState.llama32Response + token
                        )
                    }
                )
                
                uiState = uiState.copy(llama32IsGenerating = false)
                
            } catch (e: Exception) {
                uiState = uiState.copy(
                    llama32Response = "❌ Streaming failed: ${e.message}",
                    llama32IsGenerating = false
                )
            }
        }
    }
    
    // === CLIP Functions ===
    
    fun updateClipImage(bitmap: Bitmap) {
        uiState = uiState.copy(
            clipImage = bitmap,
            clipAnswer = ""
        )
    }
    
    fun updateClipQuestion(question: String) {
        uiState = uiState.copy(clipQuestion = question)
    }
    
    fun runClipAnalysis() {
        val image = uiState.clipImage
        if (image == null) {
            uiState = uiState.copy(
                clipAnswer = "⚠️ Please select an image first"
            )
            return
        }
        
        if (!uiState.clipInitialized || clipInference == null) {
            uiState = uiState.copy(
                clipAnswer = "❌ CLIP not initialized"
            )
            return
        }
        
        uiState = uiState.copy(
            clipIsProcessing = true,
            clipAnswer = "Analyzing..."
        )
        
        viewModelScope.launch {
            try {
                val startTime = System.currentTimeMillis()
                
                val similarity = clipInference?.computeImageTextSimilarity(
                    image,
                    uiState.clipQuestion
                ) ?: 0.0f
                
                val duration = System.currentTimeMillis() - startTime
                
                val result = buildString {
                    appendLine("Similarity Score: ${"%.4f".format(similarity)}")
                    appendLine()
                    appendLine("This indicates how well the text matches the image.")
                    appendLine()
                    appendLine("---")
                    appendLine("⏱️ Time: ${duration}ms")
                }
                
                uiState = uiState.copy(
                    clipAnswer = result,
                    clipIsProcessing = false
                )
                
            } catch (e: Exception) {
                uiState = uiState.copy(
                    clipAnswer = "❌ Analysis failed: ${e.message}",
                    clipIsProcessing = false
                )
            }
        }
    }
    
    // Cleanup
    override fun onCleared() {
        super.onCleared()
        Log.i(TAG, "🧹 Cleaning up models...")
        
        llama32Inference?.release()
        llama32Inference = null
        
        clipInference?.release()
        clipInference = null
    }
}

    