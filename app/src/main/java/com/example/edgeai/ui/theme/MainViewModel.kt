// In app/src/main/java/com/example/edgeai/ui/MainViewModel.kt
package com.example.edgeai.ui.theme

import android.content.Context
import android.graphics.Bitmap
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.example.edgeai.ml.ExecutorTorchCLIP
import kotlinx.coroutines.launch

// Represents the current state of our UI
data class VqaUiState(
    val selectedImage: Bitmap? = null,
    val question: String = "What color is the car?",
    val answer: String = "",
    val isLoading: Boolean = false
)

class MainViewModel : ViewModel() {
    var uiState by mutableStateOf(VqaUiState())
        private set

    private lateinit var clipInference: ExecutorTorchCLIP

    fun initialize(context: Context) {
        if (!::clipInference.isInitialized) {
            clipInference = ExecutorTorchCLIP(context)
            clipInference.initialize()
        }
    }

    fun onImageSelected(bitmap: Bitmap) {
        uiState = uiState.copy(selectedImage = bitmap, answer = "")
    }

    fun onQuestionChanged(newQuestion: String) {
        uiState = uiState.copy(question = newQuestion)
    }

    fun askQuestion() {
        val image = uiState.selectedImage ?: return
        val question = uiState.question

        viewModelScope.launch {
            uiState = uiState.copy(isLoading = true)
            
            val similarity = clipInference.computeImageTextSimilarity(image, question)
            
            val result = "Similarity Score: ${"%.4f".format(similarity)}\n\n" +
                        "This indicates how well the text matches the image."
            
            uiState = uiState.copy(answer = result, isLoading = false)
        }
    }
}
    