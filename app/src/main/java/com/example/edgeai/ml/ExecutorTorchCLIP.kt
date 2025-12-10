package com.example.edgeai.ml

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import java.io.File
import java.io.FileOutputStream
import java.io.IOException

class ExecutorTorchCLIP(private val context: Context) {

    companion object {
        private const val TAG = "ExecutorTorchCLIP"
        
        private const val INPUT_WIDTH = 224
        private const val INPUT_HEIGHT = 224
        private const val INPUT_CHANNELS = 3
        
        private val MEAN = floatArrayOf(0.485f, 0.456f, 0.406f)
        private val STD = floatArrayOf(0.229f, 0.224f, 0.225f)

        init {
            try {
                System.loadLibrary("edgeai_qnn")
                Log.i(TAG, "Native ExecuTorch + QNN library loaded successfully")
            } catch (e: UnsatisfiedLinkError) {
                Log.e(TAG, "Failed to load native library: ${e.message}", e)
                throw RuntimeException("Failed to load ExecuTorch + QNN native library", e)
            }
        }
    }

    private external fun nativeInitializeCLIP(modelPath: String): Boolean
    private external fun nativeRunImageInference(imageData: FloatArray): FloatArray?
    private external fun nativeRunTextInference(text: String): FloatArray?
    private external fun nativeComputeSimilarity(imageEmbedding: FloatArray, textEmbedding: FloatArray): Float
    private external fun nativeGetModelInfo(): String
    private external fun nativeIsInitialized(): Boolean
    private external fun nativeRelease()

    private var isInitialized = false

    fun initialize(): Boolean {
        Log.i(TAG, "Initializing ExecuTorch + QNN CLIP backend")

        try {
            val modelFilePath = getModelFilePath()
            
            if (modelFilePath == null) {
                Log.e(TAG, "Failed to get model file path")
                return false
            }

            Log.i(TAG, "Model path: $modelFilePath")
            Log.i(TAG, "Backend: ExecuTorch + QNN (Qualcomm AI Engine Direct)")
            Log.i(TAG, "Target: Hexagon HTP processor")

            val success = nativeInitializeCLIP(modelFilePath)
            
            if (success) {
                isInitialized = true
                Log.i(TAG, "ExecuTorch + QNN CLIP backend initialized")
            } else {
                Log.e(TAG, "Failed to initialize ExecuTorch + QNN backend")
                isInitialized = false
            }
            
            return success

        } catch (e: Exception) {
            Log.e(TAG, "Failed to initialize ExecuTorch + QNN CLIP", e)
            isInitialized = false
            return false
        }
    }

    fun runImageInference(bitmap: Bitmap): FloatArray? {
        if (!isInitialized) {
            Log.e(TAG, "ExecuTorch + QNN backend not initialized")
            return null
        }

        try {
            Log.i(TAG, "Running ExecuTorch + QNN CLIP image inference")
            Log.i(TAG, "Image: ${bitmap.width}x${bitmap.height}")

            val inputData = preprocessImage(bitmap)
            Log.i(TAG, "Preprocessed image data: ${inputData.size} values")

            val embedding = nativeRunImageInference(inputData)
            
            if (embedding != null) {
                Log.i(TAG, "ExecuTorch + QNN image inference completed")
                Log.i(TAG, "Image embedding size: ${embedding.size}")
                return embedding
            } else {
                Log.e(TAG, "ExecuTorch + QNN image inference failed")
                return null
            }

        } catch (e: Exception) {
            Log.e(TAG, "ExecuTorch + QNN image inference error: ${e.message}", e)
            return null
        }
    }

    fun runTextInference(text: String): FloatArray? {
        if (!isInitialized) {
            Log.e(TAG, "ExecuTorch + QNN backend not initialized")
            return null
        }

        try {
            Log.i(TAG, "Running ExecuTorch + QNN text inference")
            Log.i(TAG, "Text: $text")

            val embedding = nativeRunTextInference(text)
            
            if (embedding != null) {
                Log.i(TAG, "ExecuTorch + QNN text inference completed")
                Log.i(TAG, "Text embedding size: ${embedding.size}")
                return embedding
            } else {
                Log.e(TAG, "ExecuTorch + QNN text inference failed")
                return null
            }

        } catch (e: Exception) {
            Log.e(TAG, "ExecuTorch + QNN text inference error: ${e.message}", e)
            return null
        }
    }

    fun computeImageTextSimilarity(bitmap: Bitmap, text: String): Float {
        if (!isInitialized) {
            Log.e(TAG, "ExecuTorch + QNN backend not initialized")
            return 0.0f
        }

        try {
            Log.i(TAG, "Computing image-text similarity with ExecuTorch + QNN")
            
            val imageEmbedding = runImageInference(bitmap)
            if (imageEmbedding == null) {
                Log.e(TAG, "Failed to get image embedding")
                return 0.0f
            }
            
            val textEmbedding = runTextInference(text)
            if (textEmbedding == null) {
                Log.e(TAG, "Failed to get text embedding")
                return 0.0f
            }
            
            val similarity = nativeComputeSimilarity(imageEmbedding, textEmbedding)
            
            Log.i(TAG, "Similarity computed with ExecuTorch + QNN: $similarity")
            return similarity

        } catch (e: Exception) {
            Log.e(TAG, "Similarity computation error: ${e.message}", e)
            return 0.0f
        }
    }

    fun getModelInfo(): String {
        return if (isInitialized) {
            try {
                nativeGetModelInfo()
            } catch (e: Exception) {
                Log.e(TAG, "Failed to get model info: ${e.message}", e)
                "ExecuTorch + QNN model info unavailable"
            }
        } else {
            "ExecuTorch + QNN backend not initialized"
        }
    }

    fun isInitialized(): Boolean {
        return isInitialized && nativeIsInitialized()
    }

    fun release() {
        if (isInitialized) {
            try {
                Log.i(TAG, "Releasing ExecuTorch + QNN resources")
                nativeRelease()
                isInitialized = false
                Log.i(TAG, "ExecuTorch + QNN resources released")
            } catch (e: Exception) {
                Log.e(TAG, "Error releasing resources: ${e.message}", e)
            }
        }
    }

    private fun getModelFilePath(): String? {
        val internalDir = File(context.filesDir, "clip_models")
        if (!internalDir.exists()) {
            internalDir.mkdirs()
        }

        val modelFile = File(internalDir, "openai_clip.dlc")
        
        if (modelFile.exists()) {
            Log.i(TAG, "Model file found: ${modelFile.absolutePath}")
            return modelFile.absolutePath
        }

        if (copyModelFromAssets(modelFile)) {
            Log.i(TAG, "Model copied from assets: ${modelFile.absolutePath}")
            return modelFile.absolutePath
        }

        Log.e(TAG, "Failed to get model file")
        return null
    }

    private fun copyModelFromAssets(targetFile: File): Boolean {
        return try {
            val inputStream = context.assets.open("models_external/openai_clip.dlc")
            val outputStream = FileOutputStream(targetFile)
            
            inputStream.copyTo(outputStream)
            inputStream.close()
            outputStream.close()
            
            Log.i(TAG, "Model copied from assets to ${targetFile.absolutePath}")
            true
        } catch (e: IOException) {
            Log.e(TAG, "Failed to copy model from assets", e)
            false
        }
    }

    private fun preprocessImage(bitmap: Bitmap): FloatArray {
        Log.i(TAG, "Preprocessing image: ${bitmap.width}x${bitmap.height} -> ${INPUT_WIDTH}x${INPUT_HEIGHT}")

        val resizedBitmap = Bitmap.createScaledBitmap(bitmap, INPUT_WIDTH, INPUT_HEIGHT, true)

        val pixels = IntArray(INPUT_WIDTH * INPUT_HEIGHT)
        resizedBitmap.getPixels(pixels, 0, INPUT_WIDTH, 0, 0, INPUT_WIDTH, INPUT_HEIGHT)

        val inputData = FloatArray(INPUT_CHANNELS * INPUT_HEIGHT * INPUT_WIDTH)

        for (i in pixels.indices) {
            val pixel = pixels[i]

            val r = ((pixel shr 16) and 0xFF) / 255.0f
            val g = ((pixel shr 8) and 0xFF) / 255.0f
            val b = (pixel and 0xFF) / 255.0f

            inputData[i] = (r - MEAN[0]) / STD[0]
            inputData[INPUT_HEIGHT * INPUT_WIDTH + i] = (g - MEAN[1]) / STD[1]
            inputData[2 * INPUT_HEIGHT * INPUT_WIDTH + i] = (b - MEAN[2]) / STD[2]
        }

        if (resizedBitmap != bitmap) {
            resizedBitmap.recycle()
        }

        Log.i(TAG, "Image preprocessing completed")
        return inputData
    }
}
