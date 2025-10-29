package com.example.edgeai.ml

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import java.io.File
import java.io.FileOutputStream
import java.io.IOException

class CLIPInference(private val context: Context) {
    companion object {
        private const val TAG = "CLIPInference"

        init {
            try {
                System.loadLibrary("edgeai_qnn")
                Log.i(TAG, "Loaded native ExecuTorch/QNN library")
            } catch (e: UnsatisfiedLinkError) {
                Log.e(TAG, "Failed to load native library: ${e.message}", e)
            }
        }
    }

    // Native bindings implemented in qnn_infer.cpp
    private external fun nativeInitialize(modelPath: String): Boolean
    private external fun nativeRunInference(imageData: FloatArray, width: Int, height: Int): Map<String, FloatArray>?
    private external fun nativeRelease()

    private var isInitialized = false

    fun initialize(): Boolean {
        return try {
            val dlcPath = ensureClipDlcPresent()
            ensureContextBinaryPresent()
            val ok = nativeInitialize(dlcPath)
            isInitialized = ok
            Log.i(TAG, if (ok) "QNN CLIP initialized" else "QNN CLIP init failed")
            ok
        } catch (e: Exception) {
            Log.e(TAG, "CLIP initialize exception: ${e.message}", e)
            isInitialized = false
            false
        }
    }

    fun runInference(bitmap: Bitmap): Map<String, FloatArray>? {
        if (!isInitialized) {
            Log.w(TAG, "CLIP not initialized")
            return null
        }
        val (data, w, h) = bitmapToFloatNhwc(bitmap)
        return try {
            nativeRunInference(data, w, h)
        } catch (e: Exception) {
            Log.e(TAG, "CLIP inference exception: ${e.message}", e)
            null
        }
    }

    fun release() {
        try {
            nativeRelease()
        } catch (_: Throwable) {
        }
        isInitialized = false
    }

    val size: Int get() = if (isInitialized) 1 else 0

    private fun ensureClipDlcPresent(): String {
        // DLC expected at assets/models/clip/openai_clip.dlc
        val outDir = File(context.filesDir, "models/clip").apply { mkdirs() }
        val outFile = File(outDir, "openai_clip.dlc")
        if (outFile.exists() && outFile.length() > 0) return outFile.absolutePath

        val assetPath = "models/clip/openai_clip.dlc"
        context.assets.open(assetPath).use { input ->
            FileOutputStream(outFile).use { output ->
                input.copyTo(output)
            }
        }
        if (!outFile.exists() || outFile.length() == 0L) {
            throw IOException("Failed to materialize CLIP DLC from assets")
        }
        return outFile.absolutePath
    }

    private fun ensureContextBinaryPresent() {
        val destDir = File(context.filesDir, "models/clip/context_binaries").apply { mkdirs() }
        val dest = File(destDir, "context.bin")
        if (dest.exists() && dest.length() > 0) return
        // If asset exists, copy it
        val assetPath = "models/clip/context_binaries/context.bin"
        try {
            context.assets.open(assetPath).use { input ->
                FileOutputStream(dest).use { output ->
                    input.copyTo(output)
                }
            }
            Log.i(TAG, "Copied context.bin to ${dest.absolutePath}")
        } catch (_: Exception) {
            // Asset may not exist; continue gracefully
            Log.w(TAG, "No context.bin in assets; running without precompiled context")
        }
    }

    private fun bitmapToFloatNhwc(bitmap: Bitmap): Triple<FloatArray, Int, Int> {
        val width = bitmap.width
        val height = bitmap.height
        val pixels = IntArray(width * height)
        bitmap.getPixels(pixels, 0, width, 0, 0, width, height)

        val data = FloatArray(width * height * 3)
        var idx = 0
        for (y in 0 until height) {
            for (x in 0 until width) {
                val c = pixels[y * width + x]
                val r = ((c shr 16) and 0xFF) / 255.0f
                val g = ((c shr 8) and 0xFF) / 255.0f
                val b = (c and 0xFF) / 255.0f
                data[idx++] = r
                data[idx++] = g
                data[idx++] = b
            }
        }
        return Triple(data, width, height)
    }
}

