package com.example.edgeai

import android.Manifest
import android.app.Activity
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.Bundle
import android.os.Environment
import android.provider.MediaStore
import android.util.Log
import android.view.View
import android.widget.*
import androidx.core.content.FileProvider
import com.example.edgeai.ml.ExecutorTorchCLIP
import kotlinx.coroutines.*
import java.io.File
import java.io.IOException
import java.text.SimpleDateFormat
import java.util.*

/**
 * Specialized MainActivity for On-Device CLIP
 * Focused exclusively on Multimodal Search using ExecuTorch + QNN
 */
class MainActivityEnhanced : Activity() {
    
    // UI Components
    private lateinit var statusText: TextView
    private lateinit var clipImageView: ImageView
    private lateinit var clipQuestionInput: EditText
    private lateinit var clipResultText: TextView
    private lateinit var clipCaptureButton: Button
    private lateinit var clipGalleryButton: Button
    private lateinit var clipAnalyzeButton: Button
    
    // Model
    private var clip: ExecutorTorchCLIP? = null
    
    // State
    private var clipBitmap: Bitmap? = null
    private var currentPhotoPath: String = ""
    
    companion object {
        private const val TAG = "EdgeAI_CLIP"
        private const val REQUEST_IMAGE_CAPTURE = 1
        private const val REQUEST_GALLERY_IMAGE = 2
        private const val REQUEST_PERMISSIONS = 3
    }
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main_enhanced)
        
        Log.i(TAG, "🚀 Starting EdgeAI CLIP Demo")
        
        initializeViews()
        requestPermissions()
        initializeCLIPModel()
    }
    
    private fun initializeViews() {
        statusText = findViewById(R.id.statusText)
        clipImageView = findViewById(R.id.clipImageView)
        clipQuestionInput = findViewById(R.id.clipQuestionInput)
        clipResultText = findViewById(R.id.clipResultText)
        clipCaptureButton = findViewById(R.id.clipCaptureButton)
        clipGalleryButton = findViewById(R.id.clipGalleryButton)
        clipAnalyzeButton = findViewById(R.id.clipAnalyzeButton)
        
        setupCLIPUI()
    }
    
    private fun setupCLIPUI() {
        clipCaptureButton.setOnClickListener {
            captureImage()
        }
        
        clipGalleryButton.setOnClickListener {
            selectFromGallery()
        }
        
        clipAnalyzeButton.setOnClickListener {
            analyzeWithCLIP()
        }
        
        clipQuestionInput.setText("What is in this image?")
        clipAnalyzeButton.isEnabled = false
    }
    
    private fun requestPermissions() {
        val permissions = arrayOf(
            Manifest.permission.CAMERA,
            Manifest.permission.READ_EXTERNAL_STORAGE,
            Manifest.permission.WRITE_EXTERNAL_STORAGE
        )
        
        val missingPermissions = permissions.filter { 
            checkSelfPermission(it) != PackageManager.PERMISSION_GRANTED 
        }
        
        if (missingPermissions.isNotEmpty()) {
            requestPermissions(missingPermissions.toTypedArray(), REQUEST_PERMISSIONS)
        }
    }
    
    private fun initializeCLIPModel() {
        statusText.text = "Initializing CLIP..."
        
        CoroutineScope(Dispatchers.IO).launch {
            try {
                Log.i(TAG, "Initializing CLIP Engine...")
                clip = ExecutorTorchCLIP(this@MainActivityEnhanced)
                val success = clip?.initialize() ?: false
                
                withContext(Dispatchers.Main) {
                    if (success) {
                        statusText.text = "✅ CLIP Engine Ready (ExecuTorch + QNN)"
                        clipAnalyzeButton.isEnabled = true
                        Log.i(TAG, "✅ CLIP initialized successfully")
                    } else {
                        statusText.text = "❌ CLIP Initialization Failed"
                        Log.e(TAG, "❌ CLIP failed to initialize")
                    }
                }
            } catch (e: Exception) {
                Log.e(TAG, "CRITICAL: CLIP init exception", e)
                withContext(Dispatchers.Main) {
                    statusText.text = "❌ Error: ${e.message}"
                }
            }
        }
    }
    
    private fun captureImage() {
        val intent = Intent(MediaStore.ACTION_IMAGE_CAPTURE)
        if (intent.resolveActivity(packageManager) != null) {
            val photoFile = try {
                createImageFile()
            } catch (e: IOException) {
                null
            }
            
            photoFile?.let {
                val photoURI = FileProvider.getUriForFile(
                    this,
                    "com.example.edgeai.fileprovider",
                    it
                )
                intent.putExtra(MediaStore.EXTRA_OUTPUT, photoURI)
                startActivityForResult(intent, REQUEST_IMAGE_CAPTURE)
            }
        }
    }
    
    private fun selectFromGallery() {
        val intent = Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI)
        startActivityForResult(intent, REQUEST_GALLERY_IMAGE)
    }
    
    private fun analyzeWithCLIP() {
        if (clipBitmap == null) {
            Toast.makeText(this, "Please select an image first", Toast.LENGTH_SHORT).show()
            return
        }
        
        if (clip == null || !clip!!.isInitialized()) {
            Toast.makeText(this, "CLIP not initialized", Toast.LENGTH_SHORT).show()
            return
        }
        
        val question = clipQuestionInput.text.toString().trim()
        clipResultText.text = "Wait... Analyzing on-device..."
        clipAnalyzeButton.isEnabled = false
        
        CoroutineScope(Dispatchers.IO).launch {
            try {
                val startTime = System.currentTimeMillis()
                
                // CLIP Inference: Image/Text Embeddings Matching
                val similarity = clip?.computeImageTextSimilarity(clipBitmap!!, question) ?: 0.0f
                
                val duration = System.currentTimeMillis() - startTime
                
                withContext(Dispatchers.Main) {
                    val result = buildString {
                        appendLine("Similarity Score: ${"%.4f".format(similarity)}")
                        appendLine()
                        appendLine("Status: Match complete")
                        appendLine("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
                        appendLine("⏱️ Latency: ${duration}ms")
                        appendLine("🚀 Backend: Qualcomm QNN")
                    }
                    clipResultText.text = result
                    clipAnalyzeButton.isEnabled = true
                }
                
            } catch (e: Exception) {
                withContext(Dispatchers.Main) {
                    clipResultText.text = "❌ Analysis failed:\n${e.message}"
                    clipAnalyzeButton.isEnabled = true
                }
            }
        }
    }
    
    @Throws(IOException::class)
    private fun createImageFile(): File {
        val timeStamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.getDefault()).format(Date())
        val storageDir = getExternalFilesDir(Environment.DIRECTORY_PICTURES)
        return File.createTempFile("JPEG_${timeStamp}_", ".jpg", storageDir).apply {
            currentPhotoPath = absolutePath
        }
    }
    
    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        
        if (resultCode == Activity.RESULT_OK) {
            when (requestCode) {
                REQUEST_IMAGE_CAPTURE -> {
                    val bitmap = BitmapFactory.decodeFile(currentPhotoPath)
                    clipBitmap = bitmap
                    clipImageView.setImageBitmap(bitmap)
                    clipResultText.text = "Image captured. Click Analyze."
                }
                REQUEST_GALLERY_IMAGE -> {
                    data?.data?.let { uri ->
                        try {
                            val inputStream = contentResolver.openInputStream(uri)
                            clipBitmap = BitmapFactory.decodeStream(inputStream)
                            clipImageView.setImageBitmap(clipBitmap)
                            clipResultText.text = "Image loaded. Click Analyze."
                        } catch (e: Exception) {
                            Log.e(TAG, "Failed to load image", e)
                        }
                    }
                }
            }
        }
    }
    
    override fun onDestroy() {
        super.onDestroy()
        clip?.release()
        clipBitmap?.recycle()
    }
}
