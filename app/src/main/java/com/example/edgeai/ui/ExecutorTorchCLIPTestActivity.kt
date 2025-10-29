package com.example.edgeai.ui

import android.app.Activity
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.os.Bundle
import android.provider.MediaStore
import android.util.Log
import android.widget.Button
import android.widget.EditText
import android.widget.ImageView
import android.widget.TextView
import android.widget.Toast
import com.example.edgeai.ml.ExecutorTorchCLIPProper
import com.example.edgeai.R
import java.io.IOException

/**
 * Test Activity for ExecuTorch CLIP Integration
 * 
 * This activity demonstrates CLIP image-text similarity using ExecuTorch + QNN backend
 */
class ExecutorTorchCLIPTestActivity : Activity() {

    companion object {
        private const val TAG = "ExecutorTorchCLIPTest"
        private const val REQUEST_IMAGE_CAPTURE = 1
        private const val REQUEST_GALLERY_IMAGE = 2
    }

    private lateinit var clipInference: ExecutorTorchCLIPProper
    private lateinit var imageView: ImageView
    private lateinit var textInput: EditText
    private lateinit var similarityTextView: TextView
    private lateinit var modelInfoTextView: TextView
    private lateinit var initializeButton: Button
    private lateinit var captureButton: Button
    private lateinit var galleryButton: Button
    private lateinit var computeButton: Button
    private lateinit var testButton: Button

    private var currentBitmap: Bitmap? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_executor_torch_clip_test)

        Log.i(TAG, "🚀 Starting ExecuTorch CLIP Test Activity")

        // Initialize UI components
        initializeViews()
        
        // Initialize ExecuTorch + QNN CLIP inference engine
        clipInference = ExecutorTorchCLIPProper(this)
        
        // Set up button listeners
        setupButtonListeners()
        
        // Display initial model info
        updateModelInfo()
        
        Log.i(TAG, "✅ ExecuTorch CLIP Test Activity initialized")
    }

    private fun initializeViews() {
        imageView = findViewById(R.id.imageView)
        textInput = findViewById(R.id.textInput)
        similarityTextView = findViewById(R.id.similarityTextView)
        modelInfoTextView = findViewById(R.id.modelInfoTextView)
        initializeButton = findViewById(R.id.initializeButton)
        captureButton = findViewById(R.id.captureButton)
        galleryButton = findViewById(R.id.galleryButton)
        computeButton = findViewById(R.id.computeButton)
        testButton = findViewById(R.id.testButton)
        
        // Set default text
        textInput.setText("a photo of a cat")
    }

    private fun setupButtonListeners() {
        initializeButton.setOnClickListener {
            initializeCLIPModel()
        }

        captureButton.setOnClickListener {
            captureImage()
        }

        galleryButton.setOnClickListener {
            selectFromGallery()
        }

        computeButton.setOnClickListener {
            computeSimilarity()
        }

        testButton.setOnClickListener {
            runCLIPTests()
        }
    }

    private fun initializeCLIPModel() {
        Log.i(TAG, "🔧 Initializing CLIP model...")
        
        initializeButton.isEnabled = false
        initializeButton.text = "Initializing..."
        
        Thread {
            try {
                val success = clipInference.initializeCLIP()
                
                runOnUiThread {
                    if (success) {
                        Log.i(TAG, "✅ CLIP model initialized successfully")
                        Toast.makeText(this, "CLIP model initialized!", Toast.LENGTH_SHORT).show()
                        updateModelInfo()
                    } else {
                        Log.e(TAG, "❌ Failed to initialize CLIP model")
                        Toast.makeText(this, "Failed to initialize CLIP model", Toast.LENGTH_LONG).show()
                    }
                    
                    initializeButton.isEnabled = true
                    initializeButton.text = "Initialize Model"
                }
            } catch (e: Exception) {
                Log.e(TAG, "❌ Exception during CLIP initialization", e)
                runOnUiThread {
                    Toast.makeText(this, "Error: ${e.message}", Toast.LENGTH_LONG).show()
                    initializeButton.isEnabled = true
                    initializeButton.text = "Initialize Model"
                }
            }
        }.start()
    }

    private fun captureImage() {
        val intent = Intent(MediaStore.ACTION_IMAGE_CAPTURE)
        if (intent.resolveActivity(packageManager) != null) {
            startActivityForResult(intent, REQUEST_IMAGE_CAPTURE)
        } else {
            Toast.makeText(this, "Camera not available", Toast.LENGTH_SHORT).show()
        }
    }

    private fun selectFromGallery() {
        val intent = Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI)
        startActivityForResult(intent, REQUEST_GALLERY_IMAGE)
    }

    private fun computeSimilarity() {
        if (!clipInference.isInitialized()) {
            Toast.makeText(this, "Please initialize CLIP model first", Toast.LENGTH_SHORT).show()
            return
        }

        if (currentBitmap == null) {
            Toast.makeText(this, "Please select an image first", Toast.LENGTH_SHORT).show()
            return
        }

        val text = textInput.text.toString().trim()
        if (text.isEmpty()) {
            Toast.makeText(this, "Please enter some text", Toast.LENGTH_SHORT).show()
            return
        }

        Log.i(TAG, "🔍 Computing image-text similarity...")
        computeButton.isEnabled = false
        computeButton.text = "Computing..."

        Thread {
            try {
                val similarity = clipInference.computeImageTextSimilarity(currentBitmap!!, text)
                
                runOnUiThread {
                    Log.i(TAG, "✅ Similarity computed: $similarity")
                    similarityTextView.text = "Similarity: ${String.format("%.4f", similarity)}"
                    
                    // Color code the similarity
                    val color = when {
                        similarity > 0.7f -> android.graphics.Color.GREEN
                        similarity > 0.4f -> android.graphics.Color.YELLOW
                        else -> android.graphics.Color.RED
                    }
                    similarityTextView.setTextColor(color)
                    
                    computeButton.isEnabled = true
                    computeButton.text = "Compute Similarity"
                }
            } catch (e: Exception) {
                Log.e(TAG, "❌ Exception during similarity computation", e)
                runOnUiThread {
                    Toast.makeText(this, "Error: ${e.message}", Toast.LENGTH_LONG).show()
                    computeButton.isEnabled = true
                    computeButton.text = "Compute Similarity"
                }
            }
        }.start()
    }

    private fun runCLIPTests() {
        if (!clipInference.isInitialized()) {
            Toast.makeText(this, "Please initialize CLIP model first", Toast.LENGTH_SHORT).show()
            return
        }

        Log.i(TAG, "🧪 Running CLIP tests...")
        testButton.isEnabled = false
        testButton.text = "Running Tests..."

        Thread {
            try {
                val testResults = mutableListOf<String>()
                
                // Test 1: Image inference
                if (currentBitmap != null) {
                    val imageEmbedding = clipInference.runImageInference(currentBitmap!!)
                    if (imageEmbedding != null) {
                        testResults.add("✅ Image inference: ${imageEmbedding.size}D embedding")
                    } else {
                        testResults.add("❌ Image inference failed")
                    }
                } else {
                    testResults.add("⚠️ No image for image inference test")
                }
                
                // Test 2: Text inference
                val textEmbedding = clipInference.runTextInference("a photo of a cat")
                if (textEmbedding != null) {
                    testResults.add("✅ Text inference: ${textEmbedding.size}D embedding")
                } else {
                    testResults.add("❌ Text inference failed")
                }
                
                // Test 3: Model info
                val modelInfo = clipInference.getModelInfo()
                testResults.add("📊 Model info: ${modelInfo.length} characters")
                
                runOnUiThread {
                    val resultsText = testResults.joinToString("\n")
                    similarityTextView.text = "Test Results:\n$resultsText"
                    similarityTextView.setTextColor(android.graphics.Color.BLACK)
                    
                    testButton.isEnabled = true
                    testButton.text = "Run Tests"
                    
                    Toast.makeText(this, "Tests completed", Toast.LENGTH_SHORT).show()
                }
                
            } catch (e: Exception) {
                Log.e(TAG, "❌ Exception during CLIP tests", e)
                runOnUiThread {
                    Toast.makeText(this, "Error: ${e.message}", Toast.LENGTH_LONG).show()
                    testButton.isEnabled = true
                    testButton.text = "Run Tests"
                }
            }
        }.start()
    }

    private fun updateModelInfo() {
        val info = clipInference.getModelInfo()
        modelInfoTextView.text = info
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        
        if (resultCode == RESULT_OK) {
            when (requestCode) {
                REQUEST_IMAGE_CAPTURE -> {
                    val imageBitmap = data?.extras?.get("data") as? Bitmap
                    if (imageBitmap != null) {
                        currentBitmap = imageBitmap
                        imageView.setImageBitmap(imageBitmap)
                        Log.i(TAG, "📸 Image captured: ${imageBitmap.width}x${imageBitmap.height}")
                    }
                }
                REQUEST_GALLERY_IMAGE -> {
                    val imageUri: Uri? = data?.data
                    if (imageUri != null) {
                        try {
                            val bitmap = BitmapFactory.decodeStream(contentResolver.openInputStream(imageUri))
                            if (bitmap != null) {
                                currentBitmap = bitmap
                                imageView.setImageBitmap(bitmap)
                                Log.i(TAG, "🖼️ Image selected: ${bitmap.width}x${bitmap.height}")
                            }
                        } catch (e: IOException) {
                            Log.e(TAG, "❌ Error loading image from gallery", e)
                            Toast.makeText(this, "Error loading image", Toast.LENGTH_SHORT).show()
                        }
                    }
                }
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        try {
            clipInference.release()
            Log.i(TAG, "✅ ExecuTorch CLIP resources released")
        } catch (e: Exception) {
            Log.e(TAG, "❌ Error releasing CLIP resources", e)
        }
    }
}
