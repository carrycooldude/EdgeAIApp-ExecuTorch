package com.example.edgeai

import android.Manifest
import android.app.Activity
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.os.Bundle
import android.os.Environment
import android.provider.MediaStore
import android.util.Log
import android.widget.Button
import android.widget.EditText
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.RadioButton
import android.widget.RadioGroup
import android.widget.SeekBar
import android.widget.TextView
import android.widget.Toast
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.core.content.FileProvider
import com.example.edgeai.ml.LLaMAInference
import com.example.edgeai.ml.CLIPInference
import kotlinx.coroutines.*
import java.io.File
import java.text.SimpleDateFormat
import java.util.*

/**
 * EdgeAI CLIP Inference Demo
 * Main activity for running CLIP inference on Qualcomm EdgeAI
 */
class MainActivity : Activity() {

    // UI Components
    private lateinit var imageView: ImageView
    private lateinit var resultTextView: TextView
    private lateinit var captureButton: Button
    private lateinit var galleryButton: Button
    private lateinit var inferenceButton: Button
    
    // New UI Components for LLaMA
    private lateinit var modelSelectionGroup: RadioGroup
    private lateinit var clipModelRadio: RadioButton
    private lateinit var llamaModelRadio: RadioButton
    private lateinit var clipImageSection: LinearLayout
    private lateinit var llamaTextSection: LinearLayout
    private lateinit var clipActionButtons: LinearLayout
    private lateinit var llamaActionButtons: LinearLayout
    private lateinit var textInput: EditText
    private lateinit var maxTokensSeekBar: SeekBar
    private lateinit var maxTokensText: TextView
    private lateinit var clearTextButton: Button
    private lateinit var exampleButton: Button
    private lateinit var gemma3TestButton: Button
    private lateinit var clipTestButton: Button

    // ML Components
    private var clipInference: CLIPInference? = null
    private var llamaInference: LLaMAInference? = null
    private var currentBitmap: Bitmap? = null
    private var currentPhotoPath: String = ""
    private var currentModel: String = "CLIP" // "CLIP" or "LLaMA"

    companion object {
        private const val TAG = "EdgeAI_CLIP"
        private const val REQUEST_IMAGE_CAPTURE = 1
        private const val REQUEST_GALLERY_IMAGE = 2
        private const val REQUEST_PERMISSIONS = 3
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        Log.i(TAG, "Starting EdgeAI CLIP Demo")

        initializeViews()
        requestPermissions()
        initializeModels()
    }

    /**
     * Initialize UI components and set up click listeners
     */
    private fun initializeViews() {
        try {
            // Basic UI components
            imageView = findViewById(R.id.imageView)
            resultTextView = findViewById(R.id.resultTextView)
            captureButton = findViewById(R.id.captureButton)
            galleryButton = findViewById(R.id.galleryButton)
            inferenceButton = findViewById(R.id.inferenceButton)
            
            // New UI components for model selection
            modelSelectionGroup = findViewById(R.id.modelSelectionGroup)
            clipModelRadio = findViewById(R.id.clipModelRadio)
            llamaModelRadio = findViewById(R.id.llamaModelRadio)
            clipImageSection = findViewById(R.id.clipImageSection)
            llamaTextSection = findViewById(R.id.llamaTextSection)
            clipActionButtons = findViewById(R.id.clipActionButtons)
            llamaActionButtons = findViewById(R.id.llamaActionButtons)
            textInput = findViewById(R.id.textInput)
            maxTokensSeekBar = findViewById(R.id.maxTokensSeekBar)
            maxTokensText = findViewById(R.id.maxTokensText)
            clearTextButton = findViewById(R.id.clearTextButton)
        exampleButton = findViewById(R.id.exampleButton)
        gemma3TestButton = findViewById(R.id.gemma3TestButton)
        clipTestButton = findViewById(R.id.clipTestButton)

            // Set up model selection listener
            modelSelectionGroup.setOnCheckedChangeListener { _, checkedId ->
                when (checkedId) {
                    R.id.clipModelRadio -> switchToCLIPModel()
                    R.id.llamaModelRadio -> switchToLLaMAModel()
                }
            }

            // Set up button click listeners
            captureButton.setOnClickListener {
                Log.i(TAG, "Camera button clicked")
                captureImage()
            }
            galleryButton.setOnClickListener {
                Log.i(TAG, "Gallery button clicked")
                selectFromGallery()
            }
            clearTextButton.setOnClickListener {
                Log.i(TAG, "🗑️ Clear text button clicked")
                textInput.setText("")
            }
            exampleButton.setOnClickListener {
                Log.i(TAG, "💡 Example button clicked")
                loadExamplePrompt()
            }
            
        gemma3TestButton.setOnClickListener {
            Log.i(TAG, "🧪 Gemma3 test button clicked")
            val intent = Intent(this, com.example.edgeai.ui.ExecutorTorchGemma3TestActivity::class.java)
            startActivity(intent)
        }

        clipTestButton.setOnClickListener {
            Log.i(TAG, "🖼️ CLIP test button clicked")
            val intent = Intent(this, com.example.edgeai.ui.ExecutorTorchCLIPTestActivity::class.java)
            startActivity(intent)
        }
            
            inferenceButton.setOnClickListener {
                Log.i(TAG, "🚀 Inference button clicked")
                runInference()
            }

            // Set up max tokens seekbar
            maxTokensSeekBar.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
                override fun onProgressChanged(seekBar: SeekBar?, progress: Int, fromUser: Boolean) {
                    maxTokensText.text = "$progress tokens"
                }
                override fun onStartTrackingTouch(seekBar: SeekBar?) {}
                override fun onStopTrackingTouch(seekBar: SeekBar?) {}
            })

            // Initially disable inference until model is loaded
            inferenceButton.isEnabled = false
            resultTextView.text = "🔄 Initializing models..."

            // Start with CLIP model
            switchToCLIPModel()

            Log.i(TAG, "✅ UI components initialized successfully")

        } catch (e: Exception) {
            Log.e(TAG, "❌ Failed to initialize UI: ${e.message}", e)
            Toast.makeText(this, "UI initialization failed", Toast.LENGTH_LONG).show()
        }
    }

    /**
     * Switch to CLIP model UI
     */
    private fun switchToCLIPModel() {
        currentModel = "CLIP"
        clipImageSection.visibility = LinearLayout.VISIBLE
        llamaTextSection.visibility = LinearLayout.GONE
        clipActionButtons.visibility = LinearLayout.VISIBLE
        llamaActionButtons.visibility = LinearLayout.GONE
        inferenceButton.text = "🚀 Run CLIP Inference"
        Log.i(TAG, "🖼️ Switched to CLIP model")
    }

    /**
     * Switch to LLaMA model UI
     */
    private fun switchToLLaMAModel() {
        currentModel = "LLaMA"
        clipImageSection.visibility = LinearLayout.GONE
        llamaTextSection.visibility = LinearLayout.VISIBLE
        clipActionButtons.visibility = LinearLayout.GONE
        llamaActionButtons.visibility = LinearLayout.VISIBLE
        inferenceButton.text = "🚀 Run LLaMA Inference"
        Log.i(TAG, "📝 Switched to LLaMA model")
    }

    /**
     * Load example prompt for LLaMA
     */
    private fun loadExamplePrompt() {
        val examples = listOf(
            "Explain the concept of artificial intelligence in simple terms.",
            "Write a short story about a robot learning to paint.",
            "What are the benefits of renewable energy?",
            "Describe the process of photosynthesis.",
            "How does machine learning work?"
        )
        val randomExample = examples.random()
        textInput.setText(randomExample)
        Log.i(TAG, "Loaded example prompt: ${randomExample.take(50)}...")
    }

    /**
     * Initialize both CLIP and LLaMA models
     */
    private fun initializeModels() {
        CoroutineScope(Dispatchers.IO).launch {
            try {
        Log.i(TAG, "Initializing simplified models...")

                // Initialize CLIP (QNN DLC)
                var clipSuccess = false
                try {
                    Log.i(TAG, "🚀 Creating CLIPInference instance...")
                    clipInference = CLIPInference(this@MainActivity)
                    clipSuccess = clipInference?.initialize() == true
                    Log.i(TAG, "📊 CLIP init result: $clipSuccess")
                } catch (e: Exception) {
                    Log.e(TAG, "❌ CLIP initialization failed: ${e.message}", e)
                    clipSuccess = false
                }

                // Initialize LLaMA (simplified)
                Log.i(TAG, "Creating LLaMAInference instance...")
                llamaInference = LLaMAInference(this@MainActivity)
                Log.i(TAG, "Calling LLaMA initialize()...")
                val llamaSuccess = try {
                    llamaInference?.initialize() ?: false
                } catch (e: Exception) {
                    Log.e(TAG, "LLaMA initialization failed: ${e.message}", e)
                    false
                }
                Log.i(TAG, "LLaMA initialization result: $llamaSuccess")

                withContext(Dispatchers.Main) {
                    val status = buildString {
                        append("Models Status:\n")
                        append("CLIP: ${if (clipSuccess) "Ready" else "Failed"}\n")
                        append("LLaMA: ${if (llamaSuccess) "Ready" else "Failed"}\n\n")
                        if (clipSuccess && llamaSuccess) {
                            append("Both models are ready! Select a model and start inference.")
                        } else {
                            append("Some models failed to load. Check logs for details.")
                        }
                    }
                    
                    resultTextView.text = status
                    inferenceButton.isEnabled = clipSuccess || llamaSuccess
                    
                    if (clipSuccess && llamaSuccess) {
                        Toast.makeText(this@MainActivity, "All models loaded successfully!", Toast.LENGTH_SHORT).show()
                    } else {
                        Toast.makeText(this@MainActivity, "Some models failed to load", Toast.LENGTH_LONG).show()
                    }
                }

            } catch (e: Exception) {
                Log.e(TAG, "❌ Model initialization error: ${e.message}", e)
                withContext(Dispatchers.Main) {
                    resultTextView.text = "❌ Model initialization error: ${e.message}"
                    Toast.makeText(this@MainActivity, "Model initialization failed", Toast.LENGTH_LONG).show()
                }
            }
        }
    }


    /**
     * Request necessary permissions for camera and storage
     */
    private fun requestPermissions() {
        val permissions = arrayOf(
            Manifest.permission.CAMERA,
            Manifest.permission.WRITE_EXTERNAL_STORAGE,
            Manifest.permission.READ_EXTERNAL_STORAGE
        )

        val permissionsNeeded = permissions.filter {
            ContextCompat.checkSelfPermission(this, it) != PackageManager.PERMISSION_GRANTED
        }

        if (permissionsNeeded.isNotEmpty()) {
            Log.i(TAG, "Requesting permissions: ${permissionsNeeded.joinToString()}")
            ActivityCompat.requestPermissions(this, permissionsNeeded.toTypedArray(), REQUEST_PERMISSIONS)
        } else {
            Log.i(TAG, "All permissions already granted")
        }
    }

    /**
     * Launch camera to capture image
     */
    private fun captureImage() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            Toast.makeText(this, "Camera permission required", Toast.LENGTH_SHORT).show()
            return
        }

        try {
            val intent = Intent(MediaStore.ACTION_IMAGE_CAPTURE)
            val photoFile = createImageFile()
            val photoURI = FileProvider.getUriForFile(this, "${packageName}.fileprovider", photoFile)

            intent.putExtra(MediaStore.EXTRA_OUTPUT, photoURI)
            startActivityForResult(intent, REQUEST_IMAGE_CAPTURE)

            Log.i(TAG, "📷 Camera intent launched, saving to: $currentPhotoPath")

        } catch (e: Exception) {
            Log.e(TAG, "❌ Error launching camera: ${e.message}", e)
            Toast.makeText(this, "Error launching camera: ${e.message}", Toast.LENGTH_SHORT).show()
        }
    }

    /**
     * Open gallery to select image
     */
    private fun selectFromGallery() {
        try {
            val intent = Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI)
            startActivityForResult(intent, REQUEST_GALLERY_IMAGE)
            Log.i(TAG, "🖼️ Gallery selection launched")
        } catch (e: Exception) {
            Log.e(TAG, "❌ Error opening gallery: ${e.message}", e)
            Toast.makeText(this, "Error opening gallery", Toast.LENGTH_SHORT).show()
        }
    }

    /**
     * Create temporary file for camera capture
     */
    private fun createImageFile(): File {
        val timeStamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.getDefault()).format(Date())
        val storageDir = getExternalFilesDir(Environment.DIRECTORY_PICTURES)

        return File.createTempFile("CLIP_${timeStamp}_", ".jpg", storageDir).apply {
            currentPhotoPath = absolutePath
        }
    }

    /**
     * Handle results from camera or gallery
     */
    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)

        if (resultCode == Activity.RESULT_OK) {
            when (requestCode) {
                REQUEST_IMAGE_CAPTURE -> {
                    Log.i(TAG, "📷 Image captured, loading from: $currentPhotoPath")
                    currentBitmap = BitmapFactory.decodeFile(currentPhotoPath)
                    displayImage(currentBitmap, "Camera")
                }

                REQUEST_GALLERY_IMAGE -> {
                    data?.data?.let { uri ->
                        Log.i(TAG, "🖼️ Image selected from gallery: $uri")
                        currentBitmap = loadImageFromUri(uri)
                        displayImage(currentBitmap, "Gallery")
                    } ?: Log.w(TAG, "⚠️ No image data received from gallery")
                }
            }
        } else {
            Log.w(TAG, "⚠️ Activity result not OK: requestCode=$requestCode, resultCode=$resultCode")
        }
    }

    /**
     * Load image from URI
     */
    private fun loadImageFromUri(uri: Uri): Bitmap? {
        return try {
            val inputStream = contentResolver.openInputStream(uri)
            val bitmap = BitmapFactory.decodeStream(inputStream)
            inputStream?.close()
            Log.i(TAG, "✅ Image loaded successfully: ${bitmap?.width}x${bitmap?.height}")
            bitmap
        } catch (e: Exception) {
            Log.e(TAG, "❌ Error loading image from URI: ${e.message}", e)
            Toast.makeText(this, "Error loading selected image", Toast.LENGTH_SHORT).show()
            null
        }
    }

    /**
     * Display loaded image and enable inference
     */
    private fun displayImage(bitmap: Bitmap?, source: String) {
        bitmap?.let {
            imageView.setImageBitmap(it)
            inferenceButton.isEnabled = (clipInference != null)
            resultTextView.text = "✅ Image loaded from $source (${it.width}x${it.height}). Ready for CLIP inference!"
            Log.i(TAG, "✅ Image displayed, inference button enabled: ${inferenceButton.isEnabled}")
        } ?: run {
            Log.e(TAG, "❌ Failed to display image - bitmap is null")
            Toast.makeText(this, "Failed to load image", Toast.LENGTH_SHORT).show()
        }
    }

    /**
     * Run inference based on selected model
     */
    private fun runInference() {
        when (currentModel) {
            "CLIP" -> runCLIPInference()
            "LLaMA" -> runLLaMAInference()
            else -> {
                Toast.makeText(this, "No model selected", Toast.LENGTH_SHORT).show()
            }
        }
    }

    /**
     * Run CLIP inference on current image
     */
    private fun runCLIPInference() {
        val bitmap = currentBitmap
        if (bitmap == null) {
            Toast.makeText(this, "Please select an image first", Toast.LENGTH_SHORT).show()
            return
        }

        if (clipInference == null) {
            Toast.makeText(this, "CLIP model not initialized", Toast.LENGTH_SHORT).show()
            return
        }

        // Disable button during inference
        inferenceButton.isEnabled = false
        resultTextView.text = "Running CLIP inference on ${bitmap.width}x${bitmap.height} image..."

        Log.i(TAG, "Starting CLIP inference...")

        CoroutineScope(Dispatchers.IO).launch {
            try {
                val startTime = System.currentTimeMillis()

                // Run inference
                val results = clipInference?.runInference(bitmap)

                val inferenceTime = System.currentTimeMillis() - startTime
                Log.i(TAG, "✅ CLIP inference completed in ${inferenceTime}ms")

                // Format and display results
                val formattedResults = formatCLIPResults(results, inferenceTime)
                saveResults(formattedResults)

                withContext(Dispatchers.Main) {
                    resultTextView.text = formattedResults
                    inferenceButton.isEnabled = true
                    Toast.makeText(this@MainActivity, "✅ CLIP inference completed in ${inferenceTime}ms!", Toast.LENGTH_SHORT).show()
                }

            } catch (e: Exception) {
                Log.e(TAG, "❌ CLIP inference failed: ${e.message}", e)
                withContext(Dispatchers.Main) {
                    resultTextView.text = "❌ CLIP inference failed: ${e.message}\n\nCheck logs for more details."
                    inferenceButton.isEnabled = true
                    Toast.makeText(this@MainActivity, "CLIP inference failed", Toast.LENGTH_SHORT).show()
                }
            }
        }
    }

    /**
     * Run LLaMA inference on input text
     */
    private fun runLLaMAInference() {
        val inputText = textInput.text.toString().trim()
        if (inputText.isEmpty()) {
            Toast.makeText(this, "Please enter some text first", Toast.LENGTH_SHORT).show()
            return
        }

        if (llamaInference == null) {
            Toast.makeText(this, "LLaMA model not initialized", Toast.LENGTH_SHORT).show()
            return
        }

        val maxTokens = maxTokensSeekBar.progress

        // Disable button during inference
        inferenceButton.isEnabled = false
        resultTextView.text = "Running LLaMA inference on text: ${inputText.take(50)}..."

        Log.i(TAG, "🚀 Starting LLaMA inference...")

        CoroutineScope(Dispatchers.IO).launch {
            try {
                val startTime = System.currentTimeMillis()

                // Run inference
                val result = llamaInference?.runInference(inputText)

                val inferenceTime = System.currentTimeMillis() - startTime
                Log.i(TAG, "✅ LLaMA inference completed in ${inferenceTime}ms")

                // Format and display results
                val formattedResults = formatLLaMAResults(result, inputText, maxTokens, inferenceTime)
                saveResults(formattedResults)

                withContext(Dispatchers.Main) {
                    resultTextView.text = formattedResults
                    inferenceButton.isEnabled = true
                    Toast.makeText(this@MainActivity, "✅ LLaMA inference completed in ${inferenceTime}ms!", Toast.LENGTH_SHORT).show()
                }

            } catch (e: Exception) {
                Log.e(TAG, "❌ LLaMA inference failed: ${e.message}", e)
                withContext(Dispatchers.Main) {
                    resultTextView.text = "❌ LLaMA inference failed: ${e.message}\n\nCheck logs for more details."
                    inferenceButton.isEnabled = true
                    Toast.makeText(this@MainActivity, "LLaMA inference failed", Toast.LENGTH_SHORT).show()
                }
            }
        }
    }

    /**
     * Format CLIP inference results for display
     */
    private fun formatCLIPResults(results: Map<String, FloatArray>?, inferenceTime: Long): String {
        if (results.isNullOrEmpty()) {
            return "❌ No inference results received\n\nPossible issues:\n- Model not loaded properly\n- QNN runtime error\n- Input preprocessing error"
        }

        val builder = StringBuilder()
        builder.append("🎯 CLIP Inference Results\n")
        builder.append("=" .repeat(40) + "\n")
        builder.append("⏱️ Inference Time: ${inferenceTime}ms\n")
        builder.append("📅 Timestamp: ${SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.getDefault()).format(Date())}\n")
        builder.append("🖼️ Input Image: ${currentBitmap?.width}x${currentBitmap?.height}\n\n")

        results.forEach { (outputName, data) ->
            builder.append("🔍 Output Tensor: $outputName\n")
            builder.append("📏 Size: ${data.size} elements\n")

            if (data.isNotEmpty()) {
                val max = data.maxOrNull() ?: 0f
                val min = data.minOrNull() ?: 0f
                val mean = data.average().toFloat()
                val std = kotlin.math.sqrt(data.map { (it - mean) * (it - mean) }.average()).toFloat()

                builder.append("📊 Statistics:\n")
                builder.append("   Max:  ${"%.6f".format(max)}\n")
                builder.append("   Min:  ${"%.6f".format(min)}\n")
                builder.append("   Mean: ${"%.6f".format(mean)}\n")
                builder.append("   Std:  ${"%.6f".format(std)}\n")

                // Show sample values
                builder.append("🔢 Sample Values:\n")
                val sampleCount = minOf(10, data.size)
                for (i in 0 until sampleCount) {
                    builder.append("   [$i]: ${"%.6f".format(data[i])}\n")
                }

                if (data.size > 10) {
                    builder.append("   ... (${data.size - 10} more values)\n")
                }
            } else {
                builder.append("⚠️ Empty output tensor\n")
            }

            builder.append("\n")
        }

        builder.append("💾 Results saved to external storage\n")
        builder.append("📱 Device: ${android.os.Build.MODEL} (${android.os.Build.DEVICE})\n")

        return builder.toString()
    }

    /**
     * Format LLaMA inference results for display
     */
    private fun formatLLaMAResults(result: String?, inputText: String, maxTokens: Int, inferenceTime: Long): String {
        if (result.isNullOrEmpty()) {
            return "❌ No LLaMA inference result received\n\nPossible issues:\n- Model not loaded properly\n- QNN runtime error\n- Input processing error"
        }

        val builder = StringBuilder()
        builder.append("🎯 LLaMA Inference Results\n")
        builder.append("=" .repeat(40) + "\n")
        builder.append("⏱️ Inference Time: ${inferenceTime}ms\n")
        builder.append("📅 Timestamp: ${SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.getDefault()).format(Date())}\n")
        builder.append("📝 Input Text: ${inputText.take(100)}${if (inputText.length > 100) "..." else ""}\n")
        builder.append("🔢 Max Tokens: $maxTokens\n")
        builder.append("📏 Generated Length: ${result.length} characters\n\n")

        builder.append("🤖 Generated Response:\n")
        builder.append("-".repeat(40) + "\n")
        builder.append(result)
        builder.append("\n" + "-".repeat(40) + "\n\n")

        builder.append("💾 Results saved to external storage\n")
        builder.append("📱 Device: ${android.os.Build.MODEL} (${android.os.Build.DEVICE})\n")

        return builder.toString()
    }

    /**
     * Save results to external storage
     */
    private fun saveResults(results: String) {
        try {
            val timestamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.getDefault()).format(Date())
            val filename = "edgeai_clip_results_$timestamp.txt"
            val documentsDir = getExternalFilesDir(Environment.DIRECTORY_DOCUMENTS)
            val file = File(documentsDir, filename)

            file.writeText(results)

            Log.i(TAG, "💾 Results saved to: ${file.absolutePath}")

            // Also save a copy in app's internal files for easy access
            val internalFile = File(filesDir, "latest_clip_results.txt")
            internalFile.writeText(results)

        } catch (e: Exception) {
            Log.e(TAG, "❌ Failed to save results: ${e.message}", e)
        }
    }

    /**
     * Handle permission request results
     */
    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<out String>, grantResults: IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)

        if (requestCode == REQUEST_PERMISSIONS) {
            val deniedPermissions = permissions.filterIndexed { index, _ ->
                grantResults[index] != PackageManager.PERMISSION_GRANTED
            }

            if (deniedPermissions.isEmpty()) {
                Log.i(TAG, "✅ All permissions granted")
                Toast.makeText(this, "All permissions granted", Toast.LENGTH_SHORT).show()
            } else {
                Log.w(TAG, "⚠️ Denied permissions: ${deniedPermissions.joinToString()}")
                Toast.makeText(this, "Some permissions denied. App functionality may be limited.", Toast.LENGTH_LONG).show()
            }
        }
    }

    /**
     * Clean up resources
     */
    override fun onDestroy() {
        super.onDestroy()
        Log.i(TAG, "🧹 Cleaning up resources...")

        try {
            clipInference?.release()
            clipInference = null
            llamaInference?.release()
            llamaInference = null
            currentBitmap?.recycle()
            currentBitmap = null
        } catch (e: Exception) {
            Log.e(TAG, "❌ Error during cleanup: ${e.message}", e)
        }

        Log.i(TAG, "✅ MainActivity destroyed")
    }

    override fun onPause() {
        super.onPause()
        Log.i(TAG, "⏸️ Activity paused")
    }

    override fun onResume() {
        super.onResume()
        Log.i(TAG, "▶️ Activity resumed")
    }
}