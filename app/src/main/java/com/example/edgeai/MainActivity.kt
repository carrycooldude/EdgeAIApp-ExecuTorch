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
import android.widget.TextView
import android.widget.Toast
import androidx.core.content.FileProvider
import com.example.edgeai.ml.ExecutorTorchCLIP
import kotlinx.coroutines.*
import java.io.File
import java.io.IOException
import java.text.SimpleDateFormat
import java.util.*

class MainActivity : Activity() {

    private lateinit var imageView: ImageView
    private lateinit var resultTextView: TextView
    private lateinit var captureButton: Button
    private lateinit var galleryButton: Button
    private lateinit var inferenceButton: Button
    private lateinit var textInput: EditText

    private var clipInference: ExecutorTorchCLIP? = null
    private var currentBitmap: Bitmap? = null
    private var currentPhotoPath: String = ""

    companion object {
        private const val TAG = "EdgeAI_CLIP"
        private const val REQUEST_IMAGE_CAPTURE = 1
        private const val REQUEST_GALLERY_IMAGE = 2
        private const val REQUEST_PERMISSIONS = 3
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        Log.i(TAG, "Starting EdgeAI CLIP Demo (ExecuTorch)")

        initializeViews()
        requestPermissions()
        initializeModels()
    }
    
    private fun initializeViews() {
        imageView = findViewById(R.id.imageView)
        resultTextView = findViewById(R.id.resultTextView)
        captureButton = findViewById(R.id.captureButton)
        galleryButton = findViewById(R.id.galleryButton)
        inferenceButton = findViewById(R.id.inferenceButton)
        textInput = findViewById(R.id.textInput)
        
        captureButton.setOnClickListener {
            val takePictureIntent = Intent(MediaStore.ACTION_IMAGE_CAPTURE)
            if (takePictureIntent.resolveActivity(packageManager) != null) {
                val photoFile: File? = try {
                    createImageFile()
                } catch (ex: IOException) {
                    null
                }
                photoFile?.also {
                    val photoURI: Uri = FileProvider.getUriForFile(
                        this,
                        "com.example.edgeai.fileprovider",
                        it
                    )
                    takePictureIntent.putExtra(MediaStore.EXTRA_OUTPUT, photoURI)
                    startActivityForResult(takePictureIntent, REQUEST_IMAGE_CAPTURE)
                }
            }
        }

        galleryButton.setOnClickListener {
            val intent = Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI)
            startActivityForResult(intent, REQUEST_GALLERY_IMAGE)
        }

        inferenceButton.setOnClickListener {
            runCLIPInference()
        }
        
        inferenceButton.isEnabled = false
    }

    private fun requestPermissions() {
        if (checkSelfPermission(Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED ||
            checkSelfPermission(Manifest.permission.READ_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(arrayOf(Manifest.permission.CAMERA, Manifest.permission.READ_EXTERNAL_STORAGE), REQUEST_PERMISSIONS)
        }
    }

    @Deprecated("Deprecated in Java")
    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (resultCode == Activity.RESULT_OK) {
            when (requestCode) {
                REQUEST_IMAGE_CAPTURE -> {
                    setPic()
                }
                REQUEST_GALLERY_IMAGE -> {
                    data?.data?.let { uri ->
                        try {
                            val inputStream = contentResolver.openInputStream(uri)
                            currentBitmap = BitmapFactory.decodeStream(inputStream)
                            imageView.setImageBitmap(currentBitmap)
                            resultTextView.text = "Snapshot loaded. Ready for analysis."
                            inferenceButton.isEnabled = clipInference != null
                        } catch (e: Exception) {
                            Log.e(TAG, "Failed to load image", e)
                        }
                    }
                }
            }
        }
    }

    @Throws(IOException::class)
    private fun createImageFile(): File {
        val timeStamp: String = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.getDefault()).format(Date())
        val storageDir: File? = getExternalFilesDir(Environment.DIRECTORY_PICTURES)
        return File.createTempFile(
            "JPEG_${timeStamp}_",
            ".jpg",
            storageDir
        ).apply {
            currentPhotoPath = absolutePath
        }
    }

    private fun setPic() {
        val targetW = imageView.width.takeIf { it > 0 } ?: 1024
        val targetH = imageView.height.takeIf { it > 0 } ?: 1024

        val bmOptions = BitmapFactory.Options()
        bmOptions.inJustDecodeBounds = true
        BitmapFactory.decodeFile(currentPhotoPath, bmOptions)
        val photoW = bmOptions.outWidth
        val photoH = bmOptions.outHeight

        val scaleFactor = Math.max(1, Math.min(photoW / targetW, photoH / targetH))

        bmOptions.inJustDecodeBounds = false
        bmOptions.inSampleSize = scaleFactor
        bmOptions.inPurgeable = true

        currentBitmap = BitmapFactory.decodeFile(currentPhotoPath, bmOptions)
        imageView.setImageBitmap(currentBitmap)
        resultTextView.text = "Photo captured. Ready for analysis."
        inferenceButton.isEnabled = clipInference != null
    }

    private fun initializeModels() {
        CoroutineScope(Dispatchers.IO).launch {
            try {
                Log.i(TAG, "Initializing CLIP model...")

                var clipSuccess = false
                var clipErrorMsg = ""
                try {
                    Log.i(TAG, "Creating ExecutorTorchCLIP instance...")
                    clipInference = ExecutorTorchCLIP(this@MainActivity)
                    clipSuccess = clipInference?.initialize() == true
                    if (!clipSuccess) clipErrorMsg = "ExecuTorch initialization returned false"
                    Log.i(TAG, "CLIP init result: $clipSuccess")
                } catch (e: Exception) {
                    clipErrorMsg = "Exception: ${e.message}"
                    Log.e(TAG, "CLIP initialization failed: ${e.message}", e)
                    clipSuccess = false
                }

                withContext(Dispatchers.Main) {
                    if (clipSuccess) {
                        resultTextView.text = "ExecuTorch CLIP Ready. Select an image to start."
                        inferenceButton.isEnabled = true
                        Toast.makeText(this@MainActivity, "CLIP Model Loaded", Toast.LENGTH_SHORT).show()
                    } else {
                        resultTextView.text = "Initialization Failed: $clipErrorMsg"
                        Toast.makeText(this@MainActivity, "Error: $clipErrorMsg", Toast.LENGTH_LONG).show()
                    }
                }

            } catch (e: Exception) {
                Log.e(TAG, "Model initialization error: ${e.message}", e)
                withContext(Dispatchers.Main) {
                    resultTextView.text = "Model initialization error: ${e.message}"
                }
            }
        }
    }

    private fun runCLIPInference() {
        val bitmap = currentBitmap
        val question = textInput.text.toString().trim()
        
        if (bitmap == null) {
            Toast.makeText(this, "Please select an image first", Toast.LENGTH_SHORT).show()
            return
        }

        if (clipInference == null) {
            Toast.makeText(this, "CLIP model not initialized", Toast.LENGTH_SHORT).show()
            return
        }

        inferenceButton.isEnabled = false
        resultTextView.text = "Running ExecuTorch CLIP inference..."

        Log.i(TAG, "Starting CLIP inference...")

        CoroutineScope(Dispatchers.IO).launch {
            try {
                val startTime = System.currentTimeMillis()

                var similarity = 0.0f
                var hasText = false
                
                if (question.isNotEmpty()) {
                     hasText = true
                     similarity = clipInference?.computeImageTextSimilarity(bitmap, question) ?: 0.0f
                }

                val imageEmbedding = clipInference?.runImageInference(bitmap)

                val inferenceTime = System.currentTimeMillis() - startTime
                Log.i(TAG, "CLIP inference completed in ${inferenceTime}ms")

                val formattedResults = formatCLIPResults(imageEmbedding, inferenceTime, hasText, question, similarity)
                saveResults(formattedResults)

                withContext(Dispatchers.Main) {
                    resultTextView.text = formattedResults
                    inferenceButton.isEnabled = true
                }

            } catch (e: Exception) {
                Log.e(TAG, "CLIP inference failed: ${e.message}", e)
                withContext(Dispatchers.Main) {
                    resultTextView.text = "CLIP inference failed: ${e.message}"
                    inferenceButton.isEnabled = true
                }
            }
        }
    }
    
    private fun formatCLIPResults(data: FloatArray?, inferenceTime: Long, hasText: Boolean, question: String, score: Float): String {
        if (data == null || data.isEmpty()) {
            return "No inference results received"
        }

        val builder = StringBuilder()
        
        if (hasText) {
             builder.append("Similarity Analysis\n")
             builder.append("--------------------\n")
             builder.append("Question: $question\n")
             builder.append("Score:    ${"%.4f".format(score)}\n")
             builder.append("--------------------\n\n")
        } else {
             builder.append("Image Analysis Complete\n")
             builder.append("--------------------\n")
             builder.append("Enter a question to analyze similarity.\n\n")
        }

        builder.append("Inference Time: ${inferenceTime}ms\n")
        builder.append("Embedding Size: ${data.size} floats\n")
        builder.append("ExecuTorch + QNN (HTP)")
        
        return builder.toString()
    }

    private fun saveResults(results: String) {
        try {
            val timestamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.getDefault()).format(Date())
            val filename = "edgeai_clip_results_$timestamp.txt"
            val documentsDir = getExternalFilesDir(Environment.DIRECTORY_DOCUMENTS)
            val file = File(documentsDir, filename)
            file.writeText(results)
        } catch (e: Exception) {
            Log.e(TAG, "Failed to save results: ${e.message}", e)
        }
    }

    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<out String>, grantResults: IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == REQUEST_PERMISSIONS) {
            val deniedPermissions = permissions.filterIndexed { index, _ ->
                grantResults[index] != PackageManager.PERMISSION_GRANTED
            }
            if (deniedPermissions.isEmpty()) {
                Toast.makeText(this, "Permissions granted", Toast.LENGTH_SHORT).show()
            } else {
                Toast.makeText(this, "Permissions denied", Toast.LENGTH_LONG).show()
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        clipInference?.release()
        clipInference = null
        currentBitmap?.recycle()
        currentBitmap = null
    }
}