package com.example.edgeai.ui

import android.os.Bundle
import android.util.Log
import android.widget.Button
import android.widget.EditText
import android.widget.TextView
import android.widget.Toast
import android.app.Activity
import com.example.edgeai.ml.ExecutorTorchGemma3
import com.example.edgeai.R

/**
 * Test Activity for ExecutorTorch Gemma3-1B Integration
 * 
 * This activity demonstrates the integration of Gemma3-1B model with ExecuTorch + QNN backend
 * Based on: https://github.com/pytorch/executorch/tree/main/examples/qualcomm/oss_scripts/llama#gemma3-1b
 */
class ExecutorTorchGemma3TestActivity : Activity() {

    companion object {
        private const val TAG = "ExecutorTorchGemma3Test"
    }

    private lateinit var gemma3Inference: ExecutorTorchGemma3
    private lateinit var promptEditText: EditText
    private lateinit var responseTextView: TextView
    private lateinit var modelInfoTextView: TextView
    private lateinit var generateButton: Button
    private lateinit var initializeButton: Button
    private lateinit var testButton: Button

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_executor_torch_gemma3_test)

        Log.i(TAG, "🚀 Starting ExecutorTorch Gemma3-1B Test Activity")

        // Initialize UI components
        initializeViews()
        
        // Initialize Gemma3-1B inference engine
        gemma3Inference = ExecutorTorchGemma3(this)
        
        // Set up button listeners
        setupButtonListeners()
        
        // Display initial model info
        updateModelInfo()
        
        Log.i(TAG, "✅ ExecutorTorch Gemma3-1B Test Activity initialized")
    }

    private fun initializeViews() {
        promptEditText = findViewById(R.id.promptEditText)
        responseTextView = findViewById(R.id.responseTextView)
        modelInfoTextView = findViewById(R.id.modelInfoTextView)
        generateButton = findViewById(R.id.generateButton)
        initializeButton = findViewById(R.id.initializeButton)
        testButton = findViewById(R.id.testButton)
        
        // Set default prompt
        promptEditText.setText("I would like to learn python, could you teach me with a simple example?")
    }

    private fun setupButtonListeners() {
        initializeButton.setOnClickListener {
            initializeGemma3Model()
        }

        generateButton.setOnClickListener {
            generateResponse()
        }

        testButton.setOnClickListener {
            runGemma3Tests()
        }
    }

    private fun initializeGemma3Model() {
        Log.i(TAG, "🔧 Initializing Gemma3-1B model...")
        
        initializeButton.isEnabled = false
        initializeButton.text = "Initializing..."
        
        Thread {
            try {
                val success = gemma3Inference.initializeGemma3()
                
                runOnUiThread {
                    if (success) {
                        Toast.makeText(this, "✅ Gemma3-1B initialized successfully!", Toast.LENGTH_LONG).show()
                        Log.i(TAG, "✅ Gemma3-1B initialization successful")
                    } else {
                        Toast.makeText(this, "❌ Gemma3-1B initialization failed", Toast.LENGTH_LONG).show()
                        Log.e(TAG, "❌ Gemma3-1B initialization failed")
                    }
                    
                    initializeButton.isEnabled = true
                    initializeButton.text = "Initialize Gemma3-1B"
                    updateModelInfo()
                }
                
            } catch (e: Exception) {
                Log.e(TAG, "❌ Exception during Gemma3-1B initialization", e)
                runOnUiThread {
                    Toast.makeText(this, "❌ Error: ${e.message}", Toast.LENGTH_LONG).show()
                    initializeButton.isEnabled = true
                    initializeButton.text = "Initialize Gemma3-1B"
                }
            }
        }.start()
    }

    private fun generateResponse() {
        val prompt = promptEditText.text.toString().trim()
        
        if (prompt.isEmpty()) {
            Toast.makeText(this, "Please enter a prompt", Toast.LENGTH_SHORT).show()
            return
        }
        
        Log.i(TAG, "🤖 Generating response for prompt: $prompt")
        
        generateButton.isEnabled = false
        generateButton.text = "Generating..."
        
        Thread {
            try {
                val response = gemma3Inference.generateResponse(prompt, 100, 0.0f)
                
                runOnUiThread {
                    responseTextView.text = response
                    generateButton.isEnabled = true
                    generateButton.text = "Generate Response"
                    
                    Log.i(TAG, "✅ Response generated successfully")
                    Toast.makeText(this, "Response generated!", Toast.LENGTH_SHORT).show()
                }
                
            } catch (e: Exception) {
                Log.e(TAG, "❌ Exception during response generation", e)
                runOnUiThread {
                    responseTextView.text = "Error: ${e.message}"
                    generateButton.isEnabled = true
                    generateButton.text = "Generate Response"
                    Toast.makeText(this, "❌ Error: ${e.message}", Toast.LENGTH_LONG).show()
                }
            }
        }.start()
    }

    private fun runGemma3Tests() {
        Log.i(TAG, "🧪 Running Gemma3-1B tests...")
        
        testButton.isEnabled = false
        testButton.text = "Running Tests..."
        
        Thread {
            try {
                val testResults = gemma3Inference.runGemma3Tests()
                
                runOnUiThread {
                    val testReport = buildString {
                        appendLine("🧪 Gemma3-1B Test Results:")
                        appendLine("=" + "=".repeat(49))
                        
                        testResults.forEach { (testName, result) ->
                            appendLine("\n📋 $testName:")
                            appendLine(result)
                            appendLine("-" + "-".repeat(29))
                        }
                        
                        appendLine("\n📊 Model Statistics:")
                        val stats = gemma3Inference.getModelStats()
                        stats.forEach { (key, value) ->
                            appendLine("$key: $value")
                        }
                    }
                    
                    responseTextView.text = testReport
                    testButton.isEnabled = true
                    testButton.text = "Run Tests"
                    
                    Log.i(TAG, "✅ Tests completed successfully")
                    Toast.makeText(this, "Tests completed!", Toast.LENGTH_SHORT).show()
                }
                
            } catch (e: Exception) {
                Log.e(TAG, "❌ Exception during tests", e)
                runOnUiThread {
                    responseTextView.text = "Test Error: ${e.message}"
                    testButton.isEnabled = true
                    testButton.text = "Run Tests"
                    Toast.makeText(this, "❌ Test Error: ${e.message}", Toast.LENGTH_LONG).show()
                }
            }
        }.start()
    }

    private fun updateModelInfo() {
        val modelInfo = gemma3Inference.getModelInfo()
        modelInfoTextView.text = modelInfo
        
        val isInitialized = gemma3Inference.isInitialized()
        generateButton.isEnabled = isInitialized
        testButton.isEnabled = isInitialized
        
        if (isInitialized) {
            Log.i(TAG, "✅ Gemma3-1B model is ready for inference")
        } else {
            Log.i(TAG, "⚠️ Gemma3-1B model needs initialization")
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        Log.i(TAG, "🏁 ExecutorTorch Gemma3-1B Test Activity destroyed")
    }
}
