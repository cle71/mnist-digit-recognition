# MNIST Handwriting Recognition System

A deep neural network-based handwriting recognition system that recognizes digits (0-9) using the MNIST dataset.

## Features

- **Deep Neural Network**: Custom trained model with multiple hidden layers
- **Real-time Prediction**: Draw digits and get instant recognition
- **Confidence Scores**: See probability distribution across all digits
- **Web Interface**: User-friendly browser-based canvas for drawing
- **Touch Support**: Works on mobile devices with touch input

## System Architecture

### Backend (Python)
- **Model Training**: TensorFlow/Keras neural network
- **Architecture**: 
  - Input layer: 784 neurons (28×28 pixel flattened)
  - Hidden layer 1: 128 neurons with ReLU activation + Dropout(0.2)
  - Hidden layer 2: 64 neurons with ReLU activation + Dropout(0.2)
  - Hidden layer 3: 32 neurons with ReLU activation
  - Output layer: 10 neurons with softmax activation
- **Dataset**: MNIST (60,000 training samples, 10,000 test samples)
- **Format**: Converted to TensorFlow.js format for web deployment

### Frontend (Web)
- **Framework**: Vanilla JavaScript + TensorFlow.js
- **Canvas**: HTML5 canvas for handwriting input
- **Model Loading**: Loads pre-trained model from local files
- **Visualization**: Real-time confidence bars and probability distribution

## Installation

### Requirements

- Python 3.7 or higher
- pip package manager

### Step 1: Install Dependencies

```bash
pip install tensorflow keras tensorflowjs numpy
```

### Step 2: Train the Model

Run the training script to train and convert the model:

```bash
python train_mnist_model.py
```

This will:
1. Download and load the MNIST dataset
2. Build and train the neural network
3. Evaluate on test data
4. Convert the model to TensorFlow.js format
5. Save files to `model/tfjs_model/` directory

**Expected Output:**
- Training takes 2-5 minutes depending on your system
- Test accuracy: ~97-98%

### Step 3: Start the Web Server

```bash
python run_server.py
```

The server will:
1. Start on `http://localhost:8000`
2. Automatically open your browser (if possible)

## Usage

1. **Draw**: Use your mouse or touch to draw a digit (0-9) on the white canvas
2. **Predict**: Click the "Predict" button to recognize the digit
3. **Results**: View the predicted digit and confidence score
4. **Clear**: Click the "Clear" button to erase and try again

## Model Performance

- **Test Accuracy**: ~97-98%
- **Input Size**: 28×28 grayscale images
- **Output**: 10 classes (digits 0-9)
- **Processing Time**: <100ms per prediction on CPU

## File Structure

```
claude-deep-mnist-pjt/
├── train_mnist_model.py       # Model training script
├── run_server.py              # Web server launcher
├── index.html                 # Web interface
├── model/                     # Model directory
│   ├── tfjs_model/           # Converted TensorFlow.js model
│   │   ├── model.json        # Model architecture
│   │   └── group1-shard1of1  # Model weights
│   └── mnist_model.keras     # Original Keras model
└── README.md                  # This file
```

## Troubleshooting

### Model not loading in browser
- Make sure `train_mnist_model.py` has been run completely
- Check that `model/tfjs_model/` directory exists
- Clear browser cache and refresh the page

### Port 8000 already in use
- Edit `run_server.py` and change `PORT = 8000` to another port (e.g., 8001)

### Python package installation issues
- Try installing with: `pip install --upgrade tensorflow keras tensorflowjs`
- On Windows, you may need: `python -m pip install ...`

## How It Works

1. **Input Processing**: Drawing is converted to 28×28 grayscale image
2. **Normalization**: Pixel values are normalized to 0-1 range
3. **Model Inference**: Image is passed through the trained neural network
4. **Prediction**: Output layer produces 10 probability scores (one per digit)
5. **Result Display**: Highest probability is shown with confidence visualization

## Technologies Used

- **TensorFlow/Keras**: Deep learning framework
- **TensorFlow.js**: JavaScript ML library for browser inference
- **HTML5 Canvas**: Drawing interface
- **Python**: Backend scripting

## Performance Notes

- Model training: 2-5 minutes
- Prediction time: <100ms
- Model size: ~2-3 MB
- Browser compatibility: Modern browsers with JavaScript enabled

## License

Educational project - Free to use and modify

## Author Notes

This system demonstrates:
- Deep learning model training with Keras
- Model conversion for web deployment
- Real-time ML inference in the browser
- Interactive web interface with canvas drawing
