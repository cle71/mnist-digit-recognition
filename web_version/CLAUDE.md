# MNIST Handwriting Recognition - Web Version

## Project Overview
A web-based MNIST (Modified National Institute of Standards and Technology) digit recognition system that allows users to draw handwritten digits (0-9) on a canvas and get real-time predictions using a deep neural network.

## Current Status
- **Model**: PyTorch-trained neural network (97.92% accuracy on test set)
- **Frontend**: HTML5 Canvas + Vanilla JavaScript
- **Backend**: Python HTTP server
- **Location**: Parent directory contains trained model files

## Key Features
- Real-time handwriting recognition
- Visual confidence scores and probability distribution
- Touch and mouse input support
- Responsive web interface
- No external ML library dependencies in browser (pure JavaScript inference)

## Development Goals
- Improve web interface usability
- Enhance image preprocessing for better recognition
- Add additional features (handwriting samples, model info, etc.)
- Optimize performance

## File Structure
```
web_version/
├── CLAUDE.md           # This file
├── index.html          # Main web interface
├── styles.css          # Styling (if separated)
├── app.js              # JavaScript logic (if separated)
└── README.md           # User documentation
```

## Technical Stack
- **Language**: JavaScript (ES6+), Python
- **Framework**: None (vanilla JS)
- **Model Format**: JSON (PyTorch weights)
- **Server**: Python http.server

## Important Notes
1. Model weights are stored in `../model/tfjs_model/model.json`
2. Server runs on port 8001 (configurable in run_server.py)
3. All code must be in English
4. Comments should be clear and concise

## Next Steps
1. Create production-ready web interface
2. Improve image preprocessing algorithm
3. Add debugging features
4. Create comprehensive user guide
