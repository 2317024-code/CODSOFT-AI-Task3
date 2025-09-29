# Image Captioning System
This project implements an image captioning system using deep learning with TensorFlow and Keras. It combines a pre-trained ResNet50 model for image feature extraction with an LSTM-based model for generating text captions for images.

## Project Overview

- Language: Python
- Libraries: TensorFlow, Keras, NumPy, OpenCV
- Models:
- ResNet50 for image feature extraction
- LSTM-based neural network for caption generation

## Features:
- Generates captions for input images (JPG/PNG)
- Displays the image with the generated caption overlaid
- Handles basic error cases (e.g., invalid file paths, unsupported formats)
- Uses a predefined vocabulary for caption generation
- Fallback captions for specific cases (e.g., images with "cat" in the filename)

## Prerequisites
- Python 3.9
- Required libraries: pip install tensorflow numpy opencv-python
- Access to a pre-trained ResNet50 model

# How It Works

## Image Feature Extraction:
- Uses ResNet50 (pre-trained on ImageNet) to extract 2048-dimensional feature vectors from input images.
- Images are resized to 224x224 pixels and preprocessed using ResNet50's preprocessing function.

## Caption Generation:
- Combines image features with a partially generated caption sequence.
- Uses an LSTM-based model with an embedding layer for text and a dense layer for image features.
- Predicts the next word in the sequence iteratively, up to a maximum length (m=10).
- Applies temperature-based sampling (temp=0.7) for diverse caption generation.

## Model Architecture:
- Input: Image features (2048-dim) and partial caption sequence (padded to length 10).
- Layers: 
- Dense layer (256 units, ReLU) for image features
- Embedding + LSTM (256 units) for text
- Concatenation of image and text features
- Dense layers for final softmax output over the vocabulary
- Loss: Categorical cross-entropy
- Optimizer: Adam

## Vocabulary:
- Predefined vocabulary with tokens like <start>, <end>, and words such as "dog," "cat," "park," etc.
- Tokenizer maps words to indices for sequence processing.

## Fallback Logic:
- If the generated caption is repetitive or empty, it defaults to:
- "A cat is sitting on the grass" for images with "cat" in the filename
- "A generic scene" otherwise

## Display: 

Uses OpenCV to display the input image with the generated caption overlaid.

## Usage:
- Enter the path to a JPG or PNG image when prompted.
- The system processes the image, generates a caption and displays the image with the caption.
- Type quit to exit.

## Code Structure
- Imports: TensorFlow, Keras, NumPy, OpenCV for image and model processing.
- Vocabulary and Tokenizer: Defines a small vocabulary and initializes a Keras Tokenizer.
- Feature Extraction: extractf(imgpath): Loads, preprocesses, and extracts features from an image using ResNet50.
- Model Definition: caption(vsize, m): Builds the captioning model combining image and text inputs.
- Caption Generation: gencap(imgpath, m=10): Generates a caption for the input image using the model and tokenizer.
- Main Program: Interactive loop for inputting image paths, generating captions, and displaying results.

## Limitations
- The model is not trained; it relies on the predefined vocabulary and random sampling, so captions may not always be accurate or meaningful.
- The vocabulary is limited, restricting the variety of generated captions.
- Requires significant computational resources for ResNet50 and LSTM predictions.
- Only supports JPG and PNG formats.
