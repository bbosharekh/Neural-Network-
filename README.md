# Neural Network 


A lightweight, NumPy-only deep learning inference engine built from scratch. This project implements the forward pass of a multi-layer perceptron (MLP) to demonstrate a ground-up understanding of neural network architecture, matrix multiplication, and activation functions.

## 📌 Overview
While modern frameworks like PyTorch and TensorFlow abstract away the underlying math, this project rebuilds the core mechanics of model inference. It is designed to take inputs, propagate them through hidden layers using matrix dot products and ReLU activations, and output raw logits—all using pure linear algebra.

**Key Concepts Demonstrated:**
- Object-Oriented neural network architecture
- Feedforward computation (Inference)
- Non-linear activation functions (ReLU)
- Dynamic hidden layer sizing and weight initialization

## 💻 Quick Start

**Dependencies:** `numpy`

```python
import numpy as np
from network import NeuralNetwork

# 1. Initialize a network (e.g., 784 inputs, two 512-node hidden layers, 10 outputs)
nn = NeuralNetwork(input_size=784, hidden_layers=[512, 512], output_size=10)

# 2. Create some dummy data (e.g., a batch of 5 images, flattened to 784 pixels)
X_batch = np.random.randn(5, 784)

# 3. Run the forward pass (Inference)
predictions = nn.forward(X_batch)

print("Output shape:", predictions.shape) # Expected: (5, 10)
print("Raw Logits for first sample:\n", predictions[0])
