# XOR Neural Network

This project implements a basic **feedforward neural network** to solve the **XOR problem** using Python and Numpy. The XOR problem is a classic example in machine learning that demonstrates the capability of neural networks to learn patterns that are not linearly separable.

## Project Overview

### Problem Statement
The XOR function takes two binary inputs and produces a binary output. The output is true (1) if one and only one of the inputs is true; otherwise, it's false (0). The truth table for XOR is as follows:

| Input 1 | Input 2 | Output |
|---------|---------|--------|
|   0     |   0     |   0    |
|   0     |   1     |   1    |
|   1     |   0     |   1    |
|   1     |   1     |   0    |

### Neural Network Architecture
The neural network consists of:
- **Input Layer**: Two neurons for the inputs.
- **Hidden Layer**: Two neurons with sigmoid activation functions.
- **Output Layer**: One neuron that produces a binary output.

### How It Works
1. **Feedforward**: The input data is passed through the network, and the outputs are calculated.
2. **Loss Calculation**: The binary cross-entropy loss is computed to evaluate the network's performance.
3. **Backpropagation**: The weights and biases are adjusted based on the output error and the computed gradients.

### Training Process
The model is trained over **10,000 epochs**, where it iteratively updates the weights to minimize the loss. The following metrics are calculated after training:
- **Accuracy**: Percentage of correctly predicted outputs.
- **Precision**: Measure of the accuracy of positive predictions.
- **Recall**: Measure of the ability to find all positive samples.
- **F1 Score**: Harmonic mean of precision and recall.

### Visualization
- **Confusion Matrix**: A visual representation of the model's performance, showing true vs. predicted classifications.
- **Decision Boundary**: A contour plot that visualizes the decision regions learned by the model.

## Conclusion
This project effectively demonstrates the ability of a simple neural network to learn the XOR function, handling a non-linearly separable problem. The decision boundary and confusion matrix visually affirm the model's capability.

Feel free to explore the code further or modify it for additional experiments!
