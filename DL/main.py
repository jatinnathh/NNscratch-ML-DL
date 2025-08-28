import numpy as np
import math 


'''
Forward Pass:
   - Input data flows through layers (linear transformation + activation functions).
   - Output is predicted probabilities (for classification tasks).
    Z = W·X + b   (Linear Transformation)
    A = Activation(Z)

Loss Function:
   - We use Cross-Entropy Loss for classification.
   - Loss penalizes incorrect predictions and rewards correct ones.
   Loss (Cross-Entropy for Classification):
   L = - Σ y_true * log(y_pred)

Backpropagation:
   - Computes gradients of weights using chain rule of calculus.
   - Updates weights via Gradient Descent:
        w_new = w_old - learning_rate * gradient
   - Uses chain rule to compute gradients of loss w.r.t. weights
   - Updates weights: W = W - lr * dW

Training:
   - Iterate multiple epochs (passes over dataset).
   - Each iteration: forward pass → loss → backward pass → weight update.


'''
class NeuralNetwork:
    def __init__(self, layers, activations, lr=0.01, seed=42):
        """
        Parameters:
        -----------
        layer_sizes : list
            Example [input_dim, hidden1, hidden2, ..., output_dim]
        activations : list
            List of activation functions for each hidden/output layer
            Options: 'relu', 'sigmoid', 'tanh', 'softmax'
        lr : float
            Learning rate for gradient descent
        """
        np.random.seed(seed)
        self.layers = layers
        self.activations = activations
        self.lr = lr
        self.params = self._init_weights()

    