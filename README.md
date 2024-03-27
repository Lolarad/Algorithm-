# Algorithm-
Crypto synthetic numbers dirty bits
import numpy as np

# Define custom activation functions
def custom_softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def custom_sin(x):
    return np.sin(x)

def custom_cos(x):
    return np.cos(x)

# Define loss function
def cross_entropy(y_true, y_pred):
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

# Define model architecture
class DeepLearningModel:
    def __init__(self, input_dim, output_dim):
        self.weights = np.random.randn(output_dim, input_dim)
        self.bias = np.zeros((output_dim, 1))

    def forward(self, X):
        Z = np.dot(self.weights, X) + self.bias
        A = custom_softmax(Z)
        return A

    def backward(self, X, y_true, learning_rate):
        m = X.shape[1]
        dZ = A - y_true
        dW = 1/m * np.dot(dZ, X.T)
        db = 1/m * np.sum(dZ, axis=1, keepdims=True)

        self.weights -= learning_rate * dW
        self.bias -= learning_rate * db

# Initialize random number generators
rng1 = RandomNumberGenerator(seed=42)
rng2 = RandomNumberGenerator(seed=123)

# Generate random numbers
random_numbers_1 = rng1.generate_random_numbers(100)
random_numbers_2 = rng2.generate_random_numbers(100)

# Initialize deep learning model
input_dim = 1  # Adjust based on your dataset
output_dim = 1  # Adjust based on your dataset
model = DeepLearningModel(input_dim, output_dim)

# Gradient descent parameters
learning_rate = 0.999999999
epochs = 1000

# Train model to bring synthetic numbers closer together using gradient descent
for epoch in range(epochs):
    for x1, x2 in zip(random_numbers_1, random_numbers_2):
        dirty_bit = np.random.rand() * 2 - 1  # Generate dirty bit between -1 and 1
        y_true = np.sqrt(x1) + np.sqrt(x2) + dirty_bit  # Adjusted spacing of numbers with square root and dirty bit
        model.backward(np.array([[x1 + x2]]).T, np.array([[y_true]]), learning_rate)

# Linear descent to reduce error rate
for epoch in range(epochs):
    for x1, x2 in zip(random_numbers_1, random_numbers_2):
        dirty_bit = np.random.rand() * 2 - 1  # Generate dirty bit between -1 and 1
        y_true = np.sqrt(x1) + np.sqrt(x2) + dirty_bit  # Adjusted spacing of numbers with square root and dirty bit
        y_pred = model.forward(np.array([[x1 + x2]]).T)
        error = y_pred - y_true
        model.weights -= learning_rate * error * (x1 + x2)
        model.bias -= learning_rate * error

# Print final weights and bias
print("Final weights:", model.weights)
print("Final bias:", model.bias)
