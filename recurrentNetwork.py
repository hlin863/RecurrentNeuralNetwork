import numpy as np


class SimpleRNN:
    def __init__(self, input_dim, output_dim):
        # Initialize weights and bias
        self.w = np.random.randn(input_dim, output_dim)
        self.b = np.zeros(output_dim)
        self.y_prev = np.zeros(output_dim)

    def step(self, x):
        # Compute the new output
        y = self._activation(np.dot(x, self.w) + self.b + self.y_prev)
        # Update the previous output for the next step
        self.y_prev = y
        return y

    def _activation(self, z):
        # For the sake of simplicity, I'm using the tanh activation function.
        # However, you can use any other activation function depending on your needs.
        return np.tanh(z)


# Example
input_dim = 1
output_dim = 1
rnn = SimpleRNN(input_dim, output_dim)
x_sequence = [0.5, 1.0, 1.5, 2.0, 2.5]
outputs = []

for x in x_sequence:
    outputs.append(rnn.step([x]))

print(outputs)
