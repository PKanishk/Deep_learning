import torch
import torch.nn as nn
import torch.optim as optim

# Hyperparameters
input_size = 10  # Number of input features
hidden_size = 20 # Number of features in hidden state
output_size = 1  # Number of output features
sequence_length = 10 # Length of input sequence
num_epochs = 100 # Number of epochs
learning_rate = 0.001

# Generate dummy data
x_train = torch.randn(100, sequence_length, input_size) # (batch_size, sequence_length, input_size)
y_train = torch.randn(100, output_size)    # (batch_size, output_size)

# Initialize weights and biases
Wxh = torch.randn(hidden_size, input_size, requires_grad=True) * 0.01
Whh = torch.randn(hidden_size, hidden_size, requires_grad=True) * 0.01
Why = torch.randn(output_size, hidden_size, requires_grad=True) * 0.01
bh = torch.zeros(hidden_size, requires_grad=True)
by = torch.zeros(output_size, requires_grad=True)

# Optimizer
optimizer = optim.SGD([Wxh, Whh, Why, bh, by], lr=learning_rate)

# ReLU activation function
def relu(x):
    return torch.maximum(x, torch.tensor(0.0))

# Training loop
for epoch in range(num_epochs):
    total_loss = 0
    for i in range(x_train.size(0)):
        inputs = x_train[i]
        target = y_train[i]

        # Forward pass
        h = torch.zeros(hidden_size)
        for t in range(sequence_length):
            x_t = inputs[t]
            h = relu(torch.matmul(Wxh, x_t) + torch.matmul(Whh, h) + bh)
        
        y_pred = torch.matmul(Why, h) + by
        loss = (y_pred - target).pow(2).sum()

        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/x_train.size(0):.4f}')

# Testing the model
test_input = torch.randn(sequence_length, input_size) # Single sample with sequence length
h = torch.zeros(hidden_size)
for t in range(sequence_length):
    x_t = test_input[t]
    h = relu(torch.matmul(Wxh, x_t) + torch.matmul(Whh, h) + bh)
test_output = torch.matmul(Why, h) + by
print("Test Output:", test_output)
