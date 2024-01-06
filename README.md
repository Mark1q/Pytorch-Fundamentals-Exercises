# PyTorch Deep Learning Exercises

This repository contains Python code to solve and explain each step of the provided deep learning exercises using PyTorch. The exercises cover various aspects such as creating tensors, matrix multiplication, setting random seeds, utilizing GPU, and performing operations on tensors.

## Exercise 1: Documentation Reading

To start with deep learning, it's essential to get familiar with the PyTorch documentation. Spend 10 minutes reading about `torch.Tensor` and `torch.cuda`. The goal is not complete understanding but awareness.

## Exercise 2: Create a Random Tensor

```python
import torch

# Create a random tensor with shape (7, 7)
random_tensor = torch.rand(7, 7)
```

## Exercise 3: Matrix Multiplication

```python
# Create another random tensor with shape (1, 7)
second_tensor = torch.rand(1, 7)

# Transpose the second tensor
second_tensor = second_tensor.T

# Perform matrix multiplication
result_tensor = torch.matmul(random_tensor, second_tensor)
```

## Exercise 4: Set Random Seed

```python
# Set random seed to 0
torch.manual_seed(0)

# Repeat exercises 2 and 3
random_tensor = torch.rand(7, 7)
second_tensor = torch.rand(1, 7).T
result_tensor = torch.matmul(random_tensor, second_tensor)
```

## Exercise 5: GPU Random Seed

Check the PyTorch documentation on `torch.cuda` to find if there's a GPU equivalent for setting the random seed. If found, set the GPU random seed to 1234.

```python
# Set GPU random seed to 1234
torch.cuda.manual_seed(1234)
```

## Exercise 6: Create and Move Tensors to GPU

```python
# Set random seed to 1234
torch.manual_seed(1234)

# Create two random tensors of shape (2, 3)
tensor1 = torch.rand(2, 3)
tensor2 = torch.rand(2, 3)

# Move tensors to GPU
tensor1 = tensor1.to('cuda')
tensor2 = tensor2.to('cuda')
```

## Exercise 7: Matrix Multiplication on GPU

```python
# Perform matrix multiplication on GPU
result_tensor_gpu = torch.matmul(tensor1, tensor2)
```

## Exercise 8: Find Max and Min Values

```python
# Find maximum and minimum values of the output
max_value = torch.max(result_tensor_gpu)
min_value = torch.min(result_tensor_gpu)
```

## Exercise 9: Find Max and Min Indices

```python
# Find indices of maximum and minimum values
max_index = torch.argmax(result_tensor_gpu)
min_index = torch.argmin(result_tensor_gpu)
```

## Exercise 10: Remove Dimensions from Tensor

```python
# Create a random tensor with shape (1, 1, 1, 10)
random_tensor_4d = torch.rand(1, 1, 1, 10, seed=7)

# Remove dimensions to get a tensor of shape (10)
tensor_1d = random_tensor_4d.squeeze()

# Print shapes of both tensors
print("Original Tensor Shape:", random_tensor_4d.shape)
print("Modified Tensor Shape:", tensor_1d.shape)
```

Feel free to explore and run each exercise separately. The comments in the code provide explanations for each step.
