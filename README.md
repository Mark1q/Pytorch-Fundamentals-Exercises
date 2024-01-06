# Pytorch-Fundamentals-Exercises

PyTorch Fundamentals Exercises from the learnpytorch.io

## Exercise 1
> **Documentation reading** - A big part of deep learning is getting familiar with the documentation of a certain framework you're using. We'll be using the PyTorch documentation a lot throughout the rest of this course. So I'd recommend spending 10-minutes reading the following. See the documentation on [torch.Tensor](https://pytorch.org/docs/stable/tensors.html#torch-tensor) and for [torch.cuda](https://pytorch.org/docs/master/notes/cuda.html#cuda-semantics)

## Exercise 2
> Create a random tensor with shape (7, 7)

First, we'll start importing the torch library(_See setup [here](https://pytorch.org/get-started/locally/)_)
```python 
import torch
```
Then, we will initialize a tensor with the function `torch.rand()` and specify the shape by providing the `size` argument with the desired shape

```python
random_tensor = torch.rand(size=[7,7])
```

You can check what shape a tensor has by using `<tensor_name>.shape`, where _<tensor_name>_ is **random_tensor** in our case

```python
print(random_tensor.shape)
```
```python
torch.Size([7, 7])
```

## Exercise 3
> Perform a matrix multiplication on the tensor from 2 with another random tensor with shape (1, 7)

Let's declare another tensor with the shape (1,7) like in the last exercise

```python
another_random_tensor = torch.rand(size=[1,7])
```

To multiply 2 tensors , we'll be using `torch.matmul()` for that and declare another tensor that will store the resulting tensor

```python
multiplication_tensor = torch.matmul(random_tensor,another_random_tensor)
```
But when we try to run this code, we'll encounter an error message that will tell us that the 2 tensors can't be multiplied because of their shape
```RuntimeError: mat1 and mat2 shapes cannot be multiplied (7x7 and 1x7)```


