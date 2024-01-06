# Pytorch-Fundamentals-Exercises

PyTorch Fundamentals Exercises from the learnpytorch.io

## Exercise 1
> Documentation reading - A big part of deep learning is getting familiar with the documentation of a certain framework you're using. We'll be using the PyTorch documentation a lot throughout the rest of this course. So I'd recommend spending 10-minutes reading the following. See the documentation on [torch.Tensor](https://pytorch.org/docs/stable/tensors.html#torch-tensor) and for `torch.cuda`

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



