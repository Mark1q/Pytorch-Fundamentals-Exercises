# Pytorch-Fundamentals-Exercises

This repository contains Python code to solve and explain each step of the provided deep learning exercises using PyTorch. The exercises cover various aspects such as creating tensors, matrix multiplication, setting random seeds, utilizing GPU, and performing operations on tensors.

<br>
(All the code is in the .py file attached to this repo)

## Exercise 1
> **Documentation reading** - A big part of deep learning is getting familiar with the documentation of a certain framework you're using. We'll be using the PyTorch documentation a lot throughout the rest of this course. So I'd recommend spending 10-minutes reading the following. See the documentation on [torch.Tensor](https://pytorch.org/docs/stable/tensors.html#torch-tensor) and for [torch.cuda](https://pytorch.org/docs/master/notes/cuda.html#cuda-semantics)

## Exercise 2
> Create a random tensor with shape (7, 7)

First, we'll start importing the torch library(_See setup [here](https://pytorch.org/get-started/locally/)_).
```python 
import torch
```
Then, we will initialize a tensor with the function `torch.rand()` and specify the shape by providing the `size` argument with the desired shape.

```python
random_tensor = torch.rand(size=[7,7])
```

You can check what shape a tensor has by using `<tensor_name>.shape`, where _<tensor_name>_ is **random_tensor** in our case.

```python
print(random_tensor.shape)
```
```python
torch.Size([7, 7])
```


## Exercise 3
> Perform a matrix multiplication on the tensor from 2 with another random tensor with shape (1, 7)

Let's declare another tensor with the shape (1,7) like in the last exercise.

```python
another_random_tensor = torch.rand(size=[1,7])
```

To multiply 2 tensors , we'll be using `torch.matmul()` which takes in 2 tensors as arguments and returns the matrix multiplication of the two.

```python
multiplication_tensor = torch.matmul(random_tensor,another_random_tensor)
```
But when we try to run this code, we'll encounter an error message that will tell us that the 2 tensors can't be multiplied because of their shape.
```
Traceback (most recent call last):
  File "main.py", line 10, in <module>
    multiplication_tensor = torch.matmul(random_tensor,another_random_tensor)
RuntimeError: mat1 and mat2 shapes cannot be multiplied (7x7 and 1x7)
```

That's because, for matrix multiplication to be possible :
+ _the **number of columns** in the first matrix_ **must match** _the **number of rows** in the second matrix_, the resulting matrix having **the number of rows of the first** and **the number of columns of the second matrix**, as seen in the photo below(_source:StudyPug_)

![Photo ilustrating the rule the matrix multiplication has to follow](https://dmn92m25mtw4z.cloudfront.net/img_set/la-1-4-x-1-article/v1/la-1-4-x-1-article-878w.png)

In our scenario, we have a tensor that has the shape [7,7] and one who has the shape [1,7]. We can see that the _number of columns(7)_ in our first tensor doesn't match the _number of rows(1)_ in our second tensor.<br><br>
To fix this, we can transpose the second tensor using the `.T` attribute , which swaps the columns with the rows in our tensor.<br><br>

We can see that taking place when we check for the shape of the two matrices(the original and the transposed one).

```python
original shape : torch.Size([1, 7])
transposed shape : torch.Size([7, 1])
```

Now we need to replace `another_random_tensor` in the `torch.matmul()` function with `another_random_tensor.T` so that we have matching dimensions.

```python
multiplication_tensor = torch.matmul(random_tensor,another_random_tensor)
```

<br> The resulting shape of the tensor is
```python
torch.Size([7, 1])
```
***7*** being the row number of the first matrix([7,7]) and ***1*** being the column number of the second matrix transposed([7,1]).

## Exercise 4
> Set the random seed to 0 and do exercises 2 & 3 over again

To set a random seed, we use the function `torch.manual_seed(seed=<value>)` , where `<value>` is the specific value that we choose for the seed.
<br><br>
In our case , this value is 0.

```python
torch.manual_seed(seed=0)
```
<br>

Now we start declaring the variables that we'll be using (don't forget about the .T attribute when multiplying!!)

```python
rand_tensor = torch.rand(size=[7,7])
another_rand_tensor = torch.rand(size=[1,7])
mul_tensor = torch.matmul(rand_tensor,another_rand_tensor.T)
```
<br>

If you run this code multiple times , you will see that the variables don't change , even though we have 2 "random" tensors.
<br><br>
That's because everytime this code gets executed the seed remains the same and so do the random variables that are dependent on the seed.(see documentation on [randomness](https://pytorch.org/docs/stable/notes/randomness.html) in pytorch)

### Exercise 5
> Speaking of random seeds, we saw how to set it with torch.manual_seed() but is there a GPU equivalent? If there is, set the GPU random seed to 1234






