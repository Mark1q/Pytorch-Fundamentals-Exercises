import torch

# Exercise 2

random_tensor = torch.rand(size=[7,7])

# Exercise 3

another_random_tensor = torch.rand(size=[1,7])
multiplication_tensor = torch.matmul(random_tensor,another_random_tensor.T)

# Exercise 4

torch.manual_seed(seed=0)
rand_tensor = torch.rand(size=[7,7])
another_rand_tensor = torch.rand(size=[1,7])
mul_tensor = torch.matmul(rand_tensor,another_rand_tensor.T)

# Exercise 5

torch.cuda.manual_seed(1234)

# Exercise 6

torch.manual_seed(1234)
random_gpu_tensor = torch.rand(size=[2,3],device='cuda')
another_random_gpu_tensor = torch.rand_like(input=random_gpu_tensor)

# Exercise 7

mul_gpu_tensor = torch.matmul(random_gpu_tensor,another_random_gpu_tensor.T)

# Exercise 8

minimum_tensor = mul_gpu_tensor.min()
maximum_tensor = mul_gpu_tensor.max()

# Exercise 9

minimum_tensor_index = mul_gpu_tensor.argmin()
maximum_tensor_index = mul_gpu_tensor.argmax()

# Exercise 10

torch.manual_seed(7)

tensor = torch.rand(size=[1,1,1,10])
new_tensor = torch.reshape(input=tensor,shape=[2,10])

print(tensor,tensor.shape)
print(new_tensor,new_tensor.shape)
