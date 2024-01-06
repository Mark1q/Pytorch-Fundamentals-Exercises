import torch

random_tensor = torch.rand(size=[7,7])
another_random_tensor = torch.rand(size=[1,7])
multiplication_tensor = torch.matmul(random_tensor,another_random_tensor.T)

print(multiplication_tensor.shape)
# scalar = torch.tensor(7)
# vector = torch.tensor([1,2,3])
# MATRIX = torch.tensor([[1,2,3],[4,5,6]])
# TENSOR = torch.tensor([[[1,2,3],[4,5,6]],[[7,8,9],[10,11,12]]])

# random_tensor = torch.rand(size=[3,2])
# another_random_tensor = torch.rand(size=[2,3])
# random_tensor_like = torch.rand_like(input=random_tensor)
# random_complex_tensor = torch.randn(size=[3,2],dtype=torch.cfloat)

# zero_tensor = torch.zeros(size=[1,2,3])
# one_tensor = torch.ones(size=[3,4])
# one_tensor_like = torch.ones_like(input=one_tensor)

# zero_to_hundred_tensor = torch.arange(0,100,1)
# one_to_ten_tensor = torch.arange(0,11,1)

# addition_tensor = torch.add(random_tensor,random_tensor_like)
# subtraction_tensor = torch.sub(random_tensor,random_tensor_like)
# multiplication_tensor = torch.mul(random_tensor,random_tensor_like)
# matrix_multiplication_tensor = torch.matmul(random_tensor,another_random_tensor)

# linear = torch.nn.Linear(in_features=2,out_features=6)
# linear_tensor = linear(another_random_tensor.T)

# maximum_tensor = torch.max(input=random_tensor)
# minimum_tensor = torch.min(input=random_tensor)
# mean_tensor = torch.min(input=random_tensor) # only works for float32 
# sum_elements_tensor = torch.sum(input=random_tensor)
# index_max_tensor = torch.argmax(input=random_tensor)
# index_min_tensor = torch.argmin(input=random_tensor)

# reshaped_tensor = random_tensor.reshape(shape=[1,2,3])
# stacked_tensor = torch.stack([random_tensor,random_tensor],dim=0)
# squeezed_tensor = random_tensor.squeeze()
# unsqueezed_tensor = random_tensor.unsqueeze(dim=0)
# permuted_tensor = random_complex_tensor.permute(dims=[1,0])

# x = one_to_ten_tensor
# x = torch.stack([x,x,x],dim=1).reshape([1,11,3])

# RANDOM_SEED = torch.seed()

# torch.manual_seed(seed=RANDOM_SEED)
# random_tensor_A = torch.rand(size=[3,4])
# random_tensor_B = torch.rand(size=[3,4])

# device = "cuda" if torch.cuda.is_available() else "cpu"

# tensor_on_gpu = torch.rand(device=device,size=[3,2,3])

# print(tensor_on_gpu)