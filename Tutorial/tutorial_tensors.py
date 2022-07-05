# 1. Tensors
import torch
import numpy as np


# 1) initializing

## direct from data
data = [[1, 2],[3, 4]]
x_data = torch.tensor(data)

## from np_array
np_array = np.array(data)
x_np = torch.from_numpy(np_array)

## retain the properties
x_ones = torch.ones_like(x_data)
print(f"Ones Tensor: \n {x_ones} \n")

x_rand = torch.rand_like(x_data, dtype=torch.float)
print(f"Random Tensor: \n {x_rand} \n")

## dimension
shape = (2,3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor}")

# 2) attribute of tensor

## shape, datatype, device
tensor = torch.rand(3, 4)
print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")

# 3) operations of tensor

## gpu configuration(use .to method)
if torch.cuda.is_available():
    tensor = tensor.to("cuda")

## method like numpy
tensor2 = torch.ones(4, 4)
print(f"First row: {tensor2[0]}")
print(f"First column: {tensor2[:, 0]}")
print(f"Last column: {tensor2[..., -1]}")
tensor2[:,1] = 0
print(tensor2)

## joining (concat)
t1 = torch.cat([tensor, tensor, tensor], dim=1) ## concatenating elements
print(t1)

## arithmetic operations
y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)

y3 = torch.rand_like(tensor)
torch.matmul(tensor, tensor.T, out = y3)

z1 = tensor * tensor
z2 = tensor.mul(tensor)

z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out=z3)

##  0 dim tensor
agg = tensor2.sum()
agg_item = agg.item()
print(agg_item, type(agg_item))

## in place operations
print(f"{tensor2} \n")
tensor2.add_(5)
print(tensor2)

# 4) Bridge with Numpy

## tensor 2 numpy
t = torch.ones(5)
print(f"t: {t}")
n = t.numpy()
print(f"n: {n}")

## moreover, reflects changes
t.add_(1)
print(f"t: {t}")
print(f"n: {n}")

## numpy 2 tensor
n = np.ones(5)
t = torch.from_numpy(n)

np.add(n, 1, out=n)
print(f"t: {t}")
print(f"n: {n}")