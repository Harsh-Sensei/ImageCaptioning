import torch
import numpy as np
# Tensors initializations

device_type = "cuda" if torch.cuda.is_available() else "cpu"

sample_tensor = torch.tensor([[1, 2, 3], [73, 66, 54]], dtype=torch.float32, device=device_type, requires_grad=True)

print(sample_tensor)
print(sample_tensor.device)

# Other important initializations
print(torch.empty(size=(3, 3)))
print(torch.zeros((3, 3)))
print(torch.ones((3, 3)))
print(torch.rand((3, 3)))
print(torch.eye(5, 5))
print(torch.arange(start=0, end=10, step=1))
print(torch.linspace(start=0, end=1, steps=10))
print(torch.empty(size=(1, 5)).normal_(0, 1))
print(torch.empty(size=(1, 5)).uniform_(0, 1))
print(torch.diag(torch.ones(size=(3, 3))))

# Tensor to numpy and vice-versa
np_arr = np.ones(5)
tensor_re = torch.from_numpy(np_arr)
np_arr_re = tensor_re.numpy()
print(tensor_re.long())
print(tensor_re.float())
print(tensor_re.double())


# Operations with tensors
x = torch.tensor([1, 73, 47])
y = torch.tensor([1, 2, 3])

print(x + y)
print(x - y)
print(torch.true_divide(x, y))
t = x + y
t += x # inplace adding
t.add_(x) # inplace addition
t = t + x # addition after copy

print(torch.dot(x, y))





