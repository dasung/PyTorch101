#%% packages
import torch
import seaborn as sns
import numpy as np

# %% x -> y -> z
x = torch.tensor(1.0, requires_grad=True)
y = x**3        # depends on x 
z = 5*y - 4     # depends on y

# %%
z.backward()    # to calculate all gradients
print(x.grad)   # should be equal 5*3x**2 when x=1.0

# %% more complex network with two inputs

x11 = torch.tensor(2.0, requires_grad=True)  # input 1st node
x21 = torch.tensor(3.0, requires_grad=True)  # input 2nd node

x12 = 5 * x11 - 3 * x21         # hidden layer 1st node
x22 = 2 * x11**2 + 2 * x21      # hidden layer 2nd node

y = 4 * x12 + 3 * x22   # output layer

y.backward()    # use backward pass to get back gradients of variables

print(x11.grad)
print(x21.grad)
# %%
