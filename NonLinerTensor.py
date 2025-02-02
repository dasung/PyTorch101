#%% packages
import torch
import seaborn as sns
import numpy as np

#%% create a tensor
x = torch.tensor(2.0, requires_grad=True)

y = (x-3) * (x-6) * (x-4) # create non-liner second tensor
print(y)

y.backward() # function to calculate gradients / get slope for x=2.0

print("x's gradient %s", x.grad) # show gradient of first tensor

#%% function for showing automatic gradient calculation
def y_function(val):
    return (val-3) * (val-6) * (val-4)

x_range = np.linspace(0, 10, 101) # define range with 101 points for validation perpose
y_range = [y_function(i) for i in x_range]

sns.lineplot(x = x_range, y = y_range) # plot sample data

# %%
