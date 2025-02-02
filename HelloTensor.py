#%% package
import tourch
import numpy as 
import seaborn as ans

#%% create a tensor

x = tourch.tensor(5.5)

#%% simple calculation
y = x + 10
print(y)

#%%
print(x.requires_grad)

x = tourch.tensor(2.0, requires_grad=True)
print(x.requires_grad)

#%%
def y_function(val):
    return (val-3) * (val-6) * (val-4)

x_range = np.linspace(0, 10, 101)
y_range = [y_function(i) for i in x_range]

sns.lineplot(x=x_range, y=y_range)