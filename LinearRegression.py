#%% packages
import numpy as np
import pandas as pd
import torch
import torch.nn as nn 
import seaborn as sns

#%% data import
cars_file = './data/cars.csv'
cars = pd.read_csv(cars_file)
cars.head()

#%% visualise the statistical model
sns.scatterplot(x='wt', y='mpg', data=cars)
sns.regplot(x='wt', y='mpg', data=cars)

#%% convert data to tensors
X_list = cars.wt.values
X_np = np.array(X_list, dtype=np.float32).reshape(-1,1)
X_np.shape  # we have 32 observations and one columan

y_list = cars.mpg.values.tolist()
y_np = np.array(y_list, dtype=np.float32).reshape(-1,1)
y_np.shape

#%%  create tensors
X = torch.from_numpy(X_np)
y = torch.tensor(y_list)

#%% training the model
w = torch.rand(1, requires_grad=True, dtype=torch.float64) # define weight
b = torch.rand(1, requires_grad=True, dtype=torch.float64) # define bias

num_epochs = 1000 # num of iterations to be trained
learning_rate = 0.001
for epoch in range(num_epochs):     # iterate 1000 times through network
  for i in range(len(X)):           # each time passing data

    # forward pass
    y_predict = X[i] * w + b        # calcualte predication
    # calculate loss
    loss_tensor = torch.pow(y_predict - y[i], 2)
    # backward pass
    loss_tensor.backward()
    # extract losses
    loss_value = loss_tensor.data[0]
    # update weights and biases
    with torch.no_grad():
      w -= w.grad * learning_rate
      b -= b.grad * learning_rate
      w.grad.zero_()
      b.grad.zero_()
  print(loss_value)

#%% check results
print(f"Weight: {w.item()}, Bias: {b.item()}")

# %% make prediction - find new y for model X (Linear Regression)
y_pred = ((X * w) + b).detach().numpy()

# %% visualise the our Deep Learning model 
sns.scatterplot(x=X_list, y=y_list)
sns.lineplot(x=X_list, y=y_pred.reshape(-1), color='red') # this red-line comes from our DL model


# %% Prove results with (Statistical) Linear Regression model
from sklearn.linear_model import LinearRegression
reg = LinearRegression().fit(X_np, y_list)
print(f"Slope: {reg.coef_}, Bias: {reg.intercept_}")

# %%
