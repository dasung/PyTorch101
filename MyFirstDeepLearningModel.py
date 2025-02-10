"""
Summary of Training Process
1. Loop through epochs and batches of the training data.
    - number_epochs: how many times the entire dataset should passed through the model. 
    - train_loader: A PyTorch DataLoader object that loads the training data in batches.
    - enumerate(train_loader): Loops through the dataset, retrieving a batch of data at each step.

2. Reset gradients before each step (zero_grad()).
    -  Clears old gradients from the previous iteration to prevent accumulation.

3. Forward pass: Compute model predictions (y_hat).
    - data[0]: Represents the input data (features) in the batch.
    - model(data[0]): Feeds the input data into the model to get predictions (y_hat).

4. Compute loss between predictions and actual labels.
    - data[1]: Represents the actual target labels.
    - Computes the loss between predicted (y_hat) and actual (data[1]) values.
    - loss.item(): Extracts the loss value as a Python number for logging.

5. Backpropagation: Compute gradients (loss.backward()).
    - Computes gradients of the loss function with respect to model parameters using backpropagation.

6. Update weights using an optimization algorithm (optimizer.step()).
    - Updates the model parameters using the computed gradients.
"""

for epoch in range(number_epochs):  # Loop over the number of epochs
    for j, data in enumerate(train_loader):  # Iterate over the training dataset batch by batch

         # optimization
        optimizer.zero_grad()

        # forward pass
        y_hat = model(data[0])

        # compute loss
        loss = loss_fun(y_hat, data[1])
        losses.append(loss.item())

        # backprop
        loss.backward()

        # update weights
        optimizer.step()