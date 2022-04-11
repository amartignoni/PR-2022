import numpy as np
import torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt


def load_train(batch_size):

    path = "./data/train/"
    train_ds = torchvision.datasets.ImageFolder(
        root=path, transform=torchvision.transforms.ToTensor()
    )
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, num_workers=0
    )

    return train_loader


def load_test():
    path = "./data/test/"
    test_ds = torchvision.datasets.ImageFolder(
        root=path, transform=torchvision.transforms.ToTensor()
    )
    test_loader = torch.utils.data.DataLoader(test_ds, num_workers=0)

    return test_loader


class MLP(nn.Module):  ## adapted from the pytorch documentation
    def __init__(self, hidden_neurons):

        super(MLP, self).__init__()
        self.flatten = nn.Flatten()
        self.layers = nn.Sequential(
            nn.Linear(28*28, hidden_neurons),
            nn.ReLU(),
            nn.Linear(hidden_neurons, hidden_neurons),
            nn.ReLU(),
            nn.Linear(hidden_neurons, 10),
        )

        def forward(self, x):
            x = self.flatten(x)
            y_pred = self.layers(x)
            return y_pred


## define parameters
learning_rate = 0.001  # from 0.001 to 0.1
batch_size = 64  # change if performance requires it
epochs = 100  #

## initiate model
model = MLP()

## define loss function
loss_func = nn.CrossEntropyLoss()

## define optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

## data arrays for graphs
train_loss = []
train_acc = []
test_loss = []
test_acc = []


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    tr_loss, correct = 0, 0
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        loss = loss_fn(pred, y)
        tr_loss += loss.item()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    tr_loss /= num_batches
    correct /= size
    print(
        f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {tr_loss:>8f} \n"
    )
    return correct, tr_loss


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(
        f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"
    )
    return correct, test_loss


for e in range(epochs):
    print(f"Epoch {e+1}\n-------------------------------")
    train_accuracy, train_err = train_loop(load_train(batch_size), model, loss_func, optimizer)
    test_accuracy, test_err = test_loop(load_test(), model, loss_func)
    train_acc.append(train_accuracy)
    train_loss.append(train_err)
    test_acc.append(test_accuracy)
    test_loss.append(test_err)
print("Done!")

plot_results = False

if plot_results: # taken from exercise 2c
    plt.subplot(221)
    plt.plot(train_acc, label="Training accuracies")
    plt.plot(test_acc, label="Testing accuracies")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.subplot(222)
    plt.plot(train_loss, label="Training losses")
    plt.plot(test_loss, label="Testing losses")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
