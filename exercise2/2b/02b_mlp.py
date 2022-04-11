import numpy as np
import torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt


def load_train(batch_size):

    path = "./data/train/"
    train_ds = torchvision.datasets.ImageFolder(
        root=path, transform=torchvision.transforms.Compose(
            [torchvision.transforms.ToTensor(), torchvision.transforms.Grayscale()]
        )
    )
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size
    )

    return train_loader


def load_test():
    path = "./data/test/"
    test_ds = torchvision.datasets.ImageFolder(
        root=path, transform=torchvision.transforms.Compose(
            [torchvision.transforms.ToTensor(), torchvision.transforms.Grayscale()]
        )
    )
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size)

    return test_loader


class MLP(nn.Module):  ## adapted from the pytorch documentation
    def __init__(self, hidden_neurons):

        super(MLP, self).__init__()
        self.flatten = nn.Flatten()
        self.layers = nn.Sequential( ## three layers, hidden layer adaptable
            nn.Linear(28*28, hidden_neurons),
            nn.ReLU(),
            nn.Linear(hidden_neurons, hidden_neurons),
            nn.ReLU(),
            nn.Linear(hidden_neurons, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        y_pred = self.layers(x)
        return y_pred


## define parameters
neurons_middle_layer = 100
learning_rate = 0.01  # from 0.001 to 0.1
batch_size = 512  # change if performance requires it
epochs = 5  # to be according to graphs

## check gpu availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

## initiate model
model = MLP(neurons_middle_layer).to(device)

## define loss function
loss_func = nn.CrossEntropyLoss()

## define optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

## data arrays for graphs
train_loss = []
train_acc = []
test_loss = []
test_acc = []


def train_loop(dataloader, model, loss_fn, optimizer, batch_size):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    tr_loss, correct = 0, 0
    for batch, (x, y) in enumerate(dataloader):
        # Compute prediction and loss
        x = x.to(device)
        y = y.to(device)
        pred = model(x)
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        loss = loss_fn(pred, y)
        tr_loss += loss.item()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % batch_size == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    tr_loss /= num_batches
    correct /= size
    print(
        f"Train Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {tr_loss:>8f} \n"
    )
    return correct, tr_loss


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    te_loss, correct = 0, 0

    with torch.no_grad(): # don't backpropagate when testing
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)
            pred = model(x)
            te_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    te_loss /= num_batches
    correct /= size
    print(
        f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {te_loss:>8f} \n"
    )
    return correct, te_loss


for e in range(epochs):
    print(load_train(64))
    print(f"Epoch {e+1}\n-------------------------------")
    train_accuracy, train_err = train_loop(load_train(batch_size), model, loss_func, optimizer, batch_size)
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
