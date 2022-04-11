import math
import torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt


## creation of a dataloader using the MNIST dataset in png-format
## inspired from : https://stackoverflow.com/a/51698037
def load_train(batch_size):

    path = "./data/train/"
    train_ds = torchvision.datasets.ImageFolder(
        # convert RGB png images to grayscale tensors
        root=path,
        transform=torchvision.transforms.Compose(
            [torchvision.transforms.ToTensor(), torchvision.transforms.Grayscale()]
        ),
    )
    # create dataloader with batch size
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size)

    return train_loader


def load_test():
    path = "./data/test/"
    test_ds = torchvision.datasets.ImageFolder(
        # convert RGB png images to grayscale tensors
        root=path,
        transform=torchvision.transforms.Compose(
            [torchvision.transforms.ToTensor(), torchvision.transforms.Grayscale()]
        ),
    )
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size)

    return test_loader


class MLP(nn.Module):  ## adapted from an example in the pytorch documentation
    def __init__(self, hidden_neurons):

        super(MLP, self).__init__()
        self.flatten = nn.Flatten()
        self.layers = nn.Sequential(  ## three layers, hidden layer adaptable
            nn.Linear(784, hidden_neurons),  # input layer
            nn.ReLU(),  # activation function
            nn.Linear(hidden_neurons, hidden_neurons),  # hidden layer of adaptable size
            nn.ReLU(),
            nn.Linear(hidden_neurons, 10),  # output layer
        )

    def forward(self, x):  # implicitly called by the pytorch api during training
        x = self.flatten(x)  # convert 28x28x1 images to 784x1 images
        y_pred = self.layers(x)  # run images through the network
        return y_pred


## define parameters
neurons_middle_layer = 60  # from 10 to 100, dialed to 60
learning_rate = 0.1  # from 0.001 to 0.1, dialed to 0.1
batch_size = 256  # change if performance requires it, we settled on 256 for training on Google Colab
epochs = 100  # determined using performance data

## check gpu availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

## initiate model
model = MLP(neurons_middle_layer).to(device)
model2 = MLP(neurons_middle_layer).to(device)
model.load_state_dict(torch.load("trained_model.pth"))

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
    for batch, (x, y) in enumerate(dataloader):
        # Compute prediction and loss
        x = x.to(device)
        y = y.to(device)
        pred = model(x)  # compute prediction
        correct += (  # add to average prediction counter
            (pred.argmax(1) == y).type(torch.float).sum().item()
        )
        loss = loss_fn(pred, y)  # compute loss
        tr_loss += loss.item()  # add to average loss counter

        # Backpropagation
        optimizer.zero_grad()  # reset optimizer to prevent counting twice an error
        loss.backward()  # adjust weights
        optimizer.step()

        if batch % 30 == 0:  # print statistics
            loss, current = loss.item(), batch * len(x)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    # compute averages
    tr_loss /= num_batches
    correct /= size
    print(
        f"Train : \n Accuracy: {(100 * correct):>0.1f}%, Average loss: {tr_loss:>8f} \n"
    )
    return correct, tr_loss


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    te_loss, correct = 0, 0

    with torch.no_grad():  # don't backpropagate when testing
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)
            pred = model(x)
            te_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    te_loss /= num_batches
    correct /= size
    print(f"Test : \n Accuracy: {(100*correct):>0.1f}%, Averag loss: {te_loss:>8f} \n")
    return correct, te_loss


def test_loop2(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    te_loss, correct = 0, 0

    with torch.no_grad():  # don't backpropagate when testing
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)
            pred = model(x)
            te_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    te_loss /= num_batches
    correct /= size
    print(f"Test : \n Accuracy: {(100*correct):>0.1f}%, Averag loss: {te_loss:>8f} \n")
    return correct, te_loss


for e in range(epochs):
    print(f"Epoch {e+1}\n-------------------------------")
    # execute network, store accuracies and errors
    train_accuracy, train_err = train_loop(
        load_train(batch_size), model, loss_func, optimizer
    )
    test_accuracy, test_err = test_loop(load_test(), model, loss_func)
    train_acc.append(train_accuracy)
    train_loss.append(train_err)
    test_acc.append(test_accuracy)
    test_loss.append(test_err)

test_accuracy, test_err = test_loop2(load_test(), model, loss_func)
print("Done!")

plot_results = True

if plot_results:  # taken from exercise 2c
    # plot accuracies relative to epochs
    plt.subplot(2, 2, 1)  # two rows and two columns, first position in the plot
    plt.plot(train_acc, label="Training accuracies")
    plt.plot(test_acc, label="Testing accuracies")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    # plot losses relative to epochs
    plt.subplot(2, 2, 2)  # two rows and two columns, second position in the plot
    plt.plot(train_loss, label="Training losses")
    plt.plot(test_loss, label="Testing losses")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
