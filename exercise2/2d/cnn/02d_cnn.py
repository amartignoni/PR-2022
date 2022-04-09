# Code is inspired by the following Blogpost:
# https://www.analyticsvidhya.com/blog/2019/10/building-image-classification-models-cnn-pytorch/

"""
CNN with 3 conv layers and a fully connected classification layer
PATTERN RECOGNITION EXERCISE:
Fix the three lines below marked with PR_FILL_HERE
"""
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from matplotlib import image
import os

class Flatten(nn.Module):
    """
    Flatten a convolution block into a simple vector.

    Replaces the flattening line (view) often found into forward() methods of networks. This makes it
    easier to navigate the network with introspection
    """

    def forward(self, x):
        x = x.view(x.size()[0], -1)
        return x


class PR_CNN(nn.Module):
    """
    Simple feed forward convolutional neural network

    Attributes
    ----------
    expected_input_size : tuple(int,int)
        Expected input size (width, height)
    conv1 : torch.nn.Sequential
    conv2 : torch.nn.Sequential
    conv3 : torch.nn.Sequential
        Convolutional layers of the network
    fc : torch.nn.Linear
        Final classification fully connected layer

    """

    def __init__(self, **kwargs):
        """
        Creates an CNN_basic model from the scratch.

        Parameters
        ----------
        output_channels : int
            Number of neurons in the last layer
        input_channels : int
            Dimensionality of the input, typically 3 for RGB
        """
        super(PR_CNN, self).__init__()

        # PR_FILL_HERE: Here you have to put the expected input size in terms of width and height of your input image
        self.expected_input_size = (28, 28)

        # First layer
        self.conv1 = nn.Sequential(
            # PR_FILL_HERE: Here you have to put the input channels, output channels and the kernel size
            nn.Conv2d(in_channels=1, out_channels=24, kernel_size=5, stride=3),
            nn.LeakyReLU()
        )

        # Classification layer
        self.fc = nn.Sequential(
            Flatten(),
            # PR_FILL_HERE: Here you have to put the output size of the linear layer. DO NOT change 1536!
            nn.Linear(1536, 10)
        )

    def forward(self, x):
        """
        Computes forward pass on the network

        Parameters
        ----------
        x : Variable
            Sample to run forward pass on. (input to the model)

        Returns
        -------
        Variable
            Activations of the fully connected layer
        """
        x = self.conv1(x)
        x = self.fc(x)
        return x


class DataLoader:

    def load_data(self, directory):
        image_labels = []
        image_files = []

        for path, currentDirectory, filenames in os.walk(directory):
            for filename in filenames:
                image_labels.append(filename.split("-")[0])
                image_content = image.imread(path + "/" + filename)
                image_files.append(image_content)

        return image_labels, image_files


# Instantiate variables
batch_size = 64
learning_rate = 0.01
num_epochs = [200]

# Load data
dataLoader = DataLoader()
train_labels, train_images = dataLoader.load_data("./../../../../../Data/mnist-png-format-permutated/train")
test_labels, test_images = dataLoader.load_data("./../../../../../Data/mnist-png-format-permutated/test")

# Converting data into proper format
train_x = np.array(train_images).reshape(60000, 1, 28, 28)
train_x = train_x.astype(np.float32)
train_x = torch.from_numpy(train_x)
train_x = torch.tensor(train_x)
train_y = np.array(train_labels).astype(int)
train_y = torch.from_numpy(train_y)
train_y = torch.tensor(train_y)

test_x = np.array(test_images).reshape(10000, 1, 28, 28)
test_x = test_x.astype(np.float32)
test_x = torch.from_numpy(test_x)
test_x = torch.tensor(test_x)
test_y = np.array(test_labels).astype(int)
test_y = torch.from_numpy(test_y)
test_y = torch.tensor(test_y)

# Instantiate model
model = PR_CNN()

# Define loss function
criterion = nn.CrossEntropyLoss()

# Define optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=0.005, momentum=0.9)

# empty lists to store losses and accuracies
train_losses = []
train_accuracies = []
test_losses = []
test_accuracies = []

# Run number of epochs
for epochs in num_epochs:
    for epoch in range(epochs):
        model.train()

        # Forward pass
        #optimizer.zero_grad()
        outputs = model(train_x)
        train_loss = criterion(outputs, train_y)

        # Backward and optimize
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        train_losses.append(train_loss)

        # prediction
        with torch.no_grad():
            train_output = model(train_x.cpu())

        softmax = torch.exp(train_output).cpu()
        prob = list(softmax.numpy())
        predictions = np.argmax(prob, axis=1)
        accuracy = accuracy_score(train_y, predictions)
        train_accuracies.append(accuracy)

        with torch.no_grad():
            test_output = model(test_x.cpu())

        test_loss = criterion(test_output, test_y)
        test_losses.append(test_loss)
        softmax = torch.exp(test_output).cpu()
        prob = list(softmax.numpy())
        predictions = np.argmax(prob, axis=1)
        accuracy = accuracy_score(test_y, predictions)
        test_accuracies.append(accuracy)

        print('Epoch [{}/{}], Loss: {:.4f}, Accuracy: {}'.format(epoch + 1, epochs, train_loss.item(), accuracy))

    plt.figure(figsize=(10, 10))
    plt.plot(train_accuracies, label='Training accuracies')
    plt.plot(test_accuracies, label='Testing accuracies')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('accuracy_02d_cnn.png')
    plt.figure(figsize=(10, 10))
    plt.plot(train_losses, label='Training losses')
    plt.plot(test_losses, label='Testing losses')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('loss_02d_cnn.png')

    train_accuracies.clear()
    test_accuracies.clear()
    train_losses.clear()
    test_losses.clear()

with torch.no_grad():
    final_output = model(test_x.cpu())

final_test_loss = criterion(final_output, test_y)
test_losses.append(final_test_loss)
softmax = torch.exp(final_output).cpu()
prob = list(softmax.numpy())
predictions = np.argmax(prob, axis=1)
accuracy = accuracy_score(test_y, predictions)
print("Final Test Accuracy: {}, Final Test Loss: {}".format(accuracy, final_test_loss))
test_accuracies.append(accuracy)
