import csv
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.append(".")
from cnn_02c import PR_CNN

with open('./../../../../Competition/mnist_test.csv', 'r') as f:
    reader = csv.reader(f)
    data = list(reader)
    matrix = np.array(data, dtype=int)
    test_samples = matrix[:, :]
    test_labels = matrix[:, 0]


test_x = test_samples.reshape(10000, 1, 28, 28)
test_x = test_x.astype(np.float32)
test_x = torch.from_numpy(test_x)
test_x = torch.tensor(test_x)

# load model
model = torch.load("complete_model.pth")
model.eval()

# prediction
with torch.no_grad():
    test_output = model(test_x.cpu())

softmax = torch.exp(test_output).cpu()
prob = list(softmax.numpy())
predictions = np.argmax(prob, axis=1)

np.savetxt("cnn.csv", predictions, fmt='%i')


print("DONE!")

