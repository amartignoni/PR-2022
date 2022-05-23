import numpy as np
import pandas as pd
import time
from sklearn import svm
from sklearn.model_selection import KFold
from scipy.stats import expon
from tqdm import tqdm


# Loading the training and test sets
train_loaded_csv = pd.read_csv("mnist_train.csv", header=None)
validation_loaded_csv = pd.read_csv("mnist_validation.csv", header=None)
test_loaded_csv = pd.read_csv("mnist_test.csv", header=None)
train_data = train_loaded_csv.to_numpy()
validation_data = validation_loaded_csv.to_numpy()
test_data = test_loaded_csv.to_numpy()
# Get a sample to optimize hyperparameters (can change the size)
# Use a seed for reproducibility
np.random.seed(16)
rand_idx = np.random.choice(train_data.shape[0], size=2000, replace=False)
opt_data = train_data[rand_idx]

# Of course data_test can be the test data or the validation data
def train_eval_model(model, data_train, data_test):
    model.fit(data_train[:,1:], data_train[:,0])
    predictions = model.predict(data_test[:,1:])
    accuracy = (predictions == data_test[:,0]).sum() / len(predictions)
    return accuracy

kernels = ['linear', 'rbf']
c_values = np.logspace(-5, 5, 11)
gamma_values = np.logspace(-10, 1, 12)
scores = pd.DataFrame(columns=['kernel', 'c', 'gamma', 'accuracy', 'runtime'])


# Part 1: Hyperparameters optimization
print(f"Starting Hyperparameters optimization")
for kernel in kernels:
    if kernel == 'linear':
        for c in tqdm(c_values):
            # Use a timer
            start_time = time.perf_counter()
            accuracies = np.array([])

            # Use KFold for cross-validation
            kf = KFold(n_splits=5)
            for idx, (train_split, validation_split) in enumerate(kf.split(opt_data)):
                    svm_model = svm.SVC(C=c, kernel=kernel)
                    train_opt = opt_data[train_split]
                    validation = opt_data[validation_split]
                    accuracy = train_eval_model(svm_model, train_opt, validation)
                    accuracies = np.append(accuracies, accuracy)

            mean_acc = np.mean(accuracies)
            stop_time = time.perf_counter()
            run_time = stop_time - start_time
            new_row = pd.DataFrame({
                'kernel': kernel,
                'c': c,
                'accuracy': np.around(mean_acc, decimals=4),
                'runtime': np.around(run_time, decimals=3),
            }, index=[0])
            scores = pd.concat([scores, new_row], ignore_index=True)

    else:
        for c in tqdm(c_values):
            for gamma in gamma_values:
                # Use a timer
                start_time = time.perf_counter()
                accuracies = np.array([])

                # Use KFold for cross-validation
                kf = KFold(n_splits=5)
                for idx, (train_split, validation_split) in enumerate(kf.split(opt_data)):
                        svm_model = svm.SVC(C=c, gamma=gamma, kernel=kernel)
                        train_opt = opt_data[train_split]
                        validation = opt_data[validation_split]
                        accuracy = train_eval_model(svm_model, train_opt, validation)
                        accuracies = np.append(accuracies, accuracy)

                mean_acc = np.mean(accuracies)
                stop_time = time.perf_counter()
                run_time = stop_time - start_time
                new_row = pd.DataFrame({
                    'kernel': kernel,
                    'c': c,
                    'gamma': gamma,
                    'accuracy': np.around(mean_acc, decimals=4),
                    'runtime': np.around(run_time, decimals=3),
                }, index=[0])
                scores = pd.concat([scores, new_row], ignore_index=True)

# Save sorted scores into a csv
sorted = scores.sort_values(by=['accuracy'], ascending=False)
# Best model is the first in the sorted list
best_kernel, best_c, best_gamma, _, _ = sorted.iloc[0]
sorted.to_csv("scores.csv")


# Part 2: Main training with best model
if best_kernel == 'linear':
    print(f"Training a {best_kernel} kernel with C={best_c}")
    best_model = svm.SVC(kernel=best_kernel, C=best_c)
    accuracy = train_eval_model(best_model, train_data, validation_data)
    competition_predictions = best_model.predict(test_data)
    np.savetxt("svm.txt", competition_predictions, fmt='%d')
    print(f"SVM on MNIST terminated with parameters: kernel {kernel}, C={best_c} with accuracy {accuracy}")
else:
    print(f"Training a {best_kernel} kernel with C={best_c}, gamma={best_gamma}")
    best_model = svm.SVC(kernel=best_kernel, C=best_c, gamma=best_gamma)
    accuracy = train_eval_model(best_model, train_data, validation_data)
    competition_predictions = best_model.predict(test_data)
    np.savetxt("svm.txt", competition_predictions, fmt='%d')
    print(f"SVM on MNIST terminated with parameters: kernel {kernel}, C={best_c}, gamma={best_gamma} with accuracy {accuracy}")
