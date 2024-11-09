import json
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def load_data(path):

    with open(path, "r") as fp:
        data = json.load(fp)
    
    features = np.array(data["mfcc"])
    target = np.array(data["labels"])

    return features, target


def split_dataset(X, y, t_size, v_size, for_CNN = False):

   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=t_size)

   X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=v_size)
   
   if for_CNN:
    X_train = X_train[..., np.newaxis]
    X_val = X_val[..., np.newaxis]
    X_test = X_test[..., np.newaxis]
   
   return X_train, X_val, X_test, y_train, y_val, y_test


def plot_history(history):

    fig, ax = plt.subplots(2)

    # plotting accuracy

    ax[0].plot(history.history["accuracy"], label="train_accuracy")
    ax[0].plot(history.history["val_accuracy"], label="test_accuracy")
    ax[0].set_ylabel("Accuracy")
    ax[0].legend(fontsize = 5, loc="lower right")
    ax[0].set_title("Accuracy")

     # plotting error

    ax[1].plot(history.history["loss"], label="train_error")
    ax[1].plot(history.history["val_loss"], label="test_error")
    ax[1].set_ylabel("Error")
    ax[1].set_xlabel("Epoch")
    ax[1].legend(fontsize = 5, loc= "upper right")
    ax[1].set_title("Error")

    plt.show()


