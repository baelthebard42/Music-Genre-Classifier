import json
import numpy as np
from sklearn.model_selection import train_test_split

def load_data(path):

    with open(path, "r") as fp:
        data = json.load(fp)
    
    features = np.array(data["mfcc"])
    target = np.array(data["labels"])

    return features, target


def split_dataset(X, y, t_size, v_size):

   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=t_size)

   X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=v_size)
   
   X_train = X_train[..., np.newaxis]
   X_val = X_val[..., np.newaxis]
   X_test = X_test[..., np.newaxis]
   
   return X_train, X_val, X_test, y_train, y_val, y_test


