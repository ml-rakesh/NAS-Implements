import numpy as np
import pandas as pd
from tensorflow import keras
import sklearn
import tensorflow as tf

def load_data():
    df=pd.read_csv("/home/vandana.rolan/Training/train_final.csv").drop(["x28","x38"],axis=1).fillna(0)
    y=df["Target"].values
    df=df.drop(["Target","id"],axis=1)
    x = df.values
    # Scale and Split the data
#     scaler = sklearn.preprocessing.StandardScaler()
#     x_scaled = scaler.fit_transform(x)
    y_cat = tf.keras.utils.to_categorical(y, num_classes=2)
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(x,y_cat, test_size=0.3,random_state=100)

    print(f"train_X shape: {np.shape(X_train)}")
    print(f"train_y shape: {np.shape(y_train)}")
    print(f"valid_X shape: {np.shape(X_test)}")
    print(f"valid_y shape: {np.shape(y_test)}")
    
    return (X_train, y_train), (X_test, y_test)

if __name__ == '__main__':
    load_data()

