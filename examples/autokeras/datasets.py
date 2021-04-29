import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def load_data():
    df=pd.read_csv("/home/vandana.rolan/Training/train_final.csv").drop(["x28","x38"],axis=1).fillna(0)
#     df = df.head(10000)
    y=df["Target"].values
    df=df.drop(["Target","id"],axis=1)
    x = df.values
    x_train, x_test, y_train, y_test = train_test_split(x_scaled,y, test_size=0.3,random_state=100)
    
    print(f"train_X shape: {np.shape(x_train)}")
    print(f"train_y shape: {np.shape(y_train)}")
    print(f"valid_X shape: {np.shape(x_test)}")
    print(f"valid_y shape: {np.shape(y_test)}")

    return x_train, x_test, y_train, y_test