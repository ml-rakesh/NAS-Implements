import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from torch.utils.data import TensorDataset, DataLoader


def load_data(batch_size, train_mode):
    df=pd.read_csv("/home/vandana.rolan/Training/train_final.csv").drop(["x28","x38"],axis=1).fillna(0)
#     df = df.head(10000)
    y=df["Target"].values
    df=df.drop(["Target","id"],axis=1)
    x = df.values

    # Scale and Split the data
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    x_train, x_test, y_train, y_test = train_test_split(x_scaled,y, test_size=0.3,random_state=100)
    # Convert dataset into tensor 
    tensor_x1 = torch.Tensor(x_train)
    tensor_y1 = torch.Tensor(y_train)
    tensor_y1 = tensor_y1.reshape(-1).long()
    train_dataset = TensorDataset(tensor_x1,tensor_y1)

    tensor_x2 = torch.Tensor(x_test)
    tensor_y2 = torch.Tensor(y_test)
    tensor_y2 =tensor_y2.reshape(-1).long()
    test_dataset = TensorDataset(tensor_x2,tensor_y2)
    
    print(f"train_X shape: {np.shape(x_train)}")
    print(f"train_y shape: {np.shape(y_train)}")
    print(f"valid_X shape: {np.shape(x_test)}")
    print(f"valid_y shape: {np.shape(y_test)}")
    
    if (train_mode == 'search') | (train_mode == 'retrain'):
        return train_dataset,test_dataset
    elif train_mode == 'predict':
        return (tensor_x1,tensor_y1), (tensor_x2,tensor_y2)
    else:
        raise ValueError('wrong mode -> select one of them : [search, retrain, predict]')