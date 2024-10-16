import pandas as pd
import numpy as np 
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
import pickle as pkl


def read_data(file_path):
    data = pd.read_csv(file_path)
    return data

def split_data(X, y):
    X_train, X_val, Y_train, Y_val = train_test_split(X, y, test_size=0.3, stratify=y)
    return X_train, X_val, Y_train, Y_val


def visualizations(train):
    pass

class DataPreprocessing:
    def __init__(self, X, y):
        self.X_train, self.X_val, self.Y_train, self.Y_val = split_data(X, y)
        print("Train and Val shape: \n")
        print("Train shape: ", self.X_train.shape, self.Y_train.shape)
        print(" \nVal shape: ", self.X_val.shape, self.Y_val.shape)
        #target columns needs encoding
        
    def label_encoding(self, X_train, Y_train):
        pass
    
    def procesing(self, X_train, X_val):
        pass
        

if __name__=='__main__':
    train_path = 'data/train.csv'
    data = read_data(train_path)
    visualizations(train=data)
    y = data[['target']]
    X = data.drop(['target'], axis=1)
    # Data preprocessing
    obj = DataPreprocessing(X, y)

    

    