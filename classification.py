import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
import pickle as pkl
import os


def read_data(file_path):
    data = pd.read_csv(file_path)
    return data

def split_data(X, y):
    X_train, X_val, Y_train, Y_val = train_test_split(X, y, test_size=0.3, stratify=y)
    return X_train, X_val, Y_train, Y_val


def visualizations(train):
    directory_name = "plots"
    if not os.path.exists(directory_name):
        os.mkdir(directory_name)
    os.chdir(directory_name)
    #plot
    
    os.chdir("..")
    
    
class DataPreprocessing:
    def __init__(self, X, y, artifacts_path):
        self.X_train, self.X_val, self.Y_train, self.Y_val = split_data(X, y)
        self.artifacts_path = artifacts_path
        print("Train and Val shape: \n")
        print("Train shape: ", self.X_train.shape, self.Y_train.shape)
        print(" \nVal shape: ", self.X_val.shape, self.Y_val.shape)
        #target columns needs encoding
        
    def save_objects(self, feature_name, obj):
        with open(f"{self.artifacts_path}/{feature_name}_estimator", 'wb') as f:
            pkl.dump(obj, f)
        
    def label_encoding(self, X_train, X_val):
        label_encoder = LabelEncoder()
        cols = []
        cat_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'poutcome', 'target']

    def onehot_encoding(self, X_train, X_val):
        onehot_encoder = OneHotEncoder()
        cols = []
    
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
    visualizations(data)

    

    