import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
import pickle as pkl
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report
from sklearn.impute import SimpleImputer, KNNImputer
import xgboost as xg


def read_data(file_path):
    data = pd.read_csv(file_path)
    return data

def split_data(X, y):
    X_train, X_val, Y_train, Y_val = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
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
        print(self.Y_train.head())
        print(self.Y_val.head())
        #target columns needs encoding
        
    def save_objects(self, feature_name, obj):
        with open(f"{self.artifacts_path}/{feature_name}_estimator.pkl", 'wb') as f:
            pkl.dump(obj, f)
        
    def label_encoding(self, ):
        cat_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'poutcome']
        self.X_train[cat_cols] = self.X_train[cat_cols].fillna("missing")
        self.X_val[cat_cols] = self.X_val[cat_cols].fillna("missing")
        for col in cat_cols:
            label_encoder = LabelEncoder()
            data = self.X_train[col].values.tolist()
            data.append("missing")
            data.append("others")
            label_encoder.fit(data)
            self.X_train[col] = label_encoder.transform(self.X_train[col])
            self.X_val[col] = self.X_val[col].map(lambda s: 'Others' if s not in label_encoder.classes_ else s)
            self.X_val[col] = label_encoder.transform(self.X_val[col])
            self.save_objects(col, label_encoder)

    def onehot_encoding(self, ):
        onehot_encoder = OneHotEncoder()
        cols = []
        
    def feature_scaling(self, ):
        cols_to_scale = ['age', 'balance', 'days_since_last_contact', 'duration', 'campaign', 'pdays', 'previous']
        scaler = MinMaxScaler()
        for col in cols_to_scale:
            self.X_train[[col]] = scaler.fit_transform(self.X_train[[col]]).reshape(-1,1)
            self.X_val[[col]] = scaler.fit_transform(self.X_val[[col]]).reshape(-1,1)
            self.save_objects(f"{col}_scaler", scaler)
            
            
    
    def impute(self,):
        imputer = SimpleImputer(strategy='mean')
        train_cols = self.X_train.columns

        self.X_train = imputer.fit_transform(self.X_train)
        self.X_val = imputer.transform(self.X_val)
        self.X_train = pd.DataFrame(self.X_train, columns=train_cols)
        self.X_val = pd.DataFrame(self.X_val, columns=train_cols)

    
    def encode_labels(self, ):
        self.Y_train['target'] = self.Y_train['target'].replace({"no": 0, "yes": 1})
        self.Y_val['target'] = self.Y_val['target'].replace({"no": 0, "yes": 1})
        
    def feature_creation(self, ):
        today = pd.Timestamp.today().normalize()

        self.X_train['last contact date'] = pd.to_datetime(self.X_train['last contact date'])
        self.X_train['days_since_last_contact'] = (today - self.X_train['last contact date']).dt.days
        
        self.X_val['last contact date'] = pd.to_datetime(self.X_val['last contact date'])
        self.X_val['days_since_last_contact'] = (today - self.X_val['last contact date']).dt.days

    def drop_columns(self,):
        self.X_train.drop(['last contact date'], axis=1, inplace=True)
        self.X_val.drop(['last contact date'], axis=1, inplace=True)
        
        
        
    def procesing(self,):
        self.label_encoding()
        # self.onehot_encoding()
        self.feature_creation()
        self.drop_columns()
        self.impute()
        self.feature_scaling()
        # self.encode_labels()
        return self.X_train, self.X_val, self.Y_train, self.Y_val
        


class ModelBuilding:
    def __init__(self, X_train, X_val, Y_train, Y_val) -> None:
        self.X_train = X_train
        self.X_val  = X_val
        self.Y_train = Y_train
        self.Y_val =  Y_val
    
    
    
    def train_model(self,):
        self.model = xg.XGBClassifier()
        self.model.fit(self.X_train, self.Y_train)
        predictions = self.model.predict(self.X_val)
        # report = classification_report()
        
        
        




if __name__=='__main__':
    train_path = 'data/train.csv'
    artifacts_path = 'model'
    data = read_data(train_path)
    # visualizations(train=data)
    y = data[['target']]
    X = data.drop(['target'], axis=1)
    # Data preprocessing
    obj = DataPreprocessing(X, y, artifacts_path=artifacts_path)
    # visualizations(data)
    X_train_processed, X_val_processed, Y_train, Y_val = obj.procesing()
    Y_train.replace({"no":0, "yes": 1},  inplace=True)
    Y_val.replace({"no":0, "yes": 1},  inplace=True)
    
    #save preprocessed files
    X_train_processed.to_csv("preprocessed/X_train_preprocessed.csv", index=False)
    X_val_processed.to_csv("preprocessed/X_val_processed.csv", index=False)
    Y_train.to_csv("preprocessed/Y_train.csv", index=False)
    Y_val.to_csv("preprocessed/Y_val.csv", index=False)


    

    