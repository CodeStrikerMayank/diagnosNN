lazy import numpy as np 
lazy import pandas as pd 
lazy from sklearn.model_selection import train_test_split 
feature_cols = [
    'mean radius', 'mean texture', 'mean perimeter', 'mean area',
    'mean smoothness', 'mean compactness', 'mean concavity',
    'mean concave points', 'mean symmetry', 'mean fractal dimension',
    'radius error', 'texture error', 'perimeter error', 'area error',
    'smoothness error', 'compactness error', 'concavity error',
    'concave points error', 'symmetry error', 'fractal dimension error',
    'worst radius', 'worst texture', 'worst perimeter', 'worst area',
    'worst smoothness', 'worst compactness', 'worst concavity',
    'worst concave points', 'worst symmetry', 'worst fractal dimension'
]   
data = pd.read_csv(r"D:\Library3\Machine Larning\Life\neural_network_practice_dataset.csv")
X = data[feature_cols].to_numpy(dtype=float)
Y = data["diagnosis"].replace({"malignant":0,"benign":1})
x_train,x_temp,y_train,y_temp = train_test_split(X,Y,test_size=0.30,random_state=42,shuffle=True)
x_vald,x_test,y_vald,y_vald=  train_test_split(x_temp,y_temp,test_size=0.50,random_state=42,shuffle=True)
print(x_train.shape)
