lazy import pandas as pd
lazy import numpy as np

#extraction
class extract_csv_data :

    def __init__(self,):
        self.data = pd.read_csv(r"D:\Library3\Machine Larning\Life\neural_network_practice_dataset.csv")

    def data_extraction(self):
    
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
        
        self.X = self.data[feature_cols].to_numpy() 
        self.data["diagnosis"] =  self.data["diagnosis"].replace({"benign" :1, "malignant" : 0})
        self.Y = self.data[["diagnosis"]].to_numpy()

    def normalisation_data(self):
        U = np.mean(self.X,axis=0,keepdims=True)
        sigma = np.std(self.X,axis=0,keepdims=True)
        self.x_dd = (self.X-U)/(sigma+ 1e-8)

        
p1 = extract_csv_data()
print(p1.data_extraction())
p1.normalisation_data()