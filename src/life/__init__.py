lazy import pandas as pd
lazy import numpy as np

#extraction
class extract_csv_data :
    """we will extract the information """
    def __init__(self,):
        self.data = pd.read_csv(r"D:\Library3\Machine Larning\Life\neural_network_practice_dataset.csv")

    """Normalisation of data to treat"""
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
    #standard di
    def normalisation_data(self):
        U = np.mean(self.X,axis=0,keepdims=True)
        sigma = np.std(self.X,axis=0,keepdims=True)
        self.x_dd = (self.X-U)/(sigma+ 1e-8)
class NeuralNetwork(extract_csv_data):

    def __init__(self):
        super().__init__()
        
        self.data_extraction()
        self.normalisation_data()
    """Basic calcualtion logics"""
    @staticmethod
    def relu(z): 
        return np.maximum(z*0.01,z)

    @staticmethod
    def relu_derivative(z):
        return np.where(z > 0,1.0,0.01)

    @staticmethod
    def sigmoid(z):
        return 1/(1+(np.exp(-np.clip(z,-500,500))))

    def nerual_structure(self):
        np.random.seed(43)
        self.w1 = np.random.randn(30,20) * np.sqrt(2/2)
        self.b1 = np.zeros((1,20))

        self.w2 = np.random.randn(20,15) * np.sqrt(2/4)
        self.b2 = np.zeros((1,15))

        self.w3 =np.random.randn(15,9) * np.sqrt(2/4)
        self.b3 = np.zeros((1,9))

        self.w4 = np.random.randn(9,4) * np.sqrt(2/4)
        self.b4 = np.zeros((1,4))

        self.w5 = np.random.randn(4,1) * np.sqrt(2/4)
        self.b5 = np.zeros((1,1))

    def forward(self):
        self.nerual_structure()
        z1  = np.dot(self.x_dd,self.w1)+self.b1 
        a1  = self.relu(z1)
        print(a1)


k1 = NeuralNetwork()
k1.forward()

