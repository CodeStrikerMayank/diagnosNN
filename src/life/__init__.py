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
data = data.dropna()
X = data[feature_cols].to_numpy(dtype=float)
Y = data["diagnosis"].map({"malignant":0,"benign":1})
x_train,x_temp,y_train,y_temp = train_test_split(X,Y,test_size=0.30,random_state=42,shuffle=True)
x_vald,x_test,y_vald,y_test=  train_test_split(x_temp,y_temp,test_size=0.50,random_state=42,shuffle=True)


"""Data scaling normalistion"""
u  = np.mean(x_train,axis=0)
s = np.std(x_train,axis=0)

x_train = (x_train-u)/(s+1e-8)
x_vald = (x_vald-u)/(s+1e-8)
x_test = (x_test - u) / (s + 1e-8)

def relu(z):return np.maximum(z*0.01,z)
def sigmoid(z): return 1/(1+(np.exp(-np.clip(z,-500,500))))
def relu_der(z):return np.where(z>0.0,1.0,0.01)
def bce(z,z_pre):
    escpilion = 1e-15
    z2 = np.clip(z_pre,escpilion,1-escpilion)
    return -np.mean(z*(np.log(z2)+((1-z)*(np.log(1-z2)))))

def porcess(x_train,y_train,x_vald,y_vald,y_test,x_test):
    """Data initilisation here by in T -90"""
    np.random.seed(46)
    w1= np.random.randn(30,8)*np.sqrt(2/4)
    b1 = np.zeros((1,8))

    w2 = np.random.randn(8,6)*np.sqrt(2/4)
    b2 = np.zeros((1,6))

    w3 = np.random.randn(6,4)*np.sqrt(2/4)
    b3 = np.zeros((1,4))

    w4 = np.random.randn(4,1)*np.sqrt(2/4)
    b4 = np.zeros((1,1))

    epochs = 10000
    lrt = 0.05
    m = len(x_train)
    for epoch in range(epochs):
        z1 = np.dot(x_train,w1)+b1
        a1 = relu(z1)

        z2 = np.dot(a1,w2)+b2
        a2 = relu(z2)

        z3 = np.dot(a2,w3)+b3 
        a3 = relu(z3) 

        z4 = np.dot(a3,w4)+b4  
        a4 = sigmoid(z4)

        # backpropogation 
        dz4 = (a4-y_train)
        dw4 = (1/m)*np.dot(a3.T,dz4) 
        db4 = (1/m)*np.sum(dz4,axis=0,keepdims=True)

        dz3 = np.dot(dz4,w4.T)*relu_der(z3)
        dw3 = (1/m)*np.dot(a2.T,dz3)
        db3 = (1/m)*np.sum(dz3,axis=0,keepdims=True)

        dz2 = np.dot(dz3,w3.T)*relu_der(z2)
        dw2 = (1/m)*np.dot(a1.T,dz2)
        db2= (1/m)*np.sum(dz2,axis=0,keepdims=True)

        dz1 = np.dot(dz2,w2.T)*relu(z1)
        dw1 = (1/m)*np.dot(x_train.T,a1)
        db1 = (1/m)*np.sum(dz1,axis=0,keepdims=True)
        b1 -= lrt*dw1 
        w1 -= lrt*dw1 
        b2 -= lrt*db2
        w2 -= lrt*dw2 
        b3 -= lrt*db3  
        w3 -= lrt*dw3 
        b4 -= lrt * db4      
        w4 -= lrt*dw4             

porcess(x_train,y_train,x_vald,y_vald,y_test,x_test)
