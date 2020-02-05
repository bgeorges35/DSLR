import pandas as pd
import numpy as np
import seaborn as sns
import sys
import math

def decision_boundary(prob):
  return 1 if prob >= .5 else 0

# def cost_function(x, y, thetas )


def hypothesis_function(X, thetas):
    # print(X.shape)
    # print(thetas.shape)
    return (1 / (1 + np.exp(np.dot(X, thetas))))
    
def cost_function(X):
    len_ = len(X)
    return ()

def logistic_regression(X, Y):
    thetas = np.zeros(X.shape[1])
    learning_rate = 0.000015
    m = len(X)
    for i in range(6000):
        hypothesis_function(X, thetas)
        thetas = thetas - learning_rate * (1 / m ) * (hypothesis_function(X, thetas) - y) @ X
    return thetas

def houses(df):
    Houses = []
    Houses.append(df.replace({ 'Gryffindor': 1, 'Slytherin': 0, 'Ravenclaw': 0, 'Hufflepuff': 0 }))
    Houses.append(df.replace({ 'Gryffindor': 0, 'Slytherin': 1, 'Ravenclaw': 0, 'Hufflepuff': 0 }))
    Houses.append(df.replace({ 'Gryffindor': 0, 'Slytherin': 0, 'Ravenclaw': 1, 'Hufflepuff': 0 }))
    Houses.append(df.replace({ 'Gryffindor': 0, 'Slytherin': 0, 'Ravenclaw': 0, 'Hufflepuff': 1 }))
    Y = np.array(Houses)
    return(Y)
        
if __name__ == "__main__":
    dataset = pd.read_csv(sys.argv[1], index_col = 'Index')
    
    X = dataset[dataset.columns[6:]]
    X = X.fillna(X.mean())
    X = np.array(X)
    X0 = np.ones((X.shape[0], 1))
    Xnew = np.hstack((X,X0))
    Y = houses(dataset[dataset.columns[0]])
    
    thetas = []
    for y in Y:
        thetas.append(logistic_regression(Xnew, y))
    print(thetas)