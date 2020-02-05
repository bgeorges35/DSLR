import pandas as pd
import numpy as np
import seaborn as sns
import sys

def decision_boundary(prob):
  return 1 if prob >= .5 else 0

# def cost_function(x, y )

def logistic_regression(X, Y):
    theta = 0
    for x in X:
        
    

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
    Y = houses(dataset[dataset.columns[0]])
    
    thetas = []
    for y in Y:
        thetas.append(logistic_regression(X, y))    