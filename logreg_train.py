import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
import sklearn.model_selection as skl
from sklearn.preprocessing import StandardScaler

def writeCSV(df):
    df.to_csv('theta.csv')

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def accuracy(df, thetas):
    X, Y = preprocessing_data(df)
    prob = 0
    result = 0
    y = []
    for x in X:
        tmp_prob = 0
        for i, theta in enumerate(thetas):
            z = np.dot(x, theta)
            prob = sigmoid(z)
            if (prob > tmp_prob):
                tmp_prob = prob
                result = i
        y.append(result)
    return(y)
        
def cost_function(y, h, m):
    a = (- 1 / m) * (np.dot(y, np.log(h)) + (np.dot((1 - y), np.log(1 - h))))
    return a

def plot_cost_function(cost):
    axes = plt.axes()
    axes.grid()
    x = np.arange(len(cost))
    plt.plot(cost)
    plt.show()

def logistic_regression(X, Y):
    thetas = np.zeros(X.shape[1])
    learning_rate = 2
    m = len(X)
    cost = []
    for i in range(10000):
        z = np.dot(X, thetas)
        h = sigmoid(z)
        cost.append(cost_function(Y, h, m))
        thetas = thetas - learning_rate * (1 / m ) * np.dot((h - Y), X)
    plot_cost_function(cost)
    return thetas

def houses(df):
    Houses = []
    Houses.append(df.replace({ 'Gryffindor': 1, 'Slytherin': 0, 'Ravenclaw': 0, 'Hufflepuff': 0 }))
    Houses.append(df.replace({ 'Gryffindor': 0, 'Slytherin': 1, 'Ravenclaw': 0, 'Hufflepuff': 0 }))
    Houses.append(df.replace({ 'Gryffindor': 0, 'Slytherin': 0, 'Ravenclaw': 1, 'Hufflepuff': 0 }))
    Houses.append(df.replace({ 'Gryffindor': 0, 'Slytherin': 0, 'Ravenclaw': 0, 'Hufflepuff': 1 }))
    Y = np.array(Houses)
    return Y


def preprocessing_data(df):
    X = df[df.columns[6:]]
    X = X.fillna(X.mean())
    X = np.array(X)
    X_once = np.ones((X.shape[0], 1))
    Xnew = np.hstack((X,X_once))
    Y = houses(df[df.columns[0]])
    
    scaler = StandardScaler()
    scaler.fit(Xnew)
    Xnew = scaler.transform(Xnew)
    
    return Xnew, Y


if __name__ == "__main__":
    dataset = pd.read_csv(sys.argv[1], index_col = 'Index')
    
    train, test =  skl.train_test_split(dataset, test_size=0.2, random_state=0)
    
    X, Y = preprocessing_data(train)
    
    thetas = []
    for y in Y:
        thetas.append(logistic_regression(X, y))
    
    predict = accuracy(test, thetas)
    
    Y_test = test[test.columns[0]]
    Y_test = Y_test.replace({'Gryffindor': 0, 'Slytherin': 1, 'Ravenclaw': 2, 'Hufflepuff': 3})
    Y_test = np.array(Y_test)
    
    #Code degueu fait par Javigner !
    error = 0
    for i in np.arange(len(Y_test)):
        if (Y_test[i] != predict[i]):
            error += 1
    Accuracy = 100 - (error / len(Y_test) * 100)
    print(Accuracy)
    writeCSV(pd.DataFrame(thetas))