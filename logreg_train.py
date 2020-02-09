import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

def writeCSV(df):
    df.to_csv('theta.csv')

def prediction(df, thetas):
    X = preprocessing_data(df)
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
    return y

def Accuracy(df, thetas):
    predict = prediction(df, thetas)
    Y_test = df[df.columns[0]]
    Y_test = Y_test.replace({'Gryffindor': 0, 'Slytherin': 1, 'Ravenclaw': 2, 'Hufflepuff': 3})
    Y_test = np.array(Y_test)
    
    error = np.sum(Y_test == predict)
    Accuracy = error / len(Y_test) * 100
    print("Accuracy:", Accuracy, "%")

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def cost_function(y, h, m):
    return (- 1 / m) * (np.dot(y, np.log(h)) + (np.dot((1 - y), np.log(1 - h))))

def plot_cost_function(cost):
    axes = plt.axes()
    axes.grid()
    plt.plot(cost)
    plt.show()

def logistic_regression(X, Y):
    thetas = np.zeros(X.shape[1])
    learning_rate = 1.1
    m = len(X)
    cost = []
    for i in tqdm(range(6000)):
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
    X_once = np.ones((X.shape[0], 1))
    Xnew = np.hstack((X,X_once))
    scaler = StandardScaler()
    scaler.fit(Xnew)
    return scaler.transform(Xnew)

def find_thetas(X, Y):
    thetas = []
    for y in Y:
        thetas.append(logistic_regression(X, y))
    return thetas

if __name__ == "__main__":
    dataset = pd.read_csv(sys.argv[1], index_col = 'Index')
    #train, test =  skl.train_test_split(dataset, test_size=0.25, random_state=0)
    
    X = preprocessing_data(dataset)
    
    thetas = find_thetas(X, houses(dataset[dataset.columns[0]]))
    
    Accuracy(dataset, thetas)
    writeCSV(pd.DataFrame(thetas))