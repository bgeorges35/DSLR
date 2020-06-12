import pandas as pd
import numpy as np
import sys
from sklearn.preprocessing import StandardScaler

def writeCSV(df):
    df.to_csv('ressources/houses.csv')

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

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

def preprocessing_data(df):
    X = df[df.columns[6:]]
    X = X.fillna(X.mean())
    X = np.array(X)
    X_once = np.ones((X.shape[0], 1))
    Xnew = np.hstack((X,X_once))
    
    scaler = StandardScaler()
    scaler.fit(Xnew)
    Xnew = scaler.transform(Xnew)
    return Xnew

def read_csv(args):
    dataset = pd.read_csv(args, index_col = 'Index')
    thetas = pd.read_csv('./theta.csv', index_col = 0)
    return dataset, thetas

if __name__ == "__main__":
    dataset, thetas = read_csv(sys.argv[1])
    predict = prediction(dataset, np.array(thetas))
    print(predict)
    predict = pd.DataFrame(data=predict, columns=["Hogwarts House"])
    res = predict.replace({0: 'Gryffindor', 1: 'Slytherin', 2: 'Ravenclaw', 3: 'Hufflepuff'})
    res.index.name = "Index"
    writeCSV(res)