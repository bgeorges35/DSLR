import pandas as pd
import numpy as np
import sys
from sklearn.preprocessing import StandardScaler


def writeCSV(df):
    df.to_csv('ressources/houses.csv')


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def prediction(df, thetas):
    x = preprocessing_data(df)
    y = []
    for x in x:
        tmp_prob = 0
        for i, theta in enumerate(thetas):
            z = np.dot(x, theta)
            prob = sigmoid(z)
            if prob > tmp_prob:
                tmp_prob = prob
                result = i
        y.append(result)
    return y


def preprocessing_data(df):
    x = df[df.columns[6:]]
    x = x.fillna(x.mean())
    x = np.array(x)
    x_once = np.ones((x.shape[0], 1))
    xnew = np.hstack((x, x_once))

    scaler = StandardScaler()
    scaler.fit(xnew)
    xnew = scaler.transform(xnew)
    return xnew


def read_csv(args):
    dataset = pd.read_csv(args, index_col='Index')
    thetas = pd.read_csv('./theta.csv', index_col=0)
    return dataset, thetas


if __name__ == "__main__":
    dataset, thetas = read_csv(sys.argv[1])
    predict = prediction(dataset, np.array(thetas))
    predict = pd.DataFrame(data=predict, columns=["Hogwarts House"])
    res = predict.replace({0: 'Gryffindor', 1: 'Slytherin', 2: 'Ravenclaw', 3: 'Hufflepuff'})
    res.index.name = "Index"
    writeCSV(res)
    print(" prediction is wrote in file ressources/houses.csv")
