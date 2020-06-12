import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


def writeCSV(df):
    df.to_csv('theta.csv')


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


def accuracy(df, thetas):
    predict = prediction(df, thetas)
    y_test = df[df.columns[0]]
    y_test = y_test.replace({'Gryffindor': 0, 'Slytherin': 1, 'Ravenclaw': 2, 'Hufflepuff': 3})
    y_test = np.array(y_test)

    error = np.sum(y_test == predict)
    accuracy = error / len(y_test) * 100
    print("Accuracy:", accuracy, "%")


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def cost_function(y, h, m):
    return (- 1 / m) * (np.dot(y, np.log(h)) + (np.dot((1 - y), np.log(1 - h))))


def plot_cost_function(cost):
    axes = plt.axes()
    axes.grid()
    plt.plot(cost)
    plt.show()


def logistic_regression(x, Y):
    thetas = np.zeros(x.shape[1])
    learning_rate = 1.1
    m = len(x)
    cost = []
    for i in tqdm(range(6000)):
        z = np.dot(x, thetas)
        h = sigmoid(z)
        cost.append(cost_function(Y, h, m))
        thetas = thetas - learning_rate * (1 / m) * np.dot((h - Y), x)
    plot_cost_function(cost)
    return thetas


def houses(df):
    houses = []
    houses.append(df.replace({'Gryffindor': 1, 'Slytherin': 0, 'Ravenclaw': 0, 'Hufflepuff': 0}))
    houses.append(df.replace({'Gryffindor': 0, 'Slytherin': 1, 'Ravenclaw': 0, 'Hufflepuff': 0}))
    houses.append(df.replace({'Gryffindor': 0, 'Slytherin': 0, 'Ravenclaw': 1, 'Hufflepuff': 0}))
    houses.append(df.replace({'Gryffindor': 0, 'Slytherin': 0, 'Ravenclaw': 0, 'Hufflepuff': 1}))
    Y = np.array(houses)
    return Y


def preprocessing_data(df):
    x = df[df.columns[6:]]
    x = x.fillna(x.mean())
    x_once = np.ones((x.shape[0], 1))
    xnew = np.hstack((x, x_once))
    scaler = StandardScaler()
    scaler.fit(xnew)
    return scaler.transform(xnew)


def find_thetas(x, Y):
    thetas = []
    for y in Y:
        thetas.append(logistic_regression(x, y))
    return thetas


if __name__ == "__main__":
    dataset = pd.read_csv(sys.argv[1], index_col='Index')
    # train, test =  skl.train_test_split(dataset, test_size=0.25, random_state=0)

    x = preprocessing_data(dataset)

    thetas = find_thetas(x, houses(dataset[dataset.columns[0]]))

    accuracy(dataset, thetas)
    writeCSV(pd.DataFrame(thetas))
