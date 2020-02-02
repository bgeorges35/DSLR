import pandas as pd
import seaborn as sns;
import sys

def logistic_regression(df):
    

if __name__ == "__main__":
    dataset = pd.read_csv(sys.argv[1], index_col = 'Index')
    marks = dataset[dataset.columns[6:]]
    result = pd.concat([marks, dataset[dataset.columns[0]]], axis=1)
    logistic_regression(result)
    