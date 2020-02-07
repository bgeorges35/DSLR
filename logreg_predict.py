import pandas as pd
import sys

def decision_boundary(prob):
    return 1 if prob >= .5 else 0

if __name__ == "__main__":
    dataset = pd.read_csv(sys.argv[1])
    dataset = dataset[dataset.columns[1:]]
    for thetas in dataset.iterrows():
        print(thetas)
    