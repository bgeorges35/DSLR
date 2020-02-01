import pandas as pd
import matplotlib.pyplot as plt
import sys

def plot_pair(marks):
    pd.plotting.scatter_matrix(marks, figsize=(30, 30))
            
if __name__ == "__main__":
    dataset = pd.read_csv(sys.argv[1], index_col = 'Index')
    marks = dataset[dataset.columns[6:]]
    plot_pair(marks)