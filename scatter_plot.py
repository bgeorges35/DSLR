import pandas as pd
import matplotlib.pyplot as plt
import sys

def get_marks_by_subject(marks, subject_school):
    df = marks[subject_school]
    df.dropna(inplace=True)
    return df

def plot_scat(marks):
    plt.figure()
    plt.scatter(marks['Astronomy'], marks['Defense Against the Dark Arts'], label = 'students')
    plt.xlabel('Astronomy')
    plt.ylabel('Defense Against the Dark Arts')
    plt.show()
            
if __name__ == "__main__":
    dataset = pd.read_csv(sys.argv[1], index_col = 'Index')
    marks = dataset[dataset.columns[6:]]
    plot_scat(marks)