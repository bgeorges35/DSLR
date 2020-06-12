import pandas as pd
import matplotlib.pyplot as plt
import sys


def get_marks_by_house(dataset, marks, house, subject_school):
    df = marks[dataset['Hogwarts House'] == house][subject_school]
    df.dropna(inplace=True)
    return df


def plot_hist(dataset, marks):
    for subject_school in marks.columns.values:
        plt.figure()
        plt.hist(get_marks_by_house(dataset, marks, 'Gryffindor', subject_school), bins=50, label='Gryffindor',
                 color='r')
        plt.hist(get_marks_by_house(dataset, marks, 'Ravenclaw', subject_school), bins=50, label='Ravenclaw', color='b')
        plt.hist(get_marks_by_house(dataset, marks, 'Slytherin', subject_school), bins=50, label='Slytherin', color='g')
        plt.hist(get_marks_by_house(dataset, marks, 'Hufflepuff', subject_school), bins=50, label='Hufflepuff',
                 color='y')
        plt.legend(loc='upper right')
        plt.title(subject_school)
        plt.show()


if __name__ == "__main__":
    dataset = pd.read_csv(sys.argv[1], index_col='Index')
    marks = dataset[dataset.columns[6:]]
    plot_hist(dataset, marks)
