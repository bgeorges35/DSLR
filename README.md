# DataScience x Logistic Regression

School project @42

The goal of this project is to discorver statistics and logistic regression
### The projet is composed by 6 programs:
##### describe.py
##### histogram.py
##### logreg_predict.py
##### logreg_train.py
##### pair_plot.py
##### scatter_plot.py


## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install requirements.

```bash
pip3 install -r requirements.txt
```

## Example

```
python describe.py ressources/dataset_train.csv
python pair_plot.py ressources/dataset_train.csv
```

## Diagrams

```python describe.py ressources/dataset_train.csv```

```

          Arithmancy    Astronomy  ...       Charms       Flying
count    1566.000000  1568.000000  ...  1600.000000  1600.000000
mean    49634.570243    39.797131  ...  -243.374409    21.958012
std     16679.806036   520.298268  ...     8.783640    97.631602
min    -24370.000000  -966.740546  ...  -261.048920  -181.470000
25%     38513.000000  -489.490664  ...  -250.646580   -41.750000
50%     49013.500000   260.289446  ...  -244.867765    -2.515000
75%     60794.500000   525.151146  ...  -232.547120    50.670000
max    104956.000000  1016.211940  ...  -225.428140   279.070000

```

```python pair_plot.py ressources/dataset_train.csv```

![Image of pair plot](https://github.com/bgeorges35/DSLR/blob/master/pair_plot.png)

```python scatter_plot.py ressources/dataset_train.csv```

![Image of pair plot](https://github.com/bgeorges35/DSLR/blob/master/scatter_plot.png)


