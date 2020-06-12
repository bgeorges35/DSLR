# -*- coding: utf-8 -*-
import pandas as pd
import sys
from describeClass import Describe


def open_csv(file):
    return pd.read_csv(file)


if __name__ == '__main__':
    if len(sys.argv[1:]) == 1:
        datas = open_csv(sys.argv[1])
        datas = datas[datas.columns[6:]]
        obj = Describe()
        print(obj.dataframe(obj, datas))
