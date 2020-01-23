# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import sys
import math

def open_csv(file):
    return pd.read_csv(file)


def Quartile3(sortedLst):
    len_ = len(sortedLst) - 1
    
    if (len(sortedLst) % 2 == 1):
        Q3 = sortedLst[len_ / 4]
    else:
        Q3 = sortedLst[len_ / 4] + ((sortedLst[(len_ / 4) + 1] - sortedLst[len_ / 2]) * (3 / 4))
    return Q3

        
def Quartile1(sortedLst):
    # print(sortedLst)
    len_ = len(sortedLst)
    
    if (len_ % 2 == 1):
        Q1 = sortedLst[int(len_ / 4)]
    else:
        print('pair')
        Q1 = sortedLst[int(len_ / 4)] + ((sortedLst[int((len_ / 4) + 1)] - sortedLst[int(len_ / 4)]) * (3 / 4))
    print(Q1)
    return Q1


def Median(sortedLst):
    len_ = int(len(sortedLst) - 1)
    if (len(sortedLst) % 2 == 1):
        median = sortedLst[int(len_ / 2)]
    else:
        median = (sortedLst[int(len_ / 2)] + sortedLst[int((len_ / 2) + 1)]) / 2
    return median


if __name__ == '__main__':
    if (len(sys.argv[1:]) == 1):
        datas = open_csv(sys.argv[1])
        datas = datas[datas.columns[6:]]
        # datas.astype(float)
        count = []
        mean = []
        std = []
        min_ = []
        Q1 = []
        median = []
        Q3 = []
        # max_ []
        
        for name in datas:
            # print(name)
            col = [x for x in datas[name].tolist() if str(x) != 'nan']
            # print(col)
            # count.append(len(col))
            # mean.append((1/count[-1]) * sum(col))
            # std.append(math.sqrt(sum([(x - mean[-1]) ** 2 for x in col]) / (count[-1] - 1)))
            # min_.append(min(col))
            sortedLst = sorted(col)            
            median.append(Median(sortedLst))
            Q1.append(Quartile1(sortedLst))
            break;
        # print(count)
        # print(mean)
        # print(std)
        # print(min_)
        # print(median)
        # a = datas['Arithmancy'].median()
        # print(a)
        a = pd.DataFrame.describe(datas)