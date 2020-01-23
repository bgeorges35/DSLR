# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import sys
import math

def open_csv(file):
    return pd.read_csv(file)

if __name__ == '__main__':
    if (len(sys.argv[1:]) == 1):
        datas = open_csv(sys.argv[1])
        datas = datas[datas.columns[6:]]
        # datas.astype(float)
        count = []
        mean = []
        std = []
        min_ = []
        quantile1 = []
        median = []
        quantile3 = []
        # max_ []
        
        for name in datas:
            print(name)
            col = [x for x in datas[name].tolist() if str(x) != 'nan']
            # print(col)
            # count.append(len(col))
            # mean.append((1/count[-1]) * sum(col))
            # std.append(math.sqrt(sum([(x - mean[-1]) ** 2 for x in col]) / (count[-1] - 1)))
            # min_.append(min(col))
            sortedLst = sorted(col)            
            if (lstLen % 2 == 1):
                median.append(sortedLst[int((len(col) - 1) / 2)])
            else:
                print((sortedLst[int((len(col) - 1) / 2)] + sortedLst[int((len(col) - 1) / 2) + 1]) / 2)
            # if len(sort) % 2 == 1:
            #   median.append( sort[int((len(sort)) / 2)] )
            # else:
            #     median.append(sort[int((len(sort) + 1) / 2)])
            # print(median)
            # break;
            # 1quantile.append()
            
            # 3quantile.append()
        # print(count)
        # print(mean)
        # print(std)
        # print(min_)
        # print(median)
        a = datas['Arithmancy'].median()
        print(a)
        print(pd.DataFrame.describe(datas))