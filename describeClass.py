#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import math


class Describe:

    def Quartile3(self, sortedLst):
        len_ = len(sortedLst) - 1
        
        if (len(sortedLst) % 2 == 1):
            Q3 = sortedLst[int(len_ * 0.75)]
        else:
            Q3 = (sortedLst[int(len_ * 0.75)] + sortedLst[int(len_ * 0.75) + 1]) / 2
        return Q3

        
    def Quartile1(self, sortedLst):
        # print(sortedLst)
        len_ = len(sortedLst)
        if (len_ % 2 == 1):
            Q1 = sortedLst[int(len_ * 0.25)]
        else:
            Q1 = (sortedLst[int(len_ * 0.25)] + sortedLst[int(len_ * 0.25) + 1]) / 2 
            # print('pair')
            Q1 = (sortedLst[int(len_ * 0.25)] + sortedLst[int(len_ * 0.25) + 1]) / 2
        return Q1
    
    
    def Median(self, sortedLst):
        len_ = int(len(sortedLst) - 1)
        if (len(sortedLst) % 2 == 1):
            median = sortedLst[int(len_ / 2)]
        else:
            median = (sortedLst[int(len_ / 2)] + sortedLst[int((len_ / 2) + 1)]) / 2
        return median
    
    def Min(self, col):
        ret = col[0]
        for x in col:
            if x < ret:
                ret = x
        return ret
    
    def Max(self, col):
        ret = col[0]
        for x in col:
            if x > ret:
                ret = x
        return ret
    def DataFrame(self, datas):
        count = []
        mean = []
        std = []
        min_ = []
        max_ = []
        median = []
        Q1 = []
        Q3 = []
    
        for name in datas:
            # print(name)
            col = [x for x in datas[name].tolist() if str(x) != 'nan']
            # print(col)
            count.append(len(col))
            mean.append((1/count[-1]) * sum(col))
            std.append(math.sqrt(sum([(x - mean[-1]) ** 2 for x in col]) / (count[-1] - 1)))
            min_.append(self.Min(col))
            max_.append(self.Max(col))
            sortedLst = sorted(col)            
            median.append(self.Median(sortedLst))
            Q1.append(self.Quartile1(sortedLst))
            Q3.append(self.Quartile3(sortedLst))
        desc = list(zip(count, mean, std, min_, Q1, median, Q3, max_))
        column = datas.columns.values
        desc = np.transpose(desc)
        df = pd.DataFrame(desc, columns =column, index=['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max'])
        return df