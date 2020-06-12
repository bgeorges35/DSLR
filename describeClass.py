#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import math


class Describe:

    @staticmethod
    def quartile3(sortedLst):
        len_ = len(sortedLst) - 1

        if len(sortedLst) % 2 == 1:
            q3 = sortedLst[int(len_ * 0.75)]
        else:
            q3 = (sortedLst[int(len_ * 0.75)] + sortedLst[int(len_ * 0.75) + 1]) / 2
        return q3

    @staticmethod
    def quartile1(sortedLst):
        len_ = len(sortedLst)
        if len_ % 2 == 1:
            q1 = sortedLst[int(len_ * 0.25)]
        else:
            q1 = (sortedLst[int(len_ * 0.25)] + sortedLst[int(len_ * 0.25) + 1]) / 2
            q1 = (sortedLst[int(len_ * 0.25)] + sortedLst[int(len_ * 0.25) + 1]) / 2
        return q1

    @staticmethod
    def median(sortedLst):
        len_ = int(len(sortedLst) - 1)
        if len(sortedLst) % 2 == 1:
            median = sortedLst[int(len_ / 2)]
        else:
            median = (sortedLst[int(len_ / 2)] + sortedLst[int((len_ / 2) + 1)]) / 2
        return median

    @staticmethod
    def min(col):
        ret = col[0]
        for x in col:
            if x < ret:
                ret = x
        return ret

    @staticmethod
    def max(col):
        ret = col[0]
        for x in col:
            if x > ret:
                ret = x
        return ret

    @staticmethod
    def dataframe(self, datas):
        count = []
        mean = []
        std = []
        min_ = []
        max_ = []
        median = []
        q1 = []
        q3 = []

        for name in datas:
            col = [x for x in datas[name].tolist() if str(x) != 'nan']
            count.append(len(col))
            mean.append((1 / count[-1]) * sum(col))
            std.append(math.sqrt(sum([(x - mean[-1]) ** 2 for x in col]) / (count[-1] - 1)))
            min_.append(self.min(col))
            max_.append(self.max(col))
            sortedLst = sorted(col)
            median.append(self.median(sortedLst))
            q1.append(self.quartile1(sortedLst))
            q3.append(self.quartile3(sortedLst))
        desc = list(zip(count, mean, std, min_, q1, median, q3, max_))
        column = datas.columns.values
        desc = np.transpose(desc)
        df = pd.DataFrame(desc, columns=column, index=['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max'])
        return df
