import sys
import pandas as pd
from sklearn.metrics import confusion_matrix
import numpy as np
import warnings

warnings.filterwarnings("ignore")
np.random.seed(11)



# 真阳性（TP），真阳性（TN），假阳性（FP）和假阴性（FN）
def tss_metrix(y_true, y_yred):  # a，target     b，true
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for i in range(len(y_yred)):
        if y_yred[i] == 1 and y_yred[i] == y_true[i]:
            tp += 1
        elif y_yred[i] == 1 and y_yred[i] != y_true[i]:
            fp += 1
        elif y_yred[i] == 0 and y_yred[i] == y_true[i]:
            tn += 1
        elif y_yred[i] == 0 and y_yred[i] != y_true[i]:
            fn += 1
    k = [[tp, fn], [fp, tn]]
    tss = tp / (tp + fn) - fp / (fp + tn)
    acc = (tp + tn) / (tp + tn + fp + fn)
    # return tss, acc
    return tp, tn, fp, fn


##注意：Metric类，可以根据4分类的混淆矩阵，计算2分类的TSS等各个指标；也可以根据2分类的混淆矩阵，计算2分类的TSS等各个指标！！！

class Metric(object):
    def __init__(self, y_true, y_pred, matrix=None):
        if matrix is None:
            self.__matrix = confusion_matrix(y_true, y_pred)
        else:
            self.__matrix = np.asmatrix(matrix)

    def Matrix(self):
        return self.__matrix

    def TP(self):
        tp = np.diag(self.__matrix)
        return tp.astype(float)

    def TN(self):
        tn = self.__matrix.sum() - (self.FP() + self.FN() + self.TP())
        return tn.astype(float)

    def FP(self):
        fp = self.__matrix.sum(axis=0) - np.diag(self.__matrix)
        return fp.astype(float)

    def FN(self):
        fn = self.__matrix.sum(axis=1) - np.diag(self.__matrix)
        return fn.astype(float)

    def TPRate(self):
        return self.TP() / (self.TP() + self.FN() + sys.float_info.epsilon)

    def TNRate(self):
        return self.TN() / (self.TN() + self.FP() + sys.float_info.epsilon)

    def FPRate(self):
        return 1 - self.TNRate()

    def FNRate(self):
        return 1 - self.TPRate()

    def Accuracy(self):
        ALL = self.TP() + self.FP() + self.TN() + self.FN()
        RIGHT = self.TP() + self.TN()
        return RIGHT / (ALL + sys.float_info.epsilon)

    def Recall(self):
        return self.TP() / (self.TP() + self.FN() + sys.float_info.epsilon)

    def Precision(self):
        return self.TP() / (self.TP() + self.FP() + sys.float_info.epsilon)

    def TSS(self):
        return self.TPRate() - self.FPRate()

    def HSS(self):
        P = self.TP() + self.FN()
        N = self.TN() + self.FP()
        up = 2 * (self.TP() * self.TN() - self.FN() * self.FP())
        below = P * (self.FN() + self.TN()) + N * (self.TP() + self.FP())
        return up / (below + sys.float_info.epsilon)

    def FAR(self):
        return self.FP() / (self.FP() + self.TP() + sys.float_info.epsilon)
