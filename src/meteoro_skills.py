#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 19:48:36 2019

@author: dvdgmf
"""


class CategoricalMetrics:

    @staticmethod
    def get_TPTN(obs, pred):
        if len(obs) == len(pred):
            if all(O in [0, 1] for O in obs) and all(P in [0, 1] for P in pred):
                output_list = []
                test = {
                    (0, 0): "TN",
                    (1, 1): "TP",
                    (0, 1): "FN",
                    (1, 0): "FP"
                }
                for O, P in zip(obs, pred):
                    output_list.append(test.get((O, P)))
                return output_list
            else:
                print('Invalid values present in Y-Observed and/or Y-Predicted.'
                      'Only binary values 0 and 1 are supported!')
        else:
            print('Invalid list size. Y-Observed and Y-Predicted must match!')

    def metrics(self, obs, pred):
        tptn = self.get_TPTN(obs, pred)
        tn = tptn.count('TN')
        tp = tptn.count('TP')
        fn = tptn.count('FN')
        fp = tptn.count('FP')
        accuracy = (tp + tn) / len(tptn)
        bias = (tp + fp) / (tp + fn)
        pod = tp / (tp + fn)
        try:
            pofd = fp / (fp + tn)
        except ZeroDivisionError as e:
            pofd = 0
            print(e)
        far = fp / (tp + fp)
        csi = tp / (tp + fp + fn)
        ph = ((tp + tn) * (tp + fp)) / len(tptn)
        ets = (tp - ph) / (tp + fp + fn - ph)
        hss = ((tp * tn) - (fp * fn)) / ((((tp + fn) * (fn + tn)) + ((tp + fp) * (fp + tn))) / 2)
        hkd = pod - pofd
        return accuracy, bias, pod, pofd, far, csi, ph, ets, hss, hkd

    def accuracy(self, obs, pred):
        tptn = self.get_TPTN(obs, pred)
        tn = tptn.count('TN')
        tp = tptn.count('TP')
        accuracy = (tp+tn)/len(tptn)
        return accuracy

    def bias(self, obs, pred):
        tptn = self.get_TPTN(obs, pred)
        tp = tptn.count('TP')
        fn = tptn.count('FN')
        fp = tptn.count('FP')
        bias = (tp+fp)/(tp+fn)
        return bias

    def pod(self, obs, pred):
        tptn = self.get_TPTN(obs, pred)
        tp = tptn.count('TP')
        fn = tptn.count('FN')
        pod = tp/(tp+fn)
        return pod

    def pofd(self, obs, pred):
        tptn = self.get_TPTN(obs, pred)
        tn = tptn.count('TN')
        fp = tptn.count('FP')
        pofd = fp/(fp+tn)
        return pofd

    def far(self, obs, pred):
        tptn = self.get_TPTN(obs, pred)
        tp = tptn.count('TP')
        fp = tptn.count('FP')
        far = fp/(tp+fp)
        return far

    def csi(self, obs, pred):
        tptn = self.get_TPTN(obs, pred)
        tp = tptn.count('TP')
        fn = tptn.count('FN')
        fp = tptn.count('FP')
        csi = tp/(tp+fp+fn)
        return csi

    def ets(self, obs, pred):
        tptn = self.get_TPTN(obs, pred)
        tn = tptn.count('TN')
        tp = tptn.count('TP')
        fn = tptn.count('FN')
        fp = tptn.count('FP')
        ph = ((tp+tn)*(tp+fp))/len(tptn)
        ets = (tp - ph)/(tp+fp+fn-ph)
        return ets

    def hss(self, obs, pred):
        tptn = self.get_TPTN(obs, pred)
        tn = tptn.count('TN')
        tp = tptn.count('TP')
        fn = tptn.count('FN')
        fp = tptn.count('FP')
        hss = ((tp*tn)-(fp*fn))/((((tp+fn)*(fn+tn))+((tp+fp)*(fp+tn)))/2)
        return hss

    def hkd(self, obs, pred):
        hkd = self.pod(obs, pred) - self.pofd(obs, pred)
        return hkd


# ---------------------------------------------
#obs =   [0, 1, 1, 0, 0, 0, 1, 1, 0]
#pred =  [0, 1, 0, 0, 1, 0, 1, 1, 0]
#
#mozao_tools = CategoricalMetrics()
#print('metrics: ', mozao_tools.metrics(obs, pred))
# print('\nOBS  = ', obs,
#       '\nPRED = ', pred,
#       '\nTPTN = ', mozao_tools.get_TPTN(obs, pred),
#       '\nMETRICS:',
#       '\nAccuracy == ', mozao_tools.accuracy(obs, pred),
#       '\nBias ====== ', mozao_tools.bias(obs, pred),
#       '\nPOD ======= ', mozao_tools.pod(obs, pred),
#       '\nPOFD ====== ', mozao_tools.pofd(obs, pred),
#       '\nFAR ======= ', mozao_tools.far(obs, pred),
#       '\nCSI ======= ', mozao_tools.csi(obs, pred),
#       '\nETS ======= ', mozao_tools.ets(obs, pred),
#       '\nHSS ======= ', mozao_tools.hss(obs, pred),
#       '\nHKD ======= ', mozao_tools.hkd(obs, pred),)
