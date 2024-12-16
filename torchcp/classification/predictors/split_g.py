import warnings
import math
import torch

from torchcp.classification.predictors.base import BasePredictor
from torchcp.utils.common import calculate_conformal_value


class SplitPredictor_GNN(BasePredictor):
    """
    Split Conformal Prediction (Vovk et a., 2005) for GNNs.
    Book: https://link.springer.com/book/10.1007/978-3-031-06649-8.
    
    :param score_function: non-conformity score function.
    :param pred: the estimation results.
    :param model: a pytorch model.
    :param temperature: the temperature of Temperature Scaling.
    """
    def __init__(self, score_function, model=None, temperature=1):
        super().__init__(score_function, model, temperature)

    def score_function(self, logits):
        '''without labels'''
        scores = self.score_function(logits).to(self._device)
        return scores

    #############################
    # The calibration process
    ############################
    def calibrate(self, cal_dataloader, alpha):
        self._model.eval()
        logits_list = []
        labels_list = []
        with torch.no_grad():
            for examples in cal_dataloader:
                tmp_x, tmp_labels = examples[0].to(self._device), examples[1].to(self._device)
                tmp_logits = self._logits_transformation(self._model(tmp_x)).detach()   
                logits_list.append(tmp_logits)
                labels_list.append(tmp_labels)
            logits = torch.cat(logits_list).float()
            labels = torch.cat(labels_list)
        self.calculate_threshold(logits, labels, alpha)

    def calculate_threshold(self, logits, labels, alpha, interpolation):
        logits = logits.to(self._device)
        labels = labels.to(self._device)
        scores = self.score_function(logits, labels)    # cal_scores
        self.q_hat = self._calculate_conformal_value(scores, alpha, interpolation) # q_hat

    def _calculate_conformal_value(self, scores, alpha, interpolation='higher', marginal_q_hat = torch.inf):
        return calculate_conformal_value(scores, alpha, interpolation, marginal_q_hat)

    #############################
    # The prediction process
    ############################
    def predict(self, x_batch):
        """
        The input of score function is softmax probability.

        :param x_batch: a batch of instances.
        """
        self._model.eval()
        if self._model != None:
            x_batch = self._model(x_batch.to(self._device)).float()
        x_batch = self._logits_transformation(x_batch).detach()
        sets = self.predict_with_logits(x_batch)
        return sets

    def predict_with_logits(self, logits, q_hat=None):
        """
        The input of score function is softmax probability.
        if q_hat is not given by the function 'self.calibrate', the construction progress of prediction set is a naive method.

        :param logits: model output before softmax.
        :param q_hat: the conformal threshold.

        :return: prediction sets
        """
        
            
        scores = self.score_function(logits).to(self._device)
        if q_hat is None:
            q_hat = self.q_hat

        # print("q_hat: " + str(q_hat))
        S = self._generate_prediction_set(scores, q_hat)
        
        return S

    
    def evaluate(self, prediction_sets, val_labels):
        res_dict = {"Coverage_rate": self._metric('coverage_rate')(prediction_sets, val_labels),
                    "Average_size": self._metric('average_size')(prediction_sets, val_labels)}
        return res_dict

