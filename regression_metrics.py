# *_*coding:utf-8 *_*
import numpy as np
from scipy.stats import pearsonr
import torch
import torch.nn as nn

def cal_acc(output, target, return_finegrained=False):
    """
    :param output: (B, C), numpy array
    :param target: (B, C), numpy array
    :return: scalar, (C,) numpy array
    """
    # acc = (1 - np.abs(output - target)).mean().clip(min=0)
    accs = (1 - np.abs(output - target)).mean(axis=0)
    acc = np.mean(accs)
    if not return_finegrained:
        return acc
    else:
        return acc, accs

def cal_mae(output, target, return_finegrained=False):
    """
    :param output: (B, C), numpy array
    :param target: (B, C), numpy array
    :return: scalar, (C,) numpy array
    """
    maes = np.abs(output - target).mean(axis=0)
    mae = np.mean(maes)
    if not return_finegrained:
        return mae
    else:
        return mae, maes

def cal_mse(output, target, return_finegrained=False):
    """
    :param output: (B, C), numpy array
    :param target: (B, C), numpy array
    :return: scalar, (C,) numpy array
    """
    mse = np.square(output - target).mean()
    if not return_finegrained:
        return mse
    else:
        mses = np.square(output - target).mean(axis=0)
        return mse, mses

def cal_rmse(output, target, return_finegrained=False):
    """
    :param output: (B, C), numpy array
    :param target: (B, C), numpy array
    :return: scalar, (C,) numpy array
    """
    rmses = np.sqrt(np.square(output - target).mean(axis=0))
    rmse = np.mean(rmses)
    if not return_finegrained:
        return rmse
    else:
        return rmse, rmses

def cal_pcc(output, target, return_finegrained=False):
    """
    :param output: (B, C), numpy array
    :param target: (B, C), numpy array
    :return: scalar, (C,) numpy array
    """
    num_samples, n_dims = output.shape
    if num_samples == 1: # 'pearsonr' will lead to bug 'x and y must have length at least 2.'!
        pccs = [1.0] * n_dims
    else:
        pccs = [pearsonr(output[:,i], target[:,i])[0] for i in range(n_dims)]
    pcc = np.mean(pccs)
    if not return_finegrained:
        return pcc
    else:
        return pcc, pccs

def cal_ccc(output, target, return_finegrained=False):
    """
    :param output: (B, C), numpy array
    :param target: (B, C), numpy array
    :return: scalar, (C,) numpy array
    """
    n_dims = output.shape[-1]
    cccs = []
    for i in range(n_dims):
        preds, labels = output[:,i], target[:,i]
        preds_mean, labels_mean = np.mean(preds), np.mean(labels)
        cov_mat = np.cov(preds, labels)  # Note: unbiased
        covariance = cov_mat[0, 1]
        preds_var, labels_var = cov_mat[0, 0], cov_mat[1, 1]

        # pcc = covariance / np.sqrt(preds_var * labels_var)
        ccc = 2.0 * covariance / (preds_var + labels_var + (preds_mean - labels_mean) ** 2)

        cccs.append(ccc)
    ccc = np.mean(cccs)
    if not return_finegrained:
        return ccc
    else:
        return ccc, cccs


def cal_regression_metrics(output, target, return_finegrained=False):
    acc = cal_acc(output, target, return_finegrained)
    mae = cal_mae(output, target, return_finegrained)
    mse = cal_mse(output, target, return_finegrained)
    rmse = cal_rmse(output, target, return_finegrained)
    pcc = cal_pcc(output, target, return_finegrained)
    ccc = cal_ccc(output, target, return_finegrained)
    return {
        'acc': acc,
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'pcc': pcc,
        'ccc': ccc,
    }


class CCCLoss(nn.Module):
    def __init__(self, label_dim=1, include_batch_dim=True):
        super(CCCLoss, self).__init__()
        self.label_dim = label_dim # number of annotated dimensions (1, 2, 3, ...)
        self.include_batch_dim=include_batch_dim # when calculate ccc, merge batch dimension into temporal dimension to calculate ccc

    def forward(self, y_pred, y_true):
        """
        :param y_pred: (B, T*C) or (B,C), T is temporal, C is label dimension (e.g., arousal/valence)
        :param y_true: (B, T*C) or (B,C), the former: fine-grained frame-level task, the latter: video-level task
        :return: ccc loss, scalar
        """
        # reshape
        assert y_pred.ndim == 2, 'Error: only support 2-dim (B, T*C) or (B,C) input!'
        batch_size = y_pred.size(0)
        if self.include_batch_dim: # for both video-level (B, C) and frame-level (B, T*C) task
            y_pred = y_pred.view(1, -1, self.label_dim) # (B, T*C) --> (1, B*T, C) or (B,C) --> (1, B, C)
            y_true = y_true.view(1, -1, self.label_dim)
        else: # only for frame-level task, (B, T*C)
            y_pred = y_pred.view(batch_size, -1, self.label_dim) # (B, T*C) --> (B, T, C)
            y_true = y_true.view(batch_size, -1, self.label_dim)
            assert y_pred.size(1) > 1, 'Error: do not support video-level task, please set include_batch_dim=True!'

        # cal mean, variance, and covariance
        y_true_mean = torch.mean(y_true, dim=1, keepdim=True) # along temporal dim, (B, 1, C)
        y_pred_mean = torch.mean(y_pred, dim=1, keepdim=True)

        y_true_var = torch.mean((y_true - y_true_mean) ** 2, dim=1, keepdim=True) # (B, 1, C)
        y_pred_var = torch.mean((y_pred - y_pred_mean) ** 2, dim=1, keepdim=True)
        cov = torch.mean((y_true - y_true_mean) * (y_pred - y_pred_mean), dim=1, keepdim=True)

        ccc = torch.mean(2.0 * cov / (y_true_var + y_pred_var + (y_true_mean - y_pred_mean) ** 2)) # along batch and label dim
        ccc_loss = 1.0 - ccc # scalar

        return ccc_loss


class PCCLoss(nn.Module):
    def __init__(self, label_dim=1, include_batch_dim=True):
        super().__init__()
        self.label_dim = label_dim # number of annotated dimensions (1, 2, 3, ...)
        self.include_batch_dim=include_batch_dim # when calculate ccc, merge batch dimension into temporal dimension to calculate ccc

    def forward(self, y_pred, y_true):
        """
        :param y_pred: (B, T*C) or (B,C), T is temporal, C is label dimension (e.g., arousal/valence)
        :param y_true: (B, T*C) or (B,C), the former: fine-grained frame-level task, the latter: video-level task
        :return: ccc loss, scalar
        """
        # reshape
        assert y_pred.ndim == 2, 'Error: only support 2-dim (B, T*C) or (B,C) input!'
        batch_size = y_pred.size(0)
        if self.include_batch_dim: # for both video-level (B, C) and frame-level (B, T*C) task
            y_pred = y_pred.view(1, -1, self.label_dim) # (B, T*C) --> (1, B*T, C) or (B,C) --> (1, B, C)
            y_true = y_true.view(1, -1, self.label_dim)
        else: # only for frame-level task, (B, T*C)
            y_pred = y_pred.view(batch_size, -1, self.label_dim) # (B, T*C) --> (B, T, C)
            y_true = y_true.view(batch_size, -1, self.label_dim)
            assert y_pred.size(1) > 1, 'Error: do not support video-level task, please set include_batch_dim=True!'

        # cal mean, variance, and covariance
        y_true_mean = torch.mean(y_true, dim=1, keepdim=True) # along temporal dim, (B, 1, C)
        y_pred_mean = torch.mean(y_pred, dim=1, keepdim=True)

        y_true_var = torch.mean((y_true - y_true_mean) ** 2, dim=1, keepdim=True) # (B, 1, C)
        y_pred_var = torch.mean((y_pred - y_pred_mean) ** 2, dim=1, keepdim=True)
        cov = torch.mean((y_true - y_true_mean) * (y_pred - y_pred_mean), dim=1, keepdim=True)

        pcc = torch.mean(cov / (torch.sqrt(y_true_var * y_pred_var) + 1e-6)) # along batch and label dim
        pcc_loss = 1.0 - pcc # scalar

        return pcc_loss
