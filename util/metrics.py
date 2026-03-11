import math

import numpy as np
import pandas as pd
EPS = 1e-8

def update_metrics(pred, label, statistics, target_variate):
    if isinstance(pred, tuple):
        pred = pred[0]

    #
    if pred.dim() == 3 and pred.size(-1) == 1:
        pred = pred.squeeze(-1)          #
    if label.dim() == 3 and label.size(-1) == 1:
        label = label.squeeze(-1)        #[B, T, 1]

    if target_variate is not None:
        if pred.dim() == 3 and pred.size(-1) > 1:
            pred = pred[:, :, target_variate]
        if label.dim() == 3 and label.size(-1) > 1:
            label = label[:, :, target_variate]

        if pred.dim() == 3 and pred.size(-1) == 1:
            pred = pred.squeeze(-1)
        if label.dim() == 3 and label.size(-1) == 1:
            label = label.squeeze(-1)


    if pred.shape != label.shape:
        raise RuntimeError(f"Shape mismatch. pred={pred.shape}, label={label.shape}")

    balance = pred - label
    statistics['y_sum'] += label.abs().sum().item()
    statistics['total'] += label.numel()
    statistics['MAE'] += balance.abs().sum().item()
    statistics['MSE'] += (balance ** 2).sum().item()

    denom = pred.abs() + label.abs() + EPS
    statistics['SMAPE'] += (2 * balance.abs() / denom).sum().item()


def calculate_metrics(statistics):
    MSE_sum = statistics['MSE']
    MAE_sum = statistics['MAE']
    smape_sum = statistics['SMAPE']
    total   = statistics['total']
    y_sum   = statistics['y_sum']

    mse = MSE_sum / total
    mae = MAE_sum / total
    rmse = math.sqrt(mse)
    smape = (smape_sum / total) * 100

    metrics = {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'SMAPE': smape,
    }
    return metrics

def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    d += 1e-12
    return 0.01*(u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))


def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))


def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    smape = SMAPE(pred, true)
    mspe = MSPE(pred, true)
    rse = RSE(pred, true)
    corr = CORR(pred, true)

    return mae, mse, rmse, smape, mspe, rse, corr


def calc_ic(pred=None, label=None, index=None, df=None, return_type='all', reduction='sum'):
    if df is None:
        if isinstance(pred, tuple):
            pred = pred[0]
        df = pd.DataFrame({'pred': pred, 'label': label}, index=index)
    if index is None:
        res = []
        if return_type != 'ric':
            res.append(df['pred'].corr(df['label']))
        if return_type != 'ic':
            res.append(df['pred'].corr(df['label'], method='spearman'))
        return res
    else:
        groups = df.groupby('datetime')
        res = []
        if return_type != 'ric':
            res.append(groups.apply(lambda df: df["pred"].corr(df["label"], method="pearson")))
        if return_type != 'ic':
            res.append(groups.apply(lambda df: df["pred"].corr(df["label"], method="spearman")))
        if reduction == 'sum':
            return [r.sum() for r in res] + [len(groups)]
        elif reduction == 'mean':
            return [r.mean() for r in res]
        else:
            return [r.to_numpy().tolist() for r in res]
