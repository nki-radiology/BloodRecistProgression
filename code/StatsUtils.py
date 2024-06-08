import numpy as np
import pandas as pd
import scipy.stats as stats
import sklearn.metrics as metrics


def safe_ppv(y_true, y_pred):
    if y_true.std() == 0: return np.nan
    return metrics.precision_score(y_true, y_pred)


def safe_sensitivity(y_true, y_pred):
    if y_true.std() == 0: return np.nan
    cm = metrics.confusion_matrix(y_true, y_pred)
    return cm[0, 0] / (cm[0, 1] + cm[0, 0])


def safe_specificity(y_true, y_pred):
    if y_true.std() == 0: return np.nan
    cm = metrics.confusion_matrix(y_true, y_pred)
    return cm[1, 1] / (cm[1, 1] + cm[1, 0])


def safe_prc(y_true, y_pred):
    if y_true.std() == 0: return np.nan
    return metrics.average_precision_score(y_true, y_pred)


def safe_f1score(y_true, y_pred):
    if y_true.std() == 0: return np.nan
    return metrics.f1_score(y_true, y_pred)


def safe_rocauc(y_true, y_pred):
    return metrics.roc_auc_score(y_true, y_pred) if y_true.std() > 0 else np.nan


def safe_pval(y_true, y_pred):
    return stats.mannwhitneyu(y_pred[y_true == 1], y_pred[y_true < 1]).pvalue if y_true.std() > 0 else np.nan


def diagnostic_performance_analysis(y_true, y_pred, model, target, fold, bootstrap=1000, cutoff=None):
    assert y_true.size == y_pred.size
    result = []
    df = pd.DataFrame(np.array([y_true, y_pred]).T, columns=['y_true', 'y_pred'])
    for random_state in range(bootstrap):
        np.random.seed(random_state)
        df_ = df.groupby('y_true').apply(lambda x: x.sample(frac=1., replace=True, random_state=random_state))

        y_pred_, y_true_ = df_['y_pred'].to_numpy(), df_['y_true'].to_numpy()
        if not isinstance(cutoff, np.float):
            cutoff = np.median(y_pred_)
        result.append({
            'roc_auc': round(safe_rocauc(y_true_, y_pred_), 2),
            'avg_prc': round(safe_prc(y_true_, y_pred_), 2),
            'f1_scr': round(safe_f1score(y_true_, y_pred_ > cutoff), 2),
            'sensit': round(safe_sensitivity(y_true_, y_pred_ > cutoff), 2),
            'specif': round(safe_specificity(y_true_, y_pred_ > cutoff), 2),
            'ppv': round(safe_ppv(y_true_, y_pred_ > cutoff), 2),
            'npv': round(safe_ppv(1. - y_true_, y_pred_ <= cutoff), 2)
        })

    summary = {'N': y_true.size, 'N_pos': y_true.sum(), 'N_neg': (1. - y_true).sum(),
               'rocauc': safe_rocauc(y_true, y_pred)
               }

    for k in result[-1].keys():
        arr = np.array([r[k] for r in result])
        summary.update({k + '_50': np.percentile(arr, 50)})
        summary.update({k + '_2.5': np.percentile(arr, 2.5)})
        summary.update({k + '_97.5': np.percentile(arr, 97.5)})
        summary.update({k + '_min': np.min(arr)})
        summary.update({k + '_max': np.max(arr)})

    summary.update({'pval': round(safe_pval(y_true, y_pred), 3),
                    'median_prob': np.median(y_pred),
                    'fold': fold,
                    'target': target,
                    'model': model
                    })

    return summary
