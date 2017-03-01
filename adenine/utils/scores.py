"""Validation utils for clustering algorithms.

Notes
-----
Precision, recall and F score
    In multiclass classification / clustering, a confusion matrix can be
    obtained. To validate the result, one can use precision, recall and
    f score. These are obtained using TP, FP, FN, TN.
    In particular, for each class (true label) x, in a confusion matrix cm:
    - true positive: diagonal position, cm(x, x).
    - false positive: sum of column x (without main diagonal),
        sum(cm(:, x)) - cm(x, x).
    - false negative: sum of row x (without main diagonal),
        sum(cm(x, :), 2) - cm(x, x).
    - true negative: sum of all the matrix without tp, fp, fn.

    Averaging over all classes (with or without weighting) gives values for the
    entire model.

Author: Federico Tomasi
Copyright (c) 2016, Federico Tomasi.
Licensed under the FreeBSD license (see LICENSE.txt).
"""
import matplotlib; matplotlib.use('Agg')
import numpy as np
import pandas as pd
import seaborn as sns


def get_clones_real_estimated(filename):
    """Get true and estimated labels from a partis-generated dataset."""
    df = pd.read_csv(filename, dialect='excel-tab', header=0,
                     usecols=('SEQUENCE_ID', 'CLONE'))
    df['CLONE_ID'] = df['SEQUENCE_ID'].str.split('_').apply(lambda x: x[3])

    clone_ids = np.array(df['CLONE_ID'], dtype=str)
    found_ids = np.array(df['CLONE'], dtype=str)
    return clone_ids, found_ids


def order_cm(cm):
    """Reorder a multiclass confusion matrix."""
    # reorder rows
    idx_rows = np.max(cm, axis=1).argsort()[::-1]
    b = cm[idx_rows, :]

    # reorder cols
    max_idxs = np.ones(b.shape[1], dtype=bool)
    final_idxs = []
    for i, row in enumerate(b.copy()):
        if i == b.shape[0] or not max_idxs.any():
            break
        row[~max_idxs] = np.min(cm) - 1
        max_idx = np.argmax(row)
        final_idxs.append(max_idx)
        max_idxs[max_idx] = False

    idx_cols = np.append(np.array(final_idxs, dtype=int),
                         np.argwhere(max_idxs).T[0])  # residuals

    # needs also this one
    b = b[:, idx_cols]
    bb = b.copy()
    max_idxs = np.ones(b.shape[0], dtype=bool)
    final_idxs = []
    for i in range(b.shape[1]):
        # for each column
        if i == b.shape[1] or not max_idxs.any():
            break
        col = bb[:, i]
        col[~max_idxs] = -1
        max_idx = np.argmax(col)
        final_idxs.append(max_idx)
        max_idxs[max_idx] = False

    idx_rows2 = np.append(np.array(final_idxs, dtype=int),
                          np.argwhere(max_idxs).T[0])  # residuals

    idx = np.argsort(idx_rows)
    return b[idx_rows2, :], idx_rows2[idx], idx_cols


def confusion_matrix(true_labels, estimated_labels, ordered=True):
    """Return a confusion matrix in a multiclass / multilabel problem."""
    true_labels = np.array(true_labels, dtype=str)
    estimated_labels = np.array(estimated_labels, dtype=str)
    if true_labels.shape[0] != estimated_labels.shape[0]:
        raise ValueError("Inputs must have the same dimensions.")
    rows = np.unique(true_labels)
    cols = np.unique(estimated_labels)

    # padding only on columns
    cm = np.zeros((rows.shape[0], max(cols.shape[0], rows.shape[0])))
    from collections import Counter
    for i, row in enumerate(rows):
        idx_rows = true_labels == row
        counter = Counter(estimated_labels[idx_rows])
        for g in counter:
            idx_col = np.where(cols == g)[0][0]
            cm[i, idx_col] += counter[g]

    cols = np.append(cols, ['pad'] * (cm.shape[1] - cols.shape[0]))
    if ordered:
        cm, rr, cc = order_cm(cm)
        rows, cols = rows[rr], cols[cc]
    return cm, rows, cols


def precision_recall_fscore(a, method='micro', beta=1.):
    """Return a precision / recall value for multiclass confuison matrix cm.

    See
    http://stats.stackexchange.com/questions/44261/how-to-determine-the-quality-of-a-multiclass-classifier
    """
    def _single_measures(a, i):
        tp = a[i, i]
        fp = np.sum(a[:, i]) - tp
        fn = np.sum(a[i, :]) - tp
        tn = a.sum() - tp - fp - fn
        return tp, fp, fn, tn

    singles = zip(*[_single_measures(a, i) for i in range(min(a.shape))])
    tps, fps, fns, tns = map(lambda x: np.array(list(x), dtype=float), singles)

    if method == 'micro':
        precision = float(tps.sum()) / (tps + fps).sum()
        recall = float(tps.sum()) / (tps + fns).sum()
    elif method == 'macro':
        sum_ = tps + fps
        idx = np.where(sum_)
        precision = (tps[idx] / sum_[idx]).mean()

        sum_ = tps + fns
        idx = np.where(sum_)
        recall = (tps[idx] / sum_[idx]).mean()
    fscore = (1 + beta * beta) * precision * recall / \
        (beta * beta * precision + recall)
    return precision, recall, fscore


def show_heatmap(filename):
    """Show confusion matrix given of a partis-generated tab-delimited db."""
    true_labels, estimated_labels = get_clones_real_estimated(filename)
    cm, rows, cols = confusion_matrix(true_labels, estimated_labels)
    df = pd.DataFrame(cm, index=rows, columns=cols)
    sns.heatmap(df)
    sns.plt.show()
