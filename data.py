import numpy as np


def process_data(XX, YYall, YY, seq_len, targets=[1], pred_type=-1):
    """
    :return: (Xss, Yss, stats), where
        Xss ~ [seq_len, n_seqs, x_dims]
        Yss ~ [n_seqs, len(targets)]
        stats ~ ([x_dims], [x_dims], len(targets), len(targets))
    """
    Len = XX.shape[0]
    XX = XX.reshape((Len * 24, XX.shape[2])).astype('f')
    YY = YY.reshape((Len * 24,)).astype('f')
    YYall = YYall.reshape((20, Len * 24)).astype('f')
    Xss, Yss = [], []
    max_target_shift = max(targets)
    assert all(t>0 for t in targets)
    for j in range(0, XX.shape[0] - seq_len - max_target_shift):
        xs = XX[j: j+seq_len]
        y_hists = YY[j: j+seq_len]
        if np.isnan(xs.sum()) or np.isnan(y_hists.sum()):
            continue
        xs = np.concatenate([xs, y_hists[:,None]], axis=-1)
        if pred_type < 0:  # predict average
            ys = YY[j+seq_len-1:][targets]
        else:
            # first add single-station history as input
            y_hists = YYall[pred_type][j: j+seq_len]
            assert not np.isnan(y_hists.sum())
            xs = np.concatenate([xs, y_hists[:,None]], axis=-1)
            # 
            ys = YYall[pred_type,j+seq_len-1:][targets]
        if np.isnan(ys.sum()):
            continue
        Xss.append(xs)
        Yss.append(ys)
    print(len(Xss), 'sequences prepared')
    Xss = np.transpose(np.array(Xss), [1,0,2])
    Yss = np.array(Yss)
    stats = (
        Xss.mean(axis=(0,1)),
        Xss.std(axis=(0,1)) + 1e-8,
        Yss.mean(),
        Yss.std()
    )
    return Xss, Yss, stats


def normalize_dset(xss, yss, stats):
    xss = (xss - stats[0]) / stats[1]
    yss = (yss - stats[2]) / stats[3]
    return xss, yss
