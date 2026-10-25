
"""fedtree_hist.py - histogram-aggregation federated GBDT (federated quantile binning + summed gradient histograms).
Training LightGBM on the pooled data mapped to the global bin indices (one LightGBM bin per index) is exactly the protocol."""
import time, numpy as np, lightgbm as lgb
import fedtree_core as ftc
Q = 256; MAXB = 255

def client_sketch(X):                                     # what one client sends: Q+1 quantiles per feature
    return np.quantile(X, np.linspace(0, 1, Q + 1), axis=0)

def merge_edges(sketches, sizes):                         # server: weighted quantiles of the pooled client sketches
    P = np.concatenate(sketches, 0); W = np.concatenate([np.full(len(s), n / len(s)) for s, n in zip(sketches, sizes)])
    edges = []
    for f in range(P.shape[1]):
        o = np.argsort(P[:, f], kind='stable'); v, cw = P[o, f], np.cumsum(W[o]); cw /= cw[-1]
        cut = v[np.minimum(np.searchsorted(cw, np.linspace(0, 1, MAXB + 1)[1:-1]), len(v) - 1)]
        edges.append(np.unique(cut))
    return edges

def to_bins(X, edges):                                    # bin i = (e_{i-1}, e_i]; done locally by each client
    return np.stack([np.searchsorted(e, X[:, f], side='left') for f, e in enumerate(edges)], 1).astype(np.float32)

def run_hist(S, edges, seed, n_trees=300):
    t0 = time.time(); w = ftc.class_weights(S['y_train']); Xb = to_bins(S['X_train'], edges)
    prm = dict(ftc.BASE, seed=seed, max_bin=MAXB, min_data_in_bin=1, bin_construct_sample_cnt=len(Xb), feature_pre_filter=False)
    ds = lgb.Dataset(Xb, S['y_train'], weight=w[S['y_train']], params=prm).construct()
    nd = [len(np.unique(Xb[:, f])) for f in range(Xb.shape[1])]
    assert all(ds.feature_num_bin(f) == (v if v > 1 else 0) for f, v in enumerate(nd)), 'LightGBM merged bins - STOP'
    bst = lgb.train(prm, ds, num_boost_round=n_trees)
    Pv, Pt = bst.predict(to_bins(S['X_val'], edges)), bst.predict(to_bins(S['X_test'], edges)); out = ftc.score(S, Pv, Pt)
    leaves = [t['num_leaves'] for t in bst.dump_model()['tree_info']]
    return {'method': 'hist', 'seed': seed, 'n_trees': n_trees, 'final': out, 'history': [dict(round=n_trees, **out)],
            'bins_per_feature': [int(len(e) + 1) for e in edges], 'leaves_per_tree': leaves, 'n_features': int(Xb.shape[1]),
            'minutes': (time.time() - t0) / 60, 'pred_test': Pt.argmax(1).astype(np.uint8)}
