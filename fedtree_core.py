
"""fedtree_core.py - federated gradient-boosted trees on the leak-free CIC IoT-DIAD 2024 splits (8 classes)."""
import time, numpy as np, lightgbm as lgb
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
NC = 8; LAB = np.arange(NC)
mf1 = lambda y, p: float(f1_score(y, p, average='macro', labels=LAB))
pcf = lambda y, p: [float(v) for v in f1_score(y, p, average=None, labels=LAB)]
BASE = dict(objective='multiclass', num_class=NC, learning_rate=0.05, num_leaves=63, verbose=-1, num_threads=0)

def class_weights(y_all):
    cnt = np.bincount(y_all, minlength=NC).astype(float)            # aggregate class counts (shareable statistic)
    w = np.where(cnt > 0, len(y_all) / (NC * np.maximum(cnt, 1)), 0.0); return w   # 'balanced' weights

def score(S, P_val, P_test):
    pv, pt = P_val.argmax(1), P_test.argmax(1)
    return {'val_macro_f1': mf1(S['y_val'], pv), 'test_macro_f1': mf1(S['y_test'], pt), 'test_accuracy': float(accuracy_score(S['y_test'], pt)),
            'test_per_class_f1': pcf(S['y_test'], pt), 'n_pred_classes': int(len(np.unique(pt)))}

def run_ens(S, parts, seed, n_trees=300):
    t0 = time.time(); w = class_weights(S['y_train']); n_k = np.array([len(p) for p in parts], float); a = n_k / n_k.sum()
    Pv = np.zeros((len(S['y_val']), NC)); Pt = np.zeros((len(S['y_test']), NC)); classes_seen = []
    for k, p in enumerate(parts):
        Xk, yk = S['X_train'][p], S['y_train'][p]
        bst = lgb.train(dict(BASE, seed=seed + k), lgb.Dataset(Xk, yk, weight=w[yk]), num_boost_round=n_trees)
        Pv += a[k] * bst.predict(S['X_val']); Pt += a[k] * bst.predict(S['X_test']); classes_seen.append(int(len(np.unique(yk))))
    out = score(S, Pv, Pt)
    return {'method': 'ens', 'seed': seed, 'n_trees': n_trees, 'history': [dict(round=1, **out)], 'final': out, 'classes_per_client': classes_seen,
            'client_sizes': n_k.astype(int).tolist(), 'minutes': (time.time() - t0) / 60, 'pred_test': Pt.argmax(1).astype(np.uint8)}

def run_cyc(S, parts, seed, cycles=30, iters_per_turn=1, eval_every=3):
    """Cyclic boosting: raw scores are carried as init_score, so each turn continues the shared model exactly."""
    t0 = time.time(); w = class_weights(S['y_train']); K = len(parts); rng = np.random.default_rng(seed)
    raw = [np.zeros((len(p), NC)) for p in parts]; Rv = np.zeros((len(S['y_val']), NC)); Rt = np.zeros((len(S['y_test']), NC))
    hist, n_iter = [], 0
    soft = lambda R: np.exp(R - R.max(1, keepdims=True)) / np.exp(R - R.max(1, keepdims=True)).sum(1, keepdims=True)
    for c in range(1, cycles + 1):
        for k in rng.permutation(K):
            p = parts[k]; Xk, yk = S['X_train'][p], S['y_train'][p]
            bst = lgb.train(dict(BASE, seed=seed * 1000 + n_iter), lgb.Dataset(Xk, yk, weight=w[yk], init_score=raw[k]),
                            num_boost_round=iters_per_turn)
            for j, q in enumerate(parts): raw[j] += bst.predict(S['X_train'][q], raw_score=True)
            Rv += bst.predict(S['X_val'], raw_score=True); Rt += bst.predict(S['X_test'], raw_score=True); n_iter += iters_per_turn
        if c == 1 or c % eval_every == 0 or c == cycles: hist.append(dict(round=c, iterations=n_iter, **score(S, soft(Rv), soft(Rt))))
    return {'method': 'cyc', 'seed': seed, 'cycles': cycles, 'iterations': n_iter, 'history': hist, 'final': hist[-1],
            'client_sizes': [len(p) for p in parts], 'minutes': (time.time() - t0) / 60, 'pred_test': Rt.argmax(1).astype(np.uint8)}
