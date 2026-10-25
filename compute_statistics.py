"""Recomputes every statistic reported for the federated-trees study from the result files -> one JSON file.
Usage (from the repository root):  python3 compute_statistics.py results benchmark benchmark benchmark statistics.json
  results/   : runs of this study (grid = one-shot ensemble and cyclic boosting, grid_hist = histogram aggregation,
               grid_extra = settings checks and 50-client runs, diag = client class counts and test labels, check = reproduction checks)
  benchmark/ : federated FA-MLP runs on the same clients and seeds (10 and 50 clients), centralized references and the
               split summary, taken from the public fedids-leakfree-benchmark release (https://doi.org/10.5281/zenodo.23483716)
Also recomputes the row-normalized confusion matrices from the saved test predictions, the 50-client sizes and the
modeled transfer times in hours and minutes."""
import json, glob, os, sys, numpy as np, pandas as pd
from scipy import stats
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, matthews_corrcoef
FT, BE, BE3, BP, OUT = sys.argv[1:6]
SEEDS = [42, 123, 456, 789, 1011, 2024, 7, 99, 314, 2718]
SETTINGS = ['random_iid', 'random_dirichlet0.5', 'random_dirichlet0.1', 'random_ip', 'time_dirichlet0.1', 'time_ip']
CLS = ['Benign', 'Brute Force', 'DDOS', 'DOS', 'Mirai', 'Recon', 'Spoofing', 'Web-Based']; NC = 8
F = {}; t975 = stats.t.ppf(.975, 9)
def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); adj = np.empty_like(p); run = 0
    for i, j in enumerate(o): run = max(run, (len(p) - i) * p[j]); adj[j] = min(1, run)
    return adj
def paired(a, b):                                     # b - a in pp
    a, b = np.asarray(a, float), np.asarray(b, float); d = 100 * (b - a); se = d.std(ddof=1) / np.sqrt(len(d))
    return dict(mean=float(d.mean()), sd=float(d.std(ddof=1)), lo=float(d.mean() - t975 * se), hi=float(d.mean() + t975 * se),
                pos=int((d > 0).sum()), neg=int((d < 0).sum()), p=float(stats.ttest_rel(b, a).pvalue))
def put(prefix, dct):
    for k, v in dct.items(): F[f'{prefix}|{k}'] = v
J = lambda f: json.load(open(f))
YT = {sp: np.load(f'{FT}/diag/y_test_{sp}.npy').astype(int) for sp in ('random', 'time')}

# ---------- data ----------
spl = J(f'{BE}/splits_summary.json')
for sp in ('random', 'time'):
    F[f'data|{sp}|n_features'] = spl[sp]['n_features']
    for part, c in spl[sp]['counts'].items():
        F[f'data|{sp}|{part}|total'] = int(sum(c.values()))
        for k, v in c.items(): F[f'data|{sp}|{part}|{k}'] = int(v)
for k, v in spl['split_time_dropped_twins'].items(): F[f'data|time_dropped_twins|{k}'] = int(v)

# ---------- per-run scores ----------
gr = pd.read_csv(f'{BE}/grid_runs.csv')
def fa(tg, m): return np.array([float(gr[(gr.setting == tg) & (gr.method == m) & (gr.seed == s)].last3.iloc[0]) for s in SEEDS])
def tree(tg, m, k='test_macro_f1'):
    d = f'{FT}/grid_hist' if m == 'hist' else f'{FT}/grid'
    return np.array([J(f'{d}/{tg}_{m}_s{s}.json')['final'][k] for s in SEEDS])
LR = J(f'{BE}/fl_method_lr.json')
for m in ('fedavg', 'feddyn', 'fedprox', 'scaffold'): F[f'lr|{m}'] = LR[m]
V = {}
for tg in SETTINGS:
    for m in ('ens', 'cyc', 'hist'): V[tg, m] = tree(tg, m); V[tg, m + '_val'] = tree(tg, m, 'val_macro_f1')
    for m in ('fedavg', 'feddyn', 'fedprox', 'scaffold'): V[tg, m] = fa(tg, m)
    for m in ('ens', 'cyc', 'hist', 'fedavg', 'feddyn', 'fedprox', 'scaffold'):
        F[f'f1|{tg}|{m}|mean'] = float(V[tg, m].mean()); F[f'f1|{tg}|{m}|sd'] = float(V[tg, m].std(ddof=1))
        F[f'f1|{tg}|{m}|min'] = float(V[tg, m].min()); F[f'f1|{tg}|{m}|max'] = float(V[tg, m].max())
    for m in ('ens', 'cyc', 'hist'): F[f'valf1|{tg}|{m}|mean'] = float(V[tg, m + '_val'].mean())
# best FA-MLP rule per setting (mean of the four rules of the benchmark)
for tg in SETTINGS:
    best = max(('fedavg', 'feddyn', 'fedprox', 'scaffold'), key=lambda m: V[tg, m].mean()); F[f'best_famlp|{tg}'] = best
    F[f'best_famlp|{tg}|mean'] = float(V[tg, best].mean())

# ---------- paired comparisons with Holm families ----------
BESTP = [paired(V[tg, F[f'best_famlp|{tg}']], V[tg, 'hist']) for tg in SETTINGS]
for tg, r, h in zip(SETTINGS, BESTP, holm([r['p'] for r in BESTP])): put(f'pair|{tg}|hist-best', r); F[f'pair|{tg}|hist-best|p_holm'] = float(h)
FAM = {'vs_fedavg': [(tg, m, 'fedavg') for tg in SETTINGS for m in ('ens', 'cyc', 'hist')],
       'vs_feddyn': [(tg, m, 'feddyn') for tg in SETTINGS for m in ('ens', 'cyc', 'hist')],
       'hist_vs_tree': [(tg, 'hist', m) for tg in SETTINGS for m in ('ens', 'cyc')],
       'ens_vs_cyc': [(tg, 'ens', 'cyc') for tg in SETTINGS]}
for fam, tests in FAM.items():
    res = [paired(V[tg, ref], V[tg, m]) for tg, m, ref in tests]
    for (tg, m, ref), r, h in zip(tests, res, holm([r['p'] for r in res])):
        put(f'pair|{tg}|{m}-{ref}', r); F[f'pair|{tg}|{m}-{ref}|p_holm'] = float(h); F[f'pair|{tg}|{m}-{ref}|holm_n'] = len(tests)
F['holm_n|tree_vs_famlp'] = len(FAM['vs_fedavg'])

# ---------- centralized references ----------
cm = pd.read_csv(f'{BE3}/central_metrics.csv')
for sp in ('random', 'time'):
    for mo in ('LightGBM', 'XGBoost', 'RandomForest', 'FA-MLP', 'MLP'):
        g = cm[(cm.split == sp) & (cm.model == mo)]
        for k in ('f1_macro', 'accuracy', 'mcc', 'precision_macro', 'recall_macro', 'f1_weighted'):
            F[f'central|{sp}|{mo}|{k}'] = float(g[k].mean())
        if len(g) > 1: F[f'central|{sp}|{mo}|f1_macro_sd'] = float(g.f1_macro.std(ddof=1))
for sp in ('random', 'time'):
    c = J(f'{FT}/grid_hist/check_pooled_edges_{sp}.json'); F[f'bincheck|{sp}|pooled_edges'] = c['final']['test_macro_f1']
    F[f'bincheck|{sp}|minus_lgbm_pp'] = 100 * (c['final']['test_macro_f1'] - F[f'central|{sp}|LightGBM|f1_macro'])
c2 = J(f'{FT}/check/central_lgbm_random_s42.json'); F['check|central_lgbm_repro'] = c2['test_macro_f1']
for tg in SETTINGS:                                   # hist vs centralized LightGBM and spread over partitions
    sp = tg.split('_')[0]; F[f'hist_minus_central|{tg}'] = 100 * (V[tg, 'hist'].mean() - F[f'central|{sp}|LightGBM|f1_macro'])
F['hist|random|range_of_means'] = 100 * (max(V[t, 'hist'].mean() for t in SETTINGS if t.startswith('random')) - min(V[t, 'hist'].mean() for t in SETTINGS if t.startswith('random')))
F['hist|random|max_sd'] = float(max(V[t, 'hist'].std(ddof=1) for t in SETTINGS if t.startswith('random')))
F['hist|time|max_sd'] = float(max(V[t, 'hist'].std(ddof=1) for t in SETTINGS if t.startswith('time')))

# ---------- standard metrics on test predictions (trees) and round-100 metrics (FA-MLP) ----------
def mets(y, p):
    return dict(accuracy=accuracy_score(y, p), precision_macro=precision_score(y, p, average='macro', labels=range(NC), zero_division=0),
                recall_macro=recall_score(y, p, average='macro', labels=range(NC), zero_division=0),
                f1_macro=f1_score(y, p, average='macro', labels=range(NC)), f1_weighted=f1_score(y, p, average='weighted', labels=range(NC)),
                mcc=matthews_corrcoef(y, p))
fr = pd.read_csv(f'{BE3}/fl_final_round_metrics.csv'); fr = fr[fr.source == 'grid']
SHARE = {}
for tg in SETTINGS:
    sp, part = tg.split('_', 1); y = YT[sp]
    for m in ('ens', 'cyc', 'hist'):
        d = f'{FT}/grid_hist' if m == 'hist' else f'{FT}/grid'
        P = [np.load(f'{d}/{tg}_{m}_s{s}_pred.npy').astype(int) for s in SEEDS]
        for k, v in pd.DataFrame([mets(y, p) for p in P]).mean().items(): F[f'std|{tg}|{m}|{k}'] = float(v)
        pcf = np.mean([f1_score(y, p, average=None, labels=range(NC)) for p in P], 0)
        for c, v in zip(CLS, pcf): F[f'pcf1|{tg}|{m}|{c}'] = float(v)
        sh = np.mean([np.bincount(p, minlength=NC) / len(p) for p in P], 0); SHARE[tg, m] = sh
        for c, v in zip(CLS, sh): F[f'predshare|{tg}|{m}|{c}'] = float(v)
    for m in ('fedavg', 'feddyn'):
        g = fr[(fr.split == sp) & (fr.partition == part) & (fr.method == m)]; assert len(g) == 10, (tg, m, len(g))
        for k in ('accuracy', 'precision_macro', 'recall_macro', 'f1_macro', 'f1_weighted', 'mcc'): F[f'std|{tg}|{m}|{k}'] = float(g[k].mean())
        for c in CLS: F[f'pcf1|{tg}|{m}|{c}'] = float(g[f'f1_{c}'].mean())
for sp in ('random', 'time'):
    for c, v in zip(CLS, np.bincount(YT[sp], minlength=NC) / len(YT[sp])): F[f'trueshare|{sp}|{c}'] = float(v)
    F[f'data|{sp}|test_n'] = int(len(YT[sp]))

# ---------- confusion matrices on the time-block split with source-IP clients (row-normalized, summed over seeds) ----------
from sklearn.metrics import confusion_matrix
for tg in ('time_ip', 'random_ip'):
    sp = tg.split('_')[0]
    for m in ('ens', 'cyc', 'hist'):
        d = f'{FT}/grid_hist' if m == 'hist' else f'{FT}/grid'
        cm = sum(confusion_matrix(YT[sp], np.load(f'{d}/{tg}_{m}_s{s}_pred.npy').astype(int), labels=range(NC)) for s in SEEDS)
        rn = cm / cm.sum(1, keepdims=True)
        for i, a_ in enumerate(CLS):
            for j, b_ in enumerate(CLS): F[f'cm|{tg}|{m}|{a_}|{b_}'] = float(rn[i, j])
# ---------- client composition ----------
cc = pd.read_csv(f'{FT}/diag/client_class_counts.csv'); cc['k'] = (cc[CLS] > 0).sum(axis=1); cc['top'] = cc[CLS].max(axis=1) / cc.n
for tg, g in cc.groupby('setting'):
    F[f'clients|{tg}|classes_min'] = int(g.k.min()); F[f'clients|{tg}|classes_median'] = float(g.k.median())
    F[f'clients|{tg}|top_share_median'] = float(g.top.median()); F[f'clients|{tg}|size_min'] = int(g.n.min()); F[f'clients|{tg}|size_max'] = int(g.n.max())
    F[f'clients|{tg}|size_ratio_median'] = float(g.groupby('seed').n.apply(lambda v: v.max() / v.min()).median())
    for c in CLS: F[f'clients|{tg}|holders_{c}'] = float(g.groupby('seed')[c].apply(lambda v: (v > 0).sum()).median())

# ---------- cyclic boosting learning curve ----------
for tg in SETTINGS:
    H = [J(f'{FT}/grid/{tg}_cyc_s{s}.json')['history'] for s in SEEDS]
    for i, h in enumerate(H[0]):
        F[f'cyccurve|{tg}|{h["round"]}'] = float(np.mean([hh[i]['test_macro_f1'] for hh in H]))

# cyclic boosting with validation-based early stopping (choose the evaluated cycle with the best validation macro-F1)
CES = {}
for tg in SETTINGS:
    sel = []
    for s in SEEDS:
        h = J(f'{FT}/grid/{tg}_cyc_s{s}.json')['history']; b = max(h, key=lambda e: e['val_macro_f1']); sel.append((b['test_macro_f1'], b['round']))
    CES[tg] = np.array([x[0] for x in sel]); F[f'cyc_es|{tg}|mean'] = float(CES[tg].mean()); F[f'cyc_es|{tg}|cycle_median'] = float(np.median([x[1] for x in sel]))
    put(f'cyc_es|{tg}|vs_cyc', paired(V[tg, 'cyc'], CES[tg])); put(f'cyc_es|{tg}|vs_fedavg', paired(V[tg, 'fedavg'], CES[tg]))
    put(f'cyc_es|{tg}|hist_minus', paired(CES[tg], V[tg, 'hist']))
# ---------- run cost of the simulation and model sizes ----------
for tg in SETTINGS:
    for m in ('ens', 'cyc', 'hist'):
        d = f'{FT}/grid_hist' if m == 'hist' else f'{FT}/grid'
        F[f'minutes|{tg}|{m}'] = float(np.median([J(f'{d}/{tg}_{m}_s{s}.json')['minutes'] for s in SEEDS]))
HL = [J(f) for f in glob.glob(f'{FT}/grid_hist/*_hist_s*.json')]
F['hist|n_runs'] = len(HL); F['hist|trees'] = int(len(HL[0]['leaves_per_tree'])); F['hist|leaves_mean'] = float(np.mean([np.mean(h['leaves_per_tree']) for h in HL]))
F['hist|bins_sum_mean'] = float(np.mean([sum(h['bins_per_feature']) for h in HL])); F['hist|n_features'] = HL[0]['n_features']
F['hist|bins_sum_min'] = int(min(sum(h['bins_per_feature']) for h in HL)); F['hist|bins_sum_max'] = int(max(sum(h['bins_per_feature']) for h in HL))

# ---------- communication model (analytic; assumptions stated in the paper) ----------
K, R, ITER, PARAMS = 10, 100, 300, 56284
TREES = ITER * NC; LEAVES = 63; INNER = LEAVES - 1
B_TREE = INNER * (4 + 8 + 4 + 4) + LEAVES * 8          # feature id, threshold, two child ids; leaf values (bytes)
B_HBIN = 16                                             # one histogram bin: gradient and hessian sums as 64-bit floats
Q = 257
C = {}
C['fedavg'] = dict(up=K * R * PARAMS * 4, down=K * R * PARAMS * 4, rounds=R)
C['ens'] = dict(up=K * TREES * B_TREE, down=0, rounds=1)
C['cyc'] = dict(up=TREES * B_TREE, down=(K - 1) * TREES * B_TREE, rounds=ITER)          # one sequential turn per iteration
hist_up = K * TREES * (1 + INNER) * F['hist|bins_sum_mean'] * B_HBIN + K * Q * F['hist|n_features'] * 8
C['hist'] = dict(up=hist_up, down=K * TREES * INNER * (4 + 8) + K * F['hist|bins_sum_mean'] * 8, rounds=1 + ITER * (1 + INNER))
for m, c in C.items():
    for k, v in c.items(): F[f'comm|{m}|{k}'] = float(v)
    F[f'comm|{m}|total_GB'] = (c['up'] + c['down']) / 1e9
F['comm|tree_bytes'] = B_TREE; F['comm|model_MB|famlp'] = PARAMS * 4 / 1e6; F['comm|model_MB|gbdt'] = TREES * B_TREE / 1e6
F['comm|model_MB|ens_deployed'] = K * TREES * B_TREE / 1e6
F['comm|hist_over_fedavg'] = F['comm|hist|total_GB'] / F['comm|fedavg|total_GB']
LINKS = {'lan': (1e9, 0.001), 'wan': (100e6, 0.02), 'edge': (10e6, 0.05)}       # (bit/s per client uplink and downlink, RTT s)
for ln, (bw, rtt) in LINKS.items():
    for m, c in C.items():
        t = (c['up'] + c['down']) / K * 8 / bw + c['rounds'] * rtt   # mean traffic per client over its link + sequential round trips
        F[f'linktime|{ln}|{m}|hours'] = t / 3600; F[f'linktime|{ln}|{m}|minutes'] = t / 60
    F[f'link|{ln}|Mbps'] = bw / 1e6; F[f'link|{ln}|rtt_ms'] = rtt * 1000
    F[f'rttonly|{ln}|hist|minutes'] = C['hist']['rounds'] * rtt / 60

# ---------- reviewer checks (Cell 6), if present ----------
GX = f'{FT}/grid_extra'
if os.path.isdir(GX) and glob.glob(f'{GX}/*_s*.json'):
    ck = J(f'{GX}/check_hist_module_random_dirichlet0.5_s42.json')
    F['extra|step0_diff'] = abs(ck['final']['test_macro_f1'] - J(f'{FT}/grid_hist/random_dirichlet0.5_hist_s42.json')['final']['test_macro_f1'])
    EX = []
    for tg in ('random_ip', 'time_ip'):
        for v, ref in (('cyc5', 'cyc'), ('cyc10', 'cyc'), ('ens100', 'ens')):
            fs = [f'{GX}/{tg}_{v}_s{s}.json' for s in SEEDS]
            if not all(os.path.exists(f) for f in fs): continue
            x = np.array([J(f)['final']['test_macro_f1'] for f in fs]); F[f'extra|{tg}|{v}|mean'] = float(x.mean())
            put(f'extra|{tg}|{v}-{ref}', paired(V[tg, ref], x)); put(f'extra|{tg}|{v}-fedavg', paired(V[tg, 'fedavg'], x))
            put(f'extra|{tg}|hist-{v}', paired(x, V[tg, 'hist'])); EX.append((tg, v))
    for v in ('hist', 'cyc', 'ens'):
        fs = [f'{GX}/random_dirichlet0.5_K50_{v}_s{s}.json' for s in SEEDS]
        if not all(os.path.exists(f) for f in fs): continue
        x = np.array([J(f)['final']['test_macro_f1'] for f in fs]); F[f'k50|{v}|mean'] = float(x.mean()); F[f'k50|{v}|sd'] = float(x.std(ddof=1))
        put(f'k50|{v}|vs_k10', paired(V['random_dirichlet0.5', v], x))
    for m in ('fedavg', 'feddyn'):
        L = lambda f, rr: float(np.mean([h['test_macro_f1'] for h in J(f)['history'] if h['round'] in rr]))
        F[f'k50|{m}|r100'] = float(np.mean([L(f'{BP}/random_dirichlet0.5_K50_m10_{m}_s{s}.json', (80, 90, 100)) for s in SEEDS]))
        F[f'k50|{m}|r500'] = float(np.mean([L(f'{BP}/random_dirichlet0.5_K50_m10_R500_{m}_s{s}.json', (480, 490, 500)) for s in SEEDS]))
    F['n_extra_runs'] = len(glob.glob(f'{GX}/*_s*.json')) - 1
    CS = [J(f'{BP}/random_dirichlet0.5_K50_m10_fedavg_s{s}.json')['client_sizes'] for s in SEEDS]
    F['k50|size_min'] = int(min(min(c) for c in CS)); F['k50|size_max'] = int(max(max(c) for c in CS)); F['k50|size_mean'] = float(np.mean([np.mean(c) for c in CS]))
    F['k50|cyc|min'] = float(min(J(f'{GX}/random_dirichlet0.5_K50_cyc_s{s}.json')['final']['test_macro_f1'] for s in SEEDS))
    F['comm|hist_k50_GB'] = F['comm|hist|total_GB'] * 5
F['n_grid_runs'] = len(glob.glob(f'{FT}/grid/*_s*.json')); F['n_hist_runs'] = len(glob.glob(f'{FT}/grid_hist/*_hist_s*.json'))
json.dump(F, open(OUT, 'w'), indent=1); print(len(F), 'facts ->', OUT)
