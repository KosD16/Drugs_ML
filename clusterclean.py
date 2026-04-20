import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from xgboost import XGBClassifier

from scipy.spatial.distance import pdist, squareform
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.utils.validation import check_X_y, check_array

warnings.filterwarnings("ignore")

URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00373/drug_consumption.data"
RANDOM_STATE = 42
CLUSTER_DESIGN_FRACTION = 0.30
K = 4
CLUSTER_THR = 1
N_INIT = 200
LABEL_THR = 3
MIN_COUNT_IN_CLUSTER = 2
EXCLUDE_DRUGS = ["Caffeine", "Choc"]
OUT_DIR = "cluster_outputs"
SAVE_FILES = True
PRINT_TABLES = True

os.makedirs(OUT_DIR, exist_ok=True)

EXPECTED_MODELS = [
    "LogReg", "RandomForest", "ExtraTrees", 
    "HistGradBoost", "MLP", "XGBoost"
]

columns = [
    'ID', 'Age', 'Gender', 'Education', 'Country', 'Ethnicity',
    'Nscore', 'Escore', 'Oscore', 'Ascore', 'Cscore',
    'Impulsive', 'SS',
    'Alcohol', 'Amphet', 'Amyl', 'Benzos', 'Caffeine', 'Cannabis',
    'Choc', 'Coke', 'Crack', 'Ecstasy', 'Heroin', 'Ketamine',
    'Legalh', 'LSD', 'Meth', 'Mushrooms', 'Nicotine', 'Semer', 'VSA'
]
df = pd.read_csv(URL, names=columns)

drug_cols_raw = columns[13:]
mapper = {'CL0': 0, 'CL1': 1, 'CL2': 2, 'CL3': 3, 'CL4': 4, 'CL5': 5, 'CL6': 6}

for c in drug_cols_raw:
    if c in df.columns and df[c].dtype == object:
        df[c] = df[c].map(mapper).fillna(0).astype(int)

if "Semer" in df.columns:
    df = df[df["Semer"] == 0].copy()

df.drop(columns=["Semer"] + EXCLUDE_DRUGS, inplace=True, errors="ignore")
drug_cols_all = [c for c in columns[13:] if c in df.columns]

feature_cols = [
    'Age', 'Gender', 'Education', 'Country', 'Ethnicity',
    'Nscore', 'Escore', 'Oscore', 'Ascore', 'Cscore',
    'Impulsive', 'SS'
]

df_cluster, df_model = train_test_split(
    df,
    test_size=(1 - CLUSTER_DESIGN_FRACTION),
    random_state=RANDOM_STATE,
    shuffle=True
)
X_model = df_model[feature_cols].copy()

# HELPERS
def make_cluster_target(df_, drugs, label_thr, min_count):
    used = (df_[drugs] >= label_thr).astype(int)
    min_count = min(min_count, len(drugs))
    return (used.sum(axis=1) >= min_count).astype(int)

def dist_jaccard_from_threshold(df_, drug_cols, thr=1):
    B = (df_[drug_cols] >= thr).astype(bool).values
    dist_vec = pdist(B.T, metric="jaccard")
    dist_sq = squareform(dist_vec)
    return pd.DataFrame(dist_sq, index=drug_cols, columns=drug_cols)

def balanced_kmedoids(dist_mat, K, n_init=50, max_iter=50, random_state=42):
    rng = np.random.default_rng(random_state)
    D = np.asarray(dist_mat)
    N = D.shape[0]
    if K > N: K = N

    base = N // K
    extra = N % K
    caps = np.array([base + 1] * extra + [base] * (K - extra))
    rng.shuffle(caps)

    def init_medoids_farthest():
        medoids = [int(rng.integers(0, N))]
        for _ in range(1, K):
            dmin = np.min(D[:, medoids], axis=1)
            cand = np.argsort(-dmin)
            for idx in cand:
                if idx not in medoids:
                    medoids.append(int(idx))
                    break
        return np.array(medoids, dtype=int)

    def assign_with_capacity(medoids, caps):
        assigned = -np.ones(N, dtype=int)
        cap_left = caps.copy()
        pairs = []
        for i in range(N):
            for c in range(K):
                pairs.append((D[i, medoids[c]], i, c))
        pairs.sort(key=lambda x: x[0])

        for _, i, c in pairs:
            if assigned[i] != -1: continue
            if cap_left[c] <= 0: continue
            assigned[i] = c
            cap_left[c] -= 1

        if np.any(assigned == -1):
            for i in np.where(assigned == -1)[0]:
                c = np.where(cap_left > 0)[0][0]
                assigned[i] = c
                cap_left[c] -= 1
        return assigned

    def update_medoids(labels):
        medoids = np.zeros(K, dtype=int)
        for c in range(K):
            idx = np.where(labels == c)[0]
            subD = D[np.ix_(idx, idx)]
            medoids[c] = idx[np.argmin(subD.sum(axis=1))]
        return medoids

    def cost(labels, medoids):
        return float(np.sum(D[np.arange(N), medoids[labels]]))

    best_cost = np.inf
    best_labels = None
    best_medoids = None

    for _ in range(n_init):
        medoids = init_medoids_farthest()
        for _it in range(max_iter):
            labels = assign_with_capacity(medoids, caps)
            new_medoids = update_medoids(labels)
            if np.array_equal(new_medoids, medoids): break
            medoids = new_medoids

        cst = cost(labels, medoids)
        if cst < best_cost:
            best_cost, best_labels, best_medoids = cst, labels.copy(), medoids.copy()

    return best_labels, best_medoids, best_cost

def labels_to_cluster_map(items, labels):
    items = list(items)
    cluster_map = {}
    for cid in sorted(set(labels)):
        cluster_map[cid + 1] = [items[i] for i in range(len(items)) if labels[i] == cid]
    return cluster_map

cluster_drug_cols = drug_cols_all[:]
dist = dist_jaccard_from_threshold(df_cluster, cluster_drug_cols, thr=CLUSTER_THR)
labels, medoids, _ = balanced_kmedoids(
    dist.values, K=K, n_init=N_INIT, max_iter=50, random_state=RANDOM_STATE
)
cluster_map = labels_to_cluster_map(cluster_drug_cols, labels)

print("\n CREATED CLUSTERS (Semer users removed; Caffeine/Choc excluded) ")
for cid in sorted(cluster_map):
    print(f"Cluster {cid}: {', '.join(cluster_map[cid])}")

if SAVE_FILES:
    pd.DataFrame(
        [{"cluster_id": cid, "drugs": ", ".join(drugs)} for cid, drugs in cluster_map.items()]
    ).to_csv(os.path.join(OUT_DIR, "clusters.csv"), index=False)

cluster_targets = {}
for cid, drugs in cluster_map.items():
    cluster_targets[cid] = make_cluster_target(
        df_model, drugs, label_thr=LABEL_THR, min_count=MIN_COUNT_IN_CLUSTER
    )

class WeightedXGBClassifier(XGBClassifier):
    def fit(self, X, y, **kwargs):
        n_pos = np.sum(y == 1)
        n_neg = np.sum(y == 0)
        self.scale_pos_weight = n_neg / max(n_pos, 1e-6)
        return super().fit(X, y, **kwargs)

def make_models_for_target(y):
    models = {}

    models["LogReg"] = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=2000, class_weight="balanced",
            solver="liblinear", random_state=RANDOM_STATE
        ))
    ])

    models["RandomForest"] = RandomForestClassifier(
        n_estimators=400, min_samples_leaf=2,
        class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
    )

    models["ExtraTrees"] = ExtraTreesClassifier(
        n_estimators=600, min_samples_leaf=2,
        class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
    )

    models["HistGradBoost"] = HistGradientBoostingClassifier(
        max_depth=6, learning_rate=0.05, max_iter=300,
        random_state=RANDOM_STATE
    )

    models["MLP"] = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            alpha=1e-4,
            learning_rate_init=1e-3,
            max_iter=500,
            early_stopping=True,
            random_state=RANDOM_STATE
        ))
    ])

    models["XGBoost"] = WeightedXGBClassifier(
        n_estimators=600, learning_rate=0.05, max_depth=4,
        subsample=0.85, colsample_bytree=0.85, reg_lambda=1.0,
        eval_metric="logloss",
        random_state=RANDOM_STATE, n_jobs=-1
    )

    return models

scoring = {
    "balanced_acc": "balanced_accuracy",
    "mcc": "matthews_corrcoef",
    "pr_auc": "average_precision",
    "neg_brier": "neg_brier_score"
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

rows = []

for cid, y in cluster_targets.items():
    pos_rate = float(y.mean())
    if pos_rate < 0.01 or pos_rate > 0.99:
        continue

    models = make_models_for_target(y)

    for model_name, est in models.items():
        res = cross_validate(
            est, X_model, y,
            cv=cv, scoring=scoring,
            error_score="raise"
        )
        rows.append({
            "Cluster": cid,
            "Model": model_name,
            "pos_rate": pos_rate,
            "balanced_acc": res["test_balanced_acc"].mean(),
            "mcc": res["test_mcc"].mean(),
            "pr_auc": res["test_pr_auc"].mean(),
            "brier": -res["test_neg_brier"].mean(),
        })

df_results = pd.DataFrame(rows)

if SAVE_FILES and len(df_results) > 0:
    df_results.to_csv(os.path.join(OUT_DIR, "multi_model_results.csv"), index=False)

if len(df_results) > 0:
    figs_dir = os.path.join(OUT_DIR, "figures")
    os.makedirs(figs_dir, exist_ok=True)

    pos = df_results.groupby("Cluster")["pos_rate"].first().sort_index()
    plt.figure()
    plt.bar(pos.index.astype(str), pos.values)
    plt.xlabel("Cluster")
    plt.ylabel("Positive rate (pos_rate)")
    plt.title("Phenotype prevalence by cluster")
    plt.tight_layout()
    plt.savefig(os.path.join(figs_dir, "pos_rate_by_cluster.png"), dpi=200)
    plt.close()

    available_models = [m for m in EXPECTED_MODELS if m in df_results["Model"].unique()]

    ba_piv = (
        df_results.pivot(index="Cluster", columns="Model", values="balanced_acc")
        .reindex(columns=available_models)
        .sort_index()
    )
    ax = ba_piv.plot(kind="bar")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Balanced Accuracy")
    ax.set_title("Balanced Accuracy by Model and Cluster")
    plt.tight_layout()
    plt.savefig(os.path.join(figs_dir, "balanced_accuracy_by_model_cluster.png"), dpi=200)
    plt.close()

    pr_piv = (
        df_results.pivot(index="Cluster", columns="Model", values="pr_auc")
        .reindex(columns=available_models)
        .sort_index()
    )
    ax = pr_piv.plot(kind="bar")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("PR-AUC")
    ax.set_title("PR-AUC by Model and Cluster")
    plt.tight_layout()
    plt.savefig(os.path.join(figs_dir, "pr_auc_by_model_cluster.png"), dpi=200)
    plt.close()

    print(f"\nOptional bar charts saved to: {figs_dir}")

cols_show = ["Cluster", "pos_rate", "balanced_acc", "mcc", "pr_auc", "brier"]

for m in EXPECTED_MODELS:
    if len(df_results) > 0 and (df_results["Model"] == m).any():
        t = df_results[df_results["Model"] == m].sort_values("Cluster")[cols_show]
        if PRINT_TABLES:
            print(f"\n=== {m} ===")
            print(t.to_string(index=False))
        if SAVE_FILES:
            t.to_csv(os.path.join(OUT_DIR, f"table_{m}.csv"), index=False)

if SAVE_FILES:
    xlsx_path = os.path.join(OUT_DIR, "tables_per_model.xlsx")
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        for m in EXPECTED_MODELS:
            if len(df_results) > 0 and (df_results["Model"] == m).any():
                t = df_results[df_results["Model"] == m].sort_values("Cluster")[cols_show]
            else:
                t = pd.DataFrame(columns=cols_show)
            t.to_excel(writer, sheet_name=m[:31], index=False)

shap_dir = os.path.join(OUT_DIR, "shap_lr_per_cluster")
os.makedirs(shap_dir, exist_ok=True)

for cid, y in cluster_targets.items():
    pos_rate = float(y.mean())
    if pos_rate < 0.01 or pos_rate > 0.99:
        continue

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_model.values)

    lr = LogisticRegression(
        max_iter=2000, class_weight="balanced",
        solver="liblinear", random_state=RANDOM_STATE
    )
    lr.fit(Xs, y.values if hasattr(y, "values") else np.asarray(y))

    n_bg = min(200, Xs.shape[0])
    rng = np.random.default_rng(RANDOM_STATE)
    bg_idx = rng.choice(Xs.shape[0], size=n_bg, replace=False)
    background = Xs[bg_idx]

    explainer = shap.LinearExplainer(lr, background)
    sv = explainer.shap_values(Xs)
    if isinstance(sv, list):
        sv = sv[0]
    mean_abs = np.abs(sv).mean(axis=0)

    imp = pd.DataFrame({
        "feature": feature_cols,
        "mean_abs_shap": mean_abs
    }).sort_values("mean_abs_shap", ascending=False)

    imp.to_csv(os.path.join(shap_dir, f"shap_lr_importance_cluster_{cid}.csv"), index=False)

    plt.figure()
    top = imp.head(12).iloc[::-1]
    plt.barh(top["feature"], top["mean_abs_shap"])
    plt.xlabel("mean(|SHAP|)")
    plt.title(f"LR SHAP feature importance (Cluster {cid})")
    plt.tight_layout()
    plt.savefig(os.path.join(shap_dir, f"shap_lr_importance_cluster_{cid}.png"), dpi=200)
    plt.close()

print(f"\nSHAP outputs saved to: {shap_dir}")