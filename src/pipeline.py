import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    classification_report, silhouette_score, f1_score
)
from sklearn.utils import resample

CSV_DIR = "data/csv/"
RAW_PATH = "data/raw/dataset.csv"

# -----------------------------------------------------------------------------
# Outlier Elimination (IQR per gruppo geo)
# -----------------------------------------------------------------------------
def remove_outliers_iqr_per_group(df, group_col="geo", value_col="OBS_VALUE"):
    cleaned = []
    for key, grp in df.groupby(group_col):
        q1 = grp[value_col].quantile(0.25)
        q3 = grp[value_col].quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        cleaned.append(grp[(grp[value_col] >= lower) & (grp[value_col] <= upper)])
    return pd.concat(cleaned, ignore_index=True)


# -----------------------------------------------------------------------------
# Bootstrap Functions
# -----------------------------------------------------------------------------
def bootstrap_rmse(model_cls, x_train, y_train, x_test, y_test, n_iter=200):
    rmses = []
    for _ in range(n_iter):
        xb, yb = resample(x_train, y_train, replace=True)
        model = model_cls()
        model.fit(xb, yb)
        ypred = model.predict(x_test)
        mse = mean_squared_error(y_test, ypred)
        rmses.append(np.sqrt(mse))
    return np.array(rmses)


def bootstrap_f1(model_cls, x_train, y_train, x_test, y_test, n_iter=200):
    f1s = []
    for _ in range(n_iter):
        xb, yb = resample(x_train, y_train, replace=True)
        model = model_cls()
        model.fit(xb, yb)
        ypred = model.predict(x_test)
        f1s.append(f1_score(y_test, ypred, average="weighted", zero_division=1))
    return np.array(f1s)


# -----------------------------------------------------------------------------
# 1. Ingestion & Expansion
# -----------------------------------------------------------------------------
def ingest_and_expand(path_csv: str) -> pd.DataFrame:
    df = pd.read_csv(path_csv)
    df = df[(df["indic_bt"] == "COST") & (df["unit"] == "PCH_SM")]

    mapping = {
        "EA19": ["AT","BE","CY","EE","FI","FR","DE","GR","IE","IT","LV","LT","LU","MT","NL","PT","SK","SI","ES"],
        "EA20": ["AT","BE","CY","HR","EE","FI","FR","DE","GR","IE","IT","LV","LT","LU","MT","NL","PT","SK","SI","ES"],
        "EU27_2020": ["AT","BE","BG","HR","CY","CZ","DK","EE","FI","FR","DE","GR","HU","IE","IT","LV","LT","LU","MT","NL","PL","PT","RO","SK","SI","ES","SE"],
    }

    df_agg = df[df["geo"].isin(mapping)]
    df_rest = df[~df["geo"].isin(mapping)]
    existing_keys = set(zip(df_rest["geo"], df_rest["TIME_PERIOD"]))

    expanded = []
    for _, row in df_agg.iterrows():
        code = str(row["geo"])
        for iso in mapping[code]:
            if (iso, row["TIME_PERIOD"]) in existing_keys:
                continue
            new_row = row.copy()
            new_row["geo"] = iso
            expanded.append(new_row)

    df = pd.concat([df_rest, pd.DataFrame(expanded)], ignore_index=True)
    df = df[~df["geo"].isin(["UA","TR","ME","RS","RO"])]
    return df


# -----------------------------------------------------------------------------
# 2. Cleaning & Imputation
# -----------------------------------------------------------------------------
def clean_and_impute(df: pd.DataFrame) -> pd.DataFrame:
    drop_cols = ["DATAFLOW","LAST UPDATE","s_adj","indic_bt","freq","cpa2_1","CONF_STATUS","OBS_FLAG","unit"]
    to_drop = [c for c in drop_cols if c in df.columns]
    df = df.drop(columns=to_drop)

    for col in df.select_dtypes(include="number").columns:
        df[col].fillna(df[col].mean(), inplace=True)
    return df


# -----------------------------------------------------------------------------
# 3. Pivoting & Standardization
# -----------------------------------------------------------------------------
def pivot_and_scale(df: pd.DataFrame):
    pivot = df.pivot_table(index="TIME_PERIOD", columns="geo", values="OBS_VALUE")
    data = pivot.T.dropna(how="all").apply(lambda col: col.fillna(col.mean()), axis=1)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(data)
    return data, scaled


# -----------------------------------------------------------------------------
# 4. Feature Engineering
# -----------------------------------------------------------------------------
def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["geo","TIME_PERIOD"]).reset_index(drop=True)
    df["target_cost"] = df.groupby("geo")["OBS_VALUE"].shift(-1)
    df["prev_cost"] = df.groupby("geo")["OBS_VALUE"].shift(1)
    df["var_perc"] = (df["OBS_VALUE"] - df["prev_cost"]) / df["prev_cost"] * 100
    df["var_perc"].replace([np.inf,-np.inf], 0, inplace=True)

    def label_func(x):
        if x > 20:   return "aumento"
        if x < -20:  return "diminuzione"
        return "stabile"

    df["label_variation"] = df["var_perc"].apply(lambda x: label_func(x) if not np.isnan(x) else None)
    df.dropna(subset=["prev_cost","label_variation"], inplace=True)

    df["rolling_mean_3"] = df.groupby("geo")["OBS_VALUE"].transform(lambda x: x.rolling(3,1).mean())
    df["rolling_std_3"]  = df.groupby("geo")["OBS_VALUE"].transform(lambda x: x.rolling(3,1).std()).fillna(0)
    df["pct_change_3"]   = df.groupby("geo")["OBS_VALUE"].transform(lambda x: x.pct_change(3)).fillna(0)
    df["grew_last_year"] = (df["var_perc"] > 0).astype(int)

    def compute_slope(series):
        vals = series.values[-3:]
        if len(vals) < 3: return 0
        return np.polyfit(range(3), vals, 1)[0]

    df["slope_3"] = df.groupby("geo")["OBS_VALUE"].transform(lambda x: x.expanding().apply(compute_slope, raw=False))
    return df


# -----------------------------------------------------------------------------
# 5. Clustering
# -----------------------------------------------------------------------------
def clustering_kmeans(scaled, n_components=3, k_max=10):
    proj = PCA(n_components=n_components).fit_transform(scaled)
    best_k, best_score = 2, -1
    for k in range(2, min(k_max, proj.shape[0]) + 1):
        km = KMeans(n_clusters=k, random_state=42).fit(proj)
        score = silhouette_score(proj, km.labels_)
        if score > best_score:
            best_k, best_score = k, score
    km_final = KMeans(n_clusters=best_k, random_state=42).fit(proj)
    return km_final.labels_, best_k, best_score


def clustering_dbscan(scaled, eps=4.5, min_samples=4):
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(scaled)
    labels = np.array(db.labels_)
    mask_array = labels != -1
    count_inliers = int(np.count_nonzero(mask_array))
    if count_inliers > 1:
        sc = silhouette_score(scaled[mask_array], labels[mask_array])
    else:
        sc = None
    return labels, sc


# -----------------------------------------------------------------------------
# 6. Modeling & Evaluation
# -----------------------------------------------------------------------------
def train_and_eval_regression(df_feat: pd.DataFrame):
    features = ["OBS_VALUE","pct_change_3","rolling_std_3","grew_last_year"]
    x = df_feat[features]
    y = df_feat["target_cost"]
    train_mask = df_feat["TIME_PERIOD"] <= 2016
    x_train, x_test = x[train_mask], x[~train_mask]
    y_train, y_test = y[train_mask], y[~train_mask]

    dt_model = DecisionTreeRegressor(random_state=42).fit(x_train, y_train)
    dt_pred  = dt_model.predict(x_test)

    rf_model = RandomForestRegressor(random_state=42).fit(x_train, y_train)
    rf_pred  = rf_model.predict(x_test)

    def calc_metrics(y_t, y_p):
        mse = mean_squared_error(y_t, y_p)
        return {
            "rmse": np.sqrt(mse),
            "mae": mean_absolute_error(y_t, y_p),
            "r2": r2_score(y_t, y_p),
        }

    # bootstrap RMSE
    rmse_boot = bootstrap_rmse(lambda: RandomForestRegressor(random_state=42),
                               x_train, y_train, x_test, y_test)

    regression_df = pd.DataFrame({
        "geo": df_feat.loc[~train_mask,"geo"].values,
        "time_period": df_feat.loc[~train_mask,"TIME_PERIOD"].values,
        "obs_value": df_feat.loc[~train_mask,"OBS_VALUE"].values,
        "pred_cost": rf_pred
    })

    return regression_df, {"DT": calc_metrics(y_test, dt_pred),
                           "RF": calc_metrics(y_test, rf_pred)}, rmse_boot


def train_and_eval_classification(df_feat: pd.DataFrame):
    features = ["OBS_VALUE","pct_change_3","rolling_std_3","grew_last_year"]
    x = df_feat[features]
    y = df_feat["label_variation"]
    train_mask = df_feat["TIME_PERIOD"] <= 2016
    x_train, x_test = x[train_mask], x[~train_mask]
    y_train, y_test = y[train_mask], y[~train_mask]

    encoder = LabelEncoder().fit(y_train)
    y_train_enc = encoder.transform(y_train)
    y_test_enc  = encoder.transform(y_test)

    clf = RandomForestClassifier(n_estimators=100, random_state=42).fit(x_train, y_train_enc)
    y_pred = clf.predict(x_test)
    y_proba= clf.predict_proba(x_test)

    class_df = pd.DataFrame({
        "geo": df_feat.loc[~train_mask,"geo"].values,
        "time_period": df_feat.loc[~train_mask,"TIME_PERIOD"].values,
        "true_label": y_test.values,
        "pred_label": encoder.inverse_transform(y_pred)
    })
    for idx, cls in enumerate(encoder.classes_):
        class_df[f"prob_{cls}"] = y_proba[:, idx]

    report = classification_report(y_test_enc, y_pred, output_dict=True, zero_division=1)
    f1_boot = bootstrap_f1(lambda: RandomForestClassifier(random_state=42),
                          x_train, y_train_enc, x_test, y_test_enc)

    return class_df, report, f1_boot


# -----------------------------------------------------------------------------
# 7. Evaluation & Export
# -----------------------------------------------------------------------------
def export_all(df, regression_df, classification_df, clustered_df,
               rmse_boot, f1_boot):
    os.makedirs(CSV_DIR, exist_ok=True)

    df.to_csv(os.path.join(CSV_DIR,"dataset_for_models.csv"), index=False)
    regression_df.to_csv(os.path.join(CSV_DIR,"regression_results.csv"), index=False)
    classification_df.to_csv(os.path.join(CSV_DIR,"classification_results.csv"), index=False)
    clustered_df.to_csv(os.path.join(CSV_DIR,"clustering_results.csv"), index=False)

    # salva grafici bootstrap
    plt.figure(figsize=(6,4))
    plt.hist(rmse_boot, bins=20, edgecolor="black")
    plt.title("Distribuzione RMSE Bootstrap")
    plt.savefig(os.path.join(CSV_DIR,"rmse_bootstrap.png"), dpi=300, bbox_inches="tight")

    plt.figure(figsize=(6,4))
    plt.hist(f1_boot, bins=20, edgecolor="black")
    plt.title("Distribuzione F1-weighted Bootstrap")
    plt.savefig(os.path.join(CSV_DIR,"f1_bootstrap.png"), dpi=300, bbox_inches="tight")

    print("Esportazione completata in", CSV_DIR)


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
def main():
    df0 = ingest_and_expand(RAW_PATH)
    df0 = remove_outliers_iqr_per_group(df0)

    df1 = clean_and_impute(df0)
    data, scaled = pivot_and_scale(df1)
    df2 = feature_engineering(df1)

    labels_k, best_k, best_sil = clustering_kmeans(scaled)
    labels_db, db_sil = clustering_dbscan(scaled)

    clustered_df = df1.copy()
    clustered_df["cluster_kmeans"] = labels_k
    clustered_df["cluster_dbscan"] = labels_db

    regression_df, reg_metrics, rmse_boot = train_and_eval_regression(df2)
    print("Reg metrics:", reg_metrics)

    classification_df, clf_metrics, f1_boot = train_and_eval_classification(df2)
    print("Clf metrics:", clf_metrics)

    export_all(df2, regression_df, classification_df, clustered_df, rmse_boot, f1_boot)
    print("✅ Pipeline completata.")


if __name__ == "__main__":
    main()