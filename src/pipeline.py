import os
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score, classification_report, silhouette_score,
)

CSV_DIR = "data/csv/"
RAW_PATH = "data/raw/dataset.csv"

# -----------------------------------------------------------------------------
# 1. INGESTION & EXPANSION
# -----------------------------------------------------------------------------
def ingest_and_expand(path_csv):
    df = pd.read_csv(path_csv)
    # 1. Filtra COST e PCH_SM
    df = df[(df['indic_bt']=='COST') & (df['unit']=='PCH_SM')]
    # 2. Espandi aggregati (EA19, EA20, EU27_2020)
    mapping = {
      'EA19': ['AT','BE','CY','EE','FI','FR','DE','GR','IE','IT','LV','LT','LU','MT','NL','PT','SK','SI','ES'],
      'EA20': ['AT','BE','CY','HR','EE','FI','FR','DE','GR','IE','IT','LV','LT','LU','MT','NL','PT','SK','SI','ES'],
      'EU27_2020': ['AT','BE','BG','HR','CY','CZ','DK','EE','FI','FR','DE','GR','HU','IE','IT','LV','LT','LU','MT','NL','PL','PT','RO','SK','SI','ES','SE']
    }
    df_agg = df[df['geo'].isin(mapping)]
    df_rest = df[~df['geo'].isin(mapping)]
    existing = set(zip(df_rest['geo'], df_rest['TIME_PERIOD']))
    rows = []
    for _, r in df_agg.iterrows():
        geo = str(r['geo'])
        for iso in mapping[geo]:
            if (iso, r['TIME_PERIOD']) in existing:
                continue
            nr = r.copy()
            nr['geo'] = iso
            rows.append(nr)
    df = pd.concat([df_rest, pd.DataFrame(rows)], ignore_index=True)
    # rimuovi outlier geografici
    df = df[~df['geo'].isin(['UA','TR','ME','RS','RO'])]
    return df

# -----------------------------------------------------------------------------
# 2. CLEANING & IMPUTATION
# -----------------------------------------------------------------------------
def clean_and_impute(df):
    drop_cols = ['DATAFLOW','LAST UPDATE','s_adj','indic_bt','freq',
                 'cpa2_1','CONF_STATUS','OBS_FLAG','unit']
    df = df.drop(columns=[c for c in drop_cols if c in df])
    # imputazione globale
    for c in df.select_dtypes(include='number').columns:
        df[c].fillna(df[c].mean(), inplace=True)
    return df

# -----------------------------------------------------------------------------
# 3. PIVOT & STANDARDIZATION
# -----------------------------------------------------------------------------
def pivot_and_scale(df):
    pivot = df.pivot_table(index='TIME_PERIOD', columns='geo', values='OBS_VALUE')
    data = pivot.T.dropna(how='all').apply(lambda col: col.fillna(col.mean()), axis=1)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(data)
    return data, scaled

# -----------------------------------------------------------------------------
# 4. FEATURE ENGINEERING
# -----------------------------------------------------------------------------
def feature_engineering(df):
    df = df.sort_values(['geo','TIME_PERIOD']).reset_index(drop=True)
    # regressione
    df['target_cost'] = df.groupby('geo')['OBS_VALUE'].shift(-1)
    # classificazione
    df['prev_cost'] = df.groupby('geo')['OBS_VALUE'].shift(1)
    df['var_perc'] = (df['OBS_VALUE'] - df['prev_cost'])/df['prev_cost']*100
    df['var_perc'].replace([np.inf,-np.inf],0, inplace=True)
    def lbl(x):
        return 'aumento' if x>20 else ('diminuzione' if x < -20 else 'stabile')
    df['label_variation'] = df['var_perc'].apply(lambda x: lbl(x) if not np.isnan(x) else None)
    df.dropna(subset=['prev_cost','label_variation'], inplace=True)
    # rolling
    df['rolling_mean_3'] = df.groupby('geo')['OBS_VALUE'].transform(lambda x: x.rolling(3,1).mean())
    df['rolling_std_3']  = df.groupby('geo')['OBS_VALUE'].transform(lambda x: x.rolling(3,1).std()).fillna(0)
    df['pct_change_3']   = df.groupby('geo')['OBS_VALUE'].transform(lambda x: x.pct_change(3)).fillna(0)
    df['grew_last_year'] = (df['var_perc']>0).astype(int)
    # slope 3
    def slope(s):
        y=s.values[-3:]
        if len(y)<3: return 0
        return np.polyfit(range(3),y,1)[0]
    df['slope_3'] = df.groupby('geo')['OBS_VALUE'].transform(lambda x: x.expanding().apply(slope, raw=False))
    return df

# -----------------------------------------------------------------------------
# 5. CLUSTERING
# -----------------------------------------------------------------------------
def clustering_kmeans(scaled, n_components=3, k_max=10):
    pca = PCA(n_components=n_components)
    xp = pca.fit_transform(scaled)
    best_k, best_score = 2, -1
    for k in range(2, min(k_max, xp.shape[0])+1):
        km = KMeans(n_clusters=k, random_state=42).fit(xp)
        sc = silhouette_score(xp, km.labels_)
        if sc>best_score:
            best_k, best_score = k, sc
    km = KMeans(n_clusters=best_k, random_state=42).fit(xp)
    return km.labels_, best_k, best_score

def clustering_dbscan(scaled, eps=4.5, min_samples=4):
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(scaled)
    # Assicuriamoci che labels_ sia array NumPy
    labels = np.array(db.labels_)
    # mask_array è un boolean array
    mask_array = labels != -1
    # contiamo gli inlier (True -> 1)
    count_inliers = int(np.count_nonzero(mask_array))
    if count_inliers > 1:
        sc = silhouette_score(scaled[mask_array], labels[mask_array])
    else:
        sc = None
    return labels, sc

# -----------------------------------------------------------------------------
# 6. MODELING
# -----------------------------------------------------------------------------
def train_and_eval_regression(df_feat):
    # feature / target
    x = df_feat[['OBS_VALUE','pct_change_3','rolling_std_3','grew_last_year']]
    y = df_feat['target_cost']
    train = df_feat['TIME_PERIOD']<=2016
    xtr,xte = x[train],x[~train]
    ytr,yte = y[train],y[~train]
    # DT
    dt = DecisionTreeRegressor(random_state=42).fit(xtr,ytr)
    dtp=dt.predict(xte)
    # RF
    rf = RandomForestRegressor(random_state=42).fit(xtr,ytr)
    rfp=rf.predict(xte)
    # metrics
    def reg_metrics(y,yh):
        return {
            'rmse': mean_squared_error(y,yh,squared=False),
            'mae': mean_absolute_error(y,yh),
            'r2': r2_score(y,yh)
        }
    return {'DT':reg_metrics(yte,dtp),'RF':reg_metrics(yte,rfp)}

def train_and_eval_classification(df_feat):
    # Definiamo X/Y e split
    x = df_feat[['OBS_VALUE','pct_change_3','rolling_std_3','grew_last_year']]
    y = df_feat['label_variation']
    train_mask = df_feat['TIME_PERIOD'] <= 2016
    xtr, xte = x[train_mask], x[~train_mask]
    ytr, yte = y[train_mask], y[~train_mask]
    # Encoding
    le = LabelEncoder()
    ytr_enc = le.fit_transform(ytr)
    yte_enc = le.transform(yte)
    # Allenamento
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(xtr, ytr_enc)
    # Predizioni e probabilità
    y_pred   = clf.predict(xte)
    y_probs  = clf.predict_proba(xte)
    # Costruiamo il DataFrame di output
    classification_output = pd.DataFrame({
        'geo':              df_feat.loc[~train_mask, 'geo'].values,
        'TIME_PERIOD':      df_feat.loc[~train_mask, 'TIME_PERIOD'].values,
        'true_label':       yte.values,
        'predicted_label':  le.inverse_transform(y_pred)
    })
    # Aggiungiamo le colonne di probabilità per ciascuna classe
    for idx, cls in enumerate(le.classes_):
        classification_output[f'prob_{cls}'] = y_probs[:, idx]
    # Report delle metriche
    report = classification_report(yte_enc, y_pred, output_dict=True, zero_division=1)
    return classification_output, report

# -----------------------------------------------------------------------------
# 7. EVALUATION & EXPORT
# -----------------------------------------------------------------------------
def export_all(df, regression_df, classification_output, clustered_df):
    # Crea la cartella se non esiste
    export_path = "data/csv"
    os.makedirs(export_path, exist_ok=True)
    # 1) Dataset pulito con feature e target
    df.to_csv(os.path.join(export_path, "dataset_for_models.csv"), index=False)
    # 2) Risultati regressione
    regression_df[['geo', 'TIME_PERIOD', 'OBS_VALUE', 'pred_cost']].to_csv(
        os.path.join(export_path, "regression_results.csv"), index=False
    )
    # 3) Risultati classificazione con probabilità
    classification_output.to_csv(
        os.path.join(export_path, "classification_results.csv"), index=False
    )
    # 4) Risultati clustering (KMeans o DBSCAN)
    clustered_df.to_csv(
        os.path.join(export_path, "clustering_results.csv"), index=False
    )
    print("Esportazione completata in data/csv/")

# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
def main():
    # 1. Ingestion & Expansion
    df0 = ingest_and_expand(RAW_PATH)
    # 2. Cleaning & Imputation
    df1 = clean_and_impute(df0)
    # 3. Pivot & Scaling (per clustering)
    data, scaled = pivot_and_scale(df1)
    # 4. Feature Engineering (per regressione e classificazione)
    df2 = feature_engineering(df1)
    # 5. Clustering
    labels_k, k, sk = clustering_kmeans(scaled)
    labels_db, sdb = clustering_dbscan(scaled)
    print(f"KMeans: k={k}, silhouette={sk:.3f}")
    print(f"DBSCAN silhouette (inliers)={sdb}")
    # Per comodità, prepariamo clustered_df da esportare:
    clustered_df = df1.copy()
    clustered_df['cluster_kmeans'] = labels_k
    clustered_df['cluster_dbscan'] = labels_db
    # 6. Modeling & Evaluation
    # 6a) Regressione
    #    restituisce un dizionario con metriche e aggiunge 'pred_cost' a regression_df
    regression_df, reg_res = train_and_eval_regression(df2)
    print("Reg metrics:", reg_res)
    # 6b) Classificazione
    #    restituisce classification_output (con colonne geo, TIME_PERIOD, label_variation, prob_*)
    classification_output, clf_res = train_and_eval_classification(df2)
    print("Clf metrics:", clf_res)
    # 7. Esportazione finale
    export_all(df2, regression_df, classification_output, clustered_df)

    print("✅ Pipeline completata. CSV in", CSV_DIR)
if __name__ == "__main__":
    main()