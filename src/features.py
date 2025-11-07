
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
features.py
-----------
Pipeline utilitario para:
- Conversión de COD_GRADO -> texto usando COD_ENSE en string (Básica / Media)
- Derivar EDAD = AGNO - año(FEC_NAC_ALU)
- Balanceo del dataset: mantener todos los Retirado;Retirado;1 y muestrear el mismo número del resto (sin reemplazo, semilla fija)
- Cálculo de correlaciones: Pearson (numéricas) y Chi² + Phi/Cramér's V (categóricas vs variable binaria)
- Construcción de dataset de entrenamiento con columnas solicitadas + target Deserción
- Target Mean Encoding (OOF, KFold) para todas las categóricas seleccionadas
- Feature importance con RandomForest + Permutation Importance
- Eliminación de columnas en memoria (sin sobrescribir a menos que se pida)

Uso rápido (ejemplos):
    python features.py convert-grado --in salida_con_grado_texto.csv --out salida_con_grado_texto.csv
    python features.py correlations --in salida_con_grado_texto.csv
    python features.py balance --in salida_con_grado_texto.csv --out dataset_balanceado_retirados.csv --seed 42
    python features.py build-dataset --in salida_con_grado_texto.csv --out-csv dataset_entrenamiento_encoded.csv --out-map tme_mappings.json
    python features.py feature-importance --data dataset_entrenamiento_encoded.csv --out-csv feature_importance_permutation.csv --out-png feature_importance_top20.png
    python features.py dropcols --in dataset_balanceado_retirados.csv --cols AGNO PROM_GRAL ASISTENCIA --save
"""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, pearsonr

# Modelado
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.inspection import permutation_importance


# -----------------------------
# Utilidades base
# -----------------------------

def load_csv(path: str, sep: str = ';', dtype='str') -> pd.DataFrame:
    return pd.read_csv(path, sep=sep, dtype=dtype)

def save_csv(df: pd.DataFrame, path: str, sep: str = ';'):
    df.to_csv(path, sep=sep, index=False)

def ensure_binary_desercion(df: pd.DataFrame, col: str = 'Desercion') -> pd.DataFrame:
    if col not in df.columns:
        raise ValueError(f"No se encontró la columna '{col}'.")
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
    return df

def extraer_ano(fecha: str):
    """Extrae año desde 'AAAA', 'AAAAMM' o 'AAAAMMDD'."""
    if pd.isna(fecha):
        return np.nan
    s = str(fecha).strip()
    return int(s[:4]) if len(s) >= 4 and s[:4].isdigit() else np.nan

def derive_age(df: pd.DataFrame, agno_col='AGNO', fec_col='FEC_NAC_ALU', out_col='EDAD') -> pd.DataFrame:
    if {agno_col, fec_col}.issubset(df.columns):
        ano = pd.to_numeric(df[agno_col], errors='coerce')
        ano_nac = df[fec_col].apply(extraer_ano).astype(float)
        df[out_col] = (ano - ano_nac).where(lambda x: (x >= 0) & (x <= 120))
    return df

def convert_cod_grado_text(df: pd.DataFrame, ense_col='COD_ENSE', grado_col='COD_GRADO') -> pd.DataFrame:
    if {ense_col, grado_col}.issubset(df.columns):
        ce = df[ense_col].astype(str)
        cg = df[grado_col].astype(str)

        mapa_basica = {str(i): f"{i}º Básico" for i in range(1, 9)}
        mapa_media  = {str(i): f"{i}º Medio"  for i in range(1, 5)}

        is_basica = ce.str.contains('Básica', case=False, na=False)
        is_media  = ce.str.contains('Media',  case=False, na=False)

        nuevo = cg.copy()
        nuevo = nuevo.mask(is_basica & nuevo.isin(mapa_basica), nuevo.map(mapa_basica))
        nuevo = nuevo.mask(is_media  & nuevo.isin(mapa_media),  nuevo.map(mapa_media))
        df[grado_col] = nuevo.fillna(df[grado_col])
    return df

def drop_columns_inplace(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    return df.drop(columns=cols, errors='ignore')


# -----------------------------
# Correlaciones
# -----------------------------

def chi2_phi_cramers(serie_cat: pd.Series, serie_bin: pd.Series) -> Tuple[float, float, int, float, str]:
    tabla = pd.crosstab(serie_cat, serie_bin)
    if tabla.size == 0 or tabla.shape[1] != 2:
        return math.nan, math.nan, int(tabla.values.sum()), math.nan, 'na'
    chi2, p, dof, _ = chi2_contingency(tabla, correction=False)
    n = tabla.values.sum()
    r, c = tabla.shape
    if r == 2 and c == 2:
        phi = math.sqrt(chi2 / n) if n > 0 else math.nan
        return chi2, p, int(n), phi, 'phi'
    denom = n * (min(r - 1, c - 1))
    v = math.sqrt(chi2 / denom) if denom > 0 else math.nan
    return chi2, p, int(n), v, 'cramers_v'

def compute_correlations(df: pd.DataFrame, target='Desercion', out_num='cor_num_pearson.csv', out_cat='cor_cat_cramers_phi.csv'):
    df = ensure_binary_desercion(df, target)

    # numéricas
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    if target in num_cols:
        num_cols.remove(target)
    res_num = []
    for col in num_cols:
        x = df[col]
        mask = ~x.isna()
        if mask.sum() < 3:
            continue
        r, p = pearsonr(df.loc[mask, target], x[mask])
        res_num.append({'columna': col, 'pearson_r': r, 'p_value': p, 'n': int(mask.sum())})
    num_df = pd.DataFrame(res_num).sort_values('pearson_r', ascending=False)
    save_csv(num_df, out_num)

    # categóricas
    cat_cols = [c for c in df.columns if c not in num_cols + [target]]
    res_cat = []
    for col in cat_cols:
        s = df[col].astype('string')
        chi2, p, n, medida, tipo = chi2_phi_cramers(s, df[target])
        res_cat.append({'columna': col, 'chi2': chi2, 'p_value': p, 'n': n, 'medida': medida, 'tipo': tipo})
    cat_df = pd.DataFrame(res_cat).sort_values(['medida', 'chi2'], ascending=[False, False])
    save_csv(cat_df, out_cat)
    return num_df, cat_df


# -----------------------------
# Balanceo Retirados
# -----------------------------

def balance_retirados(df: pd.DataFrame, seed=42) -> pd.DataFrame:
    df = ensure_binary_desercion(df, 'Desercion')
    pos = df[(df['SIT_FIN'] == 'Retirado') & (df['SIT_FIN_R'] == 'Retirado') & (df['Desercion'] == 1)]
    n_pos = len(pos)
    neg = df.drop(pos.index)
    neg_sample = neg.sample(n=n_pos, replace=False, random_state=seed)
    out = pd.concat([pos, neg_sample]).sample(frac=1, random_state=seed).reset_index(drop=True)
    return out


# -----------------------------
# Target Mean Encoding (OOF)
# -----------------------------

def tme_oof(df: pd.DataFrame, target_col: str, cat_cols: List[str], n_splits=5, seed=42) -> Tuple[pd.DataFrame, Dict]:
    df = df.copy()
    y = pd.to_numeric(df[target_col], errors='coerce').fillna(0).astype(int)
    global_mean = float(y.mean())
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    mappings = {}
    for col in cat_cols:
        df[f'{col}_TME'] = np.nan

    for tr_idx, va_idx in kf.split(df):
        tr = df.iloc[tr_idx]
        va = df.iloc[va_idx]
        y_tr = y.iloc[tr_idx]
        means = tr.groupby(col)[target_col].mean() if len(cat_cols)==1 else None
        # Hacemos loop para cat_cols (correcto)
        for col in cat_cols:
            means = tr.groupby(col)[target_col].mean()
            df.iloc[va_idx, df.columns.get_loc(f'{col}_TME')] = va[col].map(means)

    for col in cat_cols:
        df[f'{col}_TME'] = df[f'{col}_TME'].astype(float).fillna(global_mean)
        full_means = df.groupby(col)[target_col].mean()
        mappings[col] = {
            'global_mean': global_mean,
            'per_category': {str(k): float(v) for k, v in full_means.items()}
        }

    encoded = df.drop(columns=cat_cols, errors='ignore')
    return encoded, mappings


# -----------------------------
# Construcción dataset de entrenamiento
# -----------------------------

ORDER_FINAL = [
    'ASISTENCIA',
    'COD_ENSE',
    'EDAD',          # reemplazo de EDAD_ALU
    'NOM_RBD',
    'COD_JOR',
    'COD_GRADO',
    'NOM_COM_RBD',
    'NOM_COM_ALU',
    'NOM_DEPROV_RBD',
    'NOM_REG_RBD_A',
    'PROM_GRAL',
    'COD_DEPE',
    'Desercion'
]

def build_training_dataset(df: pd.DataFrame,
                           out_csv='dataset_entrenamiento_encoded.csv',
                           out_map='tme_mappings.json',
                           n_splits=5, seed=42) -> Tuple[pd.DataFrame, Dict]:
    # asegurar EDAD y conversion de grado (por si acaso)
    df = derive_age(df)
    df = convert_cod_grado_text(df)

    # seleccionar columnas en orden si existen
    exists = [c for c in ORDER_FINAL if c in df.columns]
    df = df[exists].copy()

    # tipificar numéricas conocidas
    for c in ['ASISTENCIA', 'PROM_GRAL', 'EDAD', 'Desercion']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    df = ensure_binary_desercion(df, 'Desercion')

    numeric_set = {'ASISTENCIA', 'PROM_GRAL', 'EDAD', 'Desercion'}
    cat_cols = [c for c in df.columns if c not in numeric_set and c != 'Desercion']

    encoded, mappings = tme_oof(df, 'Desercion', cat_cols, n_splits=n_splits, seed=seed)

    # reordenar: mantener ASISTENCIA, PROM_GRAL, EDAD y las *_TME siguiendo ORDER_FINAL
    order_encoded = []
    for c in ORDER_FINAL:
        if c == 'Desercion':
            continue
        if c in ['ASISTENCIA', 'PROM_GRAL', 'EDAD'] and c in encoded.columns:
            order_encoded.append(c)
        else:
            t = f'{c}_TME'
            if t in encoded.columns:
                order_encoded.append(t)
    order_encoded.append('Desercion')

    encoded = encoded[[col for col in order_encoded if col in encoded.columns]]
    save_csv(encoded, out_csv)
    Path(out_map).write_text(json.dumps(mappings, ensure_ascii=False, indent=2), encoding='utf-8')
    return encoded, mappings


# -----------------------------
# Feature Importance con RF + Permutation
# -----------------------------

def feature_importance(encoded_csv: str,
                       out_csv='feature_importance_permutation.csv',
                       out_png='feature_importance_top20.png',
                       test_size=0.2, seed=42):
    import matplotlib.pyplot as plt

    df = load_csv(encoded_csv, sep=';')
    if 'Desercion' not in df.columns:
        raise ValueError("El dataset encoded debe contener la columna 'Desercion'.")

    y = pd.to_numeric(df['Desercion'], errors='coerce').fillna(0).astype(int)
    X = df.drop(columns=['Desercion'])
    # rellenar posibles NaN
    X = X.apply(pd.to_numeric, errors='coerce').fillna(0)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=300, max_depth=None, n_jobs=-1,
        random_state=seed, class_weight='balanced_subsample'
    )
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    proba = clf.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, pred)
    f1  = f1_score(y_test, pred, zero_division=0)
    auc = roc_auc_score(y_test, proba)

    print(f"Accuracy: {acc:.4f} | F1: {f1:.4f} | ROC-AUC: {auc:.4f}")

    r = permutation_importance(clf, X_test, y_test, n_repeats=5, random_state=seed, n_jobs=-1)
    imp = pd.DataFrame({
        'feature': X_test.columns,
        'importance_mean': r.importances_mean,
        'importance_std': r.importances_std
    }).sort_values('importance_mean', ascending=False)
    save_csv(imp, out_csv)

    # plot top 20
    top = imp.head(20)
    plt.figure(figsize=(10, 6))
    plt.barh(top['feature'][::-1], top['importance_mean'][::-1])
    plt.title('Permutation Importance (Top 20)')
    plt.xlabel('Mean Importance (Decrease in Score)')
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    print(f"Guardados: {out_csv} y {out_png}")


# -----------------------------
# CLI
# -----------------------------

def main():
    parser = argparse.ArgumentParser(description='Pipeline de features para Desercion')
    sub = parser.add_subparsers(dest='cmd', required=True)

    # Convertir grado
    p1 = sub.add_parser('convert-grado', help='Convertir COD_GRADO a texto usando COD_ENSE textual')
    p1.add_argument('--in', dest='inp', required=True)
    p1.add_argument('--out', dest='out', required=True)
    p1.add_argument('--sep', default=';')

    # Correlaciones
    p2 = sub.add_parser('correlations', help='Calcular correlaciones (Pearson para numéricas; Cramér/Phi para categóricas)')
    p2.add_argument('--in', dest='inp', required=True)
    p2.add_argument('--sep', default=';')
    p2.add_argument('--out-num', default='cor_num_pearson.csv')
    p2.add_argument('--out-cat', default='cor_cat_cramers_phi.csv')

    # Balance retirados
    p3 = sub.add_parser('balance', help='Balancear 1:1 manteniendo todos Retirado;Retirado;1 y muestreando el resto sin reemplazo')
    p3.add_argument('--in', dest='inp', required=True)
    p3.add_argument('--out', dest='out', required=True)
    p3.add_argument('--sep', default=';')
    p3.add_argument('--seed', type=int, default=42)

    # Build dataset entrenamiento (Target Encoding OOF)
    p4 = sub.add_parser('build-dataset', help='Construir dataset encoded (target mean OOF) con columnas solicitadas')
    p4.add_argument('--in', dest='inp', required=True)
    p4.add_argument('--sep', default=';')
    p4.add_argument('--out-csv', default='dataset_entrenamiento_encoded.csv')
    p4.add_argument('--out-map', default='tme_mappings.json')
    p4.add_argument('--splits', type=int, default=5)
    p4.add_argument('--seed', type=int, default=42)

    # Feature importance
    p5 = sub.add_parser('feature-importance', help='Calcular Permutation Importance sobre dataset encoded')
    p5.add_argument('--data', required=True)
    p5.add_argument('--out-csv', default='feature_importance_permutation.csv')
    p5.add_argument('--out-png', default='feature_importance_top20.png')
    p5.add_argument('--seed', type=int, default=42)
    p5.add_argument('--test-size', type=float, default=0.2)

    # Drop columnas en memoria (y opcionalmente guardar)
    p6 = sub.add_parser('dropcols', help='Eliminar columnas del CSV (opcionalmente guardar)')
    p6.add_argument('--in', dest='inp', required=True)
    p6.add_argument('--sep', default=';')
    p6.add_argument('--cols', nargs='+', required=True, help='Lista de columnas a eliminar')
    p6.add_argument('--save', action='store_true', help='Si se pasa, guarda los cambios sobre el mismo archivo')

    args = parser.parse_args()

    if args.cmd == 'convert-grado':
        df = load_csv(args.inp, sep=args.sep)
        df = convert_cod_grado_text(df)
        save_csv(df, args.out, sep=args.sep)
        print(f"OK → Convertido y guardado en {args.out}")

    elif args.cmd == 'correlations':
        df = load_csv(args.inp, sep=args.sep)
        num_df, cat_df = compute_correlations(df, out_num=args.out_num, out_cat=args.out_cat)
        print(f"OK → Guardados {args.out_num} y {args.out_cat}")

    elif args.cmd == 'balance':
        df = load_csv(args.inp, sep=args.sep)
        df_bal = balance_retirados(df, seed=args.seed)
        save_csv(df_bal, args.out, sep=args.sep)
        print(f"OK → Balanceado y guardado en {args.out}. Filas: {len(df_bal)}")

    elif args.cmd == 'build-dataset':
        df = load_csv(args.inp, sep=args.sep)
        encoded, mappings = build_training_dataset(df, out_csv=args.out_csv, out_map=args.out_map,
                                                   n_splits=args.splits, seed=args.seed)
        print(f"OK → Dataset encoded en {args.out_csv}; mapeos en {args.out_map}. Filas: {len(encoded)} Cols: {encoded.shape[1]}")

    elif args.cmd == 'feature-importance':
        feature_importance(args.data, out_csv=args.out_csv, out_png=args.out_png, test_size=args.test_size, seed=args.seed)

    elif args.cmd == 'dropcols':
        df = load_csv(args.inp, sep=args.sep)
        df = drop_columns_inplace(df, args.cols)
        if args.save:
            save_csv(df, args.inp, sep=args.sep)
            print(f"OK → Columnas {args.cols} eliminadas y guardado sobre {args.inp}")
        else:
            print(f"OK → Columnas {args.cols} eliminadas en memoria (no se guardó).")
            print("Columnas actuales:", list(df.columns))

if __name__ == '__main__':
    main()
