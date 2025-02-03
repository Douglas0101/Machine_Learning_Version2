#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import os

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

###############################################
# 1. TRANSFORMADORES CUSTOMIZADOS
###############################################

class IQROutlierRemover(BaseEstimator, TransformerMixin):
    """
    Transformer que realiza winsorization (clamping) de outliers
    em cada coluna numérica usando a regra do IQR.
    """

    def __init__(self, iqr_multiplier=1.5):
        self.iqr_multiplier = iqr_multiplier

    def fit(self, X, y=None):
        if not isinstance(X, np.ndarray):
            raise ValueError("IQROutlierRemover requer array NumPy como entrada.")
        q1 = np.percentile(X, 25, axis=0)
        q3 = np.percentile(X, 75, axis=0)
        iqr = q3 - q1
        self.lower_ = q1 - self.iqr_multiplier * iqr
        self.upper_ = q3 + self.iqr_multiplier * iqr
        return self

    def transform(self, X):
        X_clamped = X.copy()
        for i in range(X_clamped.shape[1]):
            X_clamped[:, i] = np.clip(X_clamped[:, i], self.lower_[i], self.upper_[i])
        return X_clamped


class DateFeaturesExtractor(BaseEstimator, TransformerMixin):
    """
    Transforma colunas de data (str) em datetime e cria colunas de dia/mes/ano/dia_semana.
    """

    def __init__(self, date_cols):
        self.date_cols = date_cols

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("DateFeaturesExtractor requer pandas DataFrame como entrada.")

        X_out = X.copy()
        for col in self.date_cols:
            if col in X_out.columns:
                # Ajuste para remover warnings de inferência de formato
                # Se você tem certeza que é 'YYYY-MM-DD', use format='%Y-%m-%d'
                X_out[col] = pd.to_datetime(X_out[col], format='%Y-%m-%d', errors='coerce')
                X_out[col + "_ano"] = X_out[col].dt.year
                X_out[col + "_mes"] = X_out[col].dt.month
                X_out[col + "_dia"] = X_out[col].dt.day
                X_out[col + "_dia_semana"] = X_out[col].dt.dayofweek
        return X_out

###############################################
# 2. FUNÇÕES DE PRÉ-PROCESSAMENTO ESPECÍFICAS
###############################################

def preprocess_hospital_data(file_path):
    df = pd.read_csv(file_path, sep=';', na_values=["", "NaN", "None"])

    # Converter datas no DataFrame principal (sem 'errors=ignore')
    for col in ["Data_Admissao", "Data_Alta", "Data_Nascimento"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], format='%Y-%m-%d', errors='coerce')

    # Criar Dias_Internacao se não existir
    if "Dias_Internacao" not in df.columns:
        df["Dias_Internacao"] = (df["Data_Alta"] - df["Data_Admissao"]).dt.days

    # Extrair features de data (Data_Admissao, Data_Alta)
    date_cols = ["Data_Admissao", "Data_Alta"]
    date_pipeline = Pipeline([
        ("date_features", DateFeaturesExtractor(date_cols))
    ])
    df = date_pipeline.fit_transform(df)

    numeric_cols = ["Quantidade_Exames", "Custo_Total", "Dias_Internacao"]
    cat_cols = ["Genero", "Diagnostico", "Procedimento"]

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("outlier", IQROutlierRemover(iqr_multiplier=1.5)),
        ("scaler", StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        # Em versões 1.2+ do sklearn, use sparse_output=False
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, numeric_cols),
        ("cat", cat_pipeline, cat_cols)
    ], remainder='passthrough')

    df_out = preprocessor.fit_transform(df)

    # Reconstruir DataFrame final
    num_col_names = numeric_cols
    cat_enc = preprocessor.named_transformers_["cat"].named_steps["encoder"]
    cat_new_cols = cat_enc.get_feature_names_out(cat_cols)

    pass_cols = list(df.columns.difference(numeric_cols + cat_cols))
    final_col_names = list(num_col_names) + list(cat_new_cols) + pass_cols
    df_final = pd.DataFrame(df_out, columns=final_col_names)

    # Converter colunas datetime passadas por remainder
    for c in df_final.columns:
        if "Data_" in c and ("mes" not in c and "ano" not in c and "dia" not in c and "dia_semana" not in c):
            df_final[c] = pd.to_datetime(df_final[c], format='%Y-%m-%d', errors='coerce')

    return df_final.reset_index(drop=True)


def preprocess_bancario_data(file_path):
    df = pd.read_csv(file_path, sep=';', na_values=["", "NaN", "None"])

    # Converter datas
    for col in ["Data_Abertura", "Data_Ultimo_Movimento"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], format='%Y-%m-%d', errors='coerce')

    # Criar Anos_Conta
    today = pd.to_datetime("today")
    df["Anos_Conta"] = (today - df["Data_Abertura"]).dt.days / 365.0

    numeric_cols = ["Saldo_Atual", "Limite_Credito", "Score_Credito", "Renda_Mensal", "Anos_Conta"]
    cat_cols = ["Tipo_Conta", "Status_Conta"]

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("outlier", IQROutlierRemover(iqr_multiplier=1.5)),
        ("scaler", StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, numeric_cols),
        ("cat", cat_pipeline, cat_cols)
    ], remainder='passthrough')

    df_out = preprocessor.fit_transform(df)

    num_col_names = numeric_cols
    cat_enc = preprocessor.named_transformers_["cat"].named_steps["encoder"]
    cat_new_cols = cat_enc.get_feature_names_out(cat_cols)
    pass_cols = list(df.columns.difference(numeric_cols + cat_cols))

    final_col_names = list(num_col_names) + list(cat_new_cols) + pass_cols
    df_final = pd.DataFrame(df_out, columns=final_col_names)

    # Ajustar datas que passaram no remainder
    if "Data_Abertura" in df_final.columns:
        df_final["Data_Abertura"] = pd.to_datetime(df_final["Data_Abertura"], format='%Y-%m-%d', errors='coerce')
    if "Data_Ultimo_Movimento" in df_final.columns:
        df_final["Data_Ultimo_Movimento"] = pd.to_datetime(df_final["Data_Ultimo_Movimento"], format='%Y-%m-%d', errors='coerce')

    return df_final.reset_index(drop=True)


def preprocess_nuclear_data(file_path):
    df = pd.read_csv(file_path, sep=';', na_values=["", "NaN", "None"])

    if "Data_Inspecao" in df.columns:
        df["Data_Inspecao"] = pd.to_datetime(df["Data_Inspecao"], format='%Y-%m-%d', errors='coerce')

    # Criar ano e mes
    df["Ano_Inspecao"] = df["Data_Inspecao"].dt.year
    df["Mes_Inspecao"] = df["Data_Inspecao"].dt.month

    numeric_cols = [
        "Nivel_Radiacao", "Temperatura_Reator", "Pressao_Reator",
        "Eficiência_Operacional", "Falhas_Detectadas", "Custo_Manutencao",
        "Potencia_Gerada", "Ano_Inspecao", "Mes_Inspecao"
    ]
    cat_cols = ["Status_Operacional", "Nome_Planta", "Localizacao"]

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("outlier", IQROutlierRemover(iqr_multiplier=1.5)),
        ("scaler", StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, numeric_cols),
        ("cat", cat_pipeline, cat_cols)
    ], remainder='passthrough')

    df_out = preprocessor.fit_transform(df)

    num_col_names = numeric_cols
    cat_enc = preprocessor.named_transformers_["cat"].named_steps["encoder"]
    cat_new_cols = cat_enc.get_feature_names_out(cat_cols)
    pass_cols = list(df.columns.difference(numeric_cols + cat_cols))

    final_col_names = list(num_col_names) + list(cat_new_cols) + pass_cols
    df_final = pd.DataFrame(df_out, columns=final_col_names)

    # Ajustar data no final
    if "Data_Inspecao" in df_final.columns:
        df_final["Data_Inspecao"] = pd.to_datetime(df_final["Data_Inspecao"], format='%Y-%m-%d', errors='coerce')

    return df_final.reset_index(drop=True)


###############################################
# 3. MAIN (Exemplo de uso)
###############################################
if __name__ == "__main__":
    base_dir = "../data"
    hosp_csv = os.path.join(base_dir, "dataset_hospitalar_avancado.csv")
    banco_csv = os.path.join(base_dir, "dataset_bancario_avancado.csv")
    nuc_csv = os.path.join(base_dir, "dataset_energia_nuclear_avancado.csv")

    df_hosp_pre = preprocess_hospital_data(hosp_csv)
    print("Hospital DataFrame shape:", df_hosp_pre.shape)
    print(df_hosp_pre.head())

    df_banco_pre = preprocess_bancario_data(banco_csv)
    print("Bancario DataFrame shape:", df_banco_pre.shape)
    print(df_banco_pre.head())

    df_nuc_pre = preprocess_nuclear_data(nuc_csv)
    print("Nuclear DataFrame shape:", df_nuc_pre.shape)
    print(df_nuc_pre.head())

    # Salvar CSV final
    df_hosp_pre.to_csv("hospital_preprocessed.csv", index=False, sep=";")
    df_banco_pre.to_csv("bancario_preprocessed.csv", index=False, sep=";")
    df_nuc_pre.to_csv("nuclear_preprocessed.csv", index=False, sep=";")
    print("Pré-processamento concluído com sucesso.")
