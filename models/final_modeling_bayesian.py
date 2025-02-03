#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script final de modelagem bayesiana (exemplos).
Aplicações:
1) Leitura dos dados pré-processados (hospital, bancário, nuclear)
2) Seleção apenas de colunas numéricas (removendo colunas textuais/IDs)
3) Imputação para eliminar NaN (SimpleImputer com strategy="mean")
4) Split Treino/Teste
5) Treinamento com modelos bayesianos:
   - Regressão: BayesianRidge
   - Classificação: GaussianNB (usando a coluna 'Status_Conta_Em_Atraso' como target)
6) Exemplo opcional com PyMC, com extração dos parâmetros usando np.moveaxis para evitar conflitos de shape.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score, r2_score
from sklearn.linear_model import BayesianRidge
from sklearn.naive_bayes import GaussianNB
from sklearn.impute import SimpleImputer


######################################
# 1. MODELO PARA DATASET HOSPITALAR
######################################
def model_hospital():
    """
    Regressão Bayesiana para prever 'Custo_Total'.
    Após carregar o CSV pré-processado, seleciona-se somente as colunas numéricas,
    eliminando quaisquer colunas textuais (por exemplo, IDs, nomes, CPF, etc.).
    Em seguida, aplica-se imputação para substituir NaN pela média.
    """
    print("\n=== MODELAGEM HOSPITALAR ===")
    # Carregar o dataset pré-processado
    df_hosp = pd.read_csv("../data/hospital_preprocessed.csv", sep=";")

    # Selecionar somente as colunas numéricas (descarta automaticamente colunas textuais)
    df_hosp_numeric = df_hosp.select_dtypes(include=[np.number])

    target_col = "Custo_Total"
    if target_col not in df_hosp_numeric.columns:
        print(f"[AVISO] Coluna {target_col} não encontrada no dataset numérico.")
        return

    X_cols = df_hosp_numeric.columns.difference([target_col])
    X = df_hosp_numeric[X_cols].values
    y = df_hosp_numeric[target_col].values

    # Imputação para garantir que X e y não contenham NaN
    imputer = SimpleImputer(strategy="mean")
    X = imputer.fit_transform(X)
    y = np.where(pd.isna(y), np.nanmean(y), y)

    # Split Treino/Teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Treinamento com BayesianRidge
    model = BayesianRidge()
    model.fit(X_train, y_train)

    # Previsão e avaliação
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print(f"BayesianRidge (Hospital) -> RMSE: {rmse:.4f}, R²: {r2:.4f}")
    print("Intercept:", model.intercept_)


######################################
# 2. MODELO PARA DATASET BANCÁRIO
######################################
def model_bancario():
    """
    Classificação Bayesiana para prever se a conta está em atraso.
    Como o dataset bancário não possui a coluna 'Status_Conta', utiliza-se
    'Status_Conta_Em_Atraso' como target.
    Apenas as colunas numéricas são selecionadas e é aplicada imputação para eliminar NaN.
    """
    print("\n=== MODELAGEM BANCÁRIA ===")
    df_banco = pd.read_csv("../data/bancario_preprocessed.csv", sep=";")

    target_col = "Status_Conta_Em_Atraso"
    if target_col not in df_banco.columns:
        print(f"[AVISO] Coluna {target_col} não encontrada em bancario_preprocessed.csv.")
        return

    # Selecionar somente as colunas numéricas
    df_banco_numeric = df_banco.select_dtypes(include=[np.number])

    if target_col not in df_banco_numeric.columns:
        # Se o target não estiver numérico, tente convertê-lo
        try:
            df_banco[target_col] = pd.to_numeric(df_banco[target_col], errors='coerce')
            df_banco_numeric = df_banco.select_dtypes(include=[np.number])
        except Exception as e:
            print(f"[Erro] Não foi possível converter {target_col}: {e}")
            return

    X_cols = df_banco_numeric.columns.difference([target_col])
    X = df_banco_numeric[X_cols].values
    y = df_banco_numeric[target_col].values

    # Imputação para X e y
    imputer = SimpleImputer(strategy="mean")
    X = imputer.fit_transform(X)
    from scipy.stats import mode
    if np.any(pd.isna(y)):
        y_mode = mode(y, nan_policy='omit').mode[0]
        y = np.where(pd.isna(y), y_mode, y)

    # Split Treino/Teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Treinamento com GaussianNB
    model_nb = GaussianNB()
    model_nb.fit(X_train, y_train)

    y_pred = model_nb.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"GaussianNB (Bancário) -> Acurácia: {acc:.4f}")


######################################
# 3. MODELO PARA DATASET NUCLEAR
######################################
def model_nuclear():
    """
    Regressão Bayesiana para prever 'Potencia_Gerada' usando BayesianRidge.
    Apenas as colunas numéricas são usadas (descartando colunas textuais).
    É aplicada imputação para eliminar NaN.
    """
    print("\n=== MODELAGEM NUCLEAR ===")
    df_nuc = pd.read_csv("../data/nuclear_preprocessed.csv", sep=";")

    df_nuc_numeric = df_nuc.select_dtypes(include=[np.number])

    target_col = "Potencia_Gerada"
    if target_col not in df_nuc_numeric.columns:
        print(f"[AVISO] Coluna {target_col} não encontrada no dataset numérico.")
        return

    X_cols = df_nuc_numeric.columns.difference([target_col])
    X = df_nuc_numeric[X_cols].values
    y = df_nuc_numeric[target_col].values

    imputer = SimpleImputer(strategy="mean")
    X = imputer.fit_transform(X)
    y = np.where(pd.isna(y), np.nanmean(y), y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = BayesianRidge()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print(f"BayesianRidge (Nuclear) -> RMSE: {rmse:.4f}, R²: {r2:.4f}")


######################################
# 4. EXEMPLO PyMC (Avançado) OPCIONAL
######################################
def model_nuclear_pymc():
    """
    Exemplo de regressão bayesiana usando PyMC e MCMC para prever 'Potencia_Gerada'
    no dataset Nuclear. Utiliza três features: "Temperatura_Reator", "Pressao_Reator" e "Nivel_Radiacao".

    Nesta versão, extraímos os parâmetros do traço (trace) usando stack para empilhar as dimensões "chain" e "draw"
    na nova dimensão "sample". Em seguida, transpusimos beta_samples para que sua forma seja (n_draws, n_features),
    de modo que a multiplicação matricial com X_test.T (com shape (n_features, n_test)) seja compatível.

    Necessita: pip install pymc arviz
    """
    print("\n=== MODELAGEM NUCLEAR COM PyMC (OPCIONAL) ===")
    try:
        import pymc as pm
        import arviz as az
    except ImportError:
        print("PyMC ou Arviz não instalados. Pular este exemplo.")
        return

    # Carregar o dataset e selecionar apenas as colunas numéricas
    df_nuc = pd.read_csv("../data/nuclear_preprocessed.csv", sep=";")
    df_nuc_numeric = df_nuc.select_dtypes(include=[np.number])

    target_col = "Potencia_Gerada"
    if target_col not in df_nuc_numeric.columns:
        print(f"[AVISO] Coluna {target_col} não encontrada no dataset numérico.")
        return

    # Usar 3 features específicas, se disponíveis
    features = ["Temperatura_Reator", "Pressao_Reator", "Nivel_Radiacao"]
    for c in features:
        if c not in df_nuc_numeric.columns:
            print(f"[AVISO] Coluna {c} não encontrada no dataset nuclear pré-processado.")
            return

    X = df_nuc_numeric[features].values
    y = df_nuc_numeric[target_col].values

    # Imputação para garantir que X e y não contenham NaN
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy="mean")
    X = imputer.fit_transform(X)
    y = np.where(pd.isna(y), np.nanmean(y), y)

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    n_features = X_train.shape[1]

    with pm.Model() as model_pymc:
        # Priors
        alpha = pm.Normal("alpha", mu=0, sigma=10)
        beta = pm.Normal("beta", mu=0, sigma=5, shape=n_features)
        sigma = pm.HalfNormal("sigma", sigma=5)

        # Modelo linear para os dados de treino
        mu_train = alpha + pm.math.dot(X_train, beta)
        pm.Normal("obs", mu=mu_train, sigma=sigma, observed=y_train)

        # Amostragem MCMC
        trace = pm.sample(1000, tune=1000, chains=2, cores=1, target_accept=0.95)

    print("PyMC MCMC concluído. Sumário:")
    print(az.summary(trace, var_names=["alpha", "beta", "sigma"]))

    # Extração dos parâmetros usando stack para empilhar "chain" e "draw" na dimensão "sample"
    alpha_samples = trace.posterior["alpha"].stack(sample=("chain", "draw")).values  # shape: (n_draws,)
    beta_samples = trace.posterior["beta"].stack(
        sample=("chain", "draw")).values  # inicialmente com shape (n_features, n_draws)

    # Transpor beta_samples para que tenha shape (n_draws, n_features)
    beta_samples = beta_samples.T

    # Diagnóstico: imprimir shapes
    print(f"alpha_samples.shape: {alpha_samples.shape}")  # Espera: (n_draws,)
    print(f"beta_samples.shape: {beta_samples.shape}")  # Espera: (n_draws, n_features)

    # Calcular as predições para X_test para cada draw:
    # np.dot(beta_samples, X_test.T) -> (n_draws, n_test)
    y_pred_samples = alpha_samples[:, None] + np.dot(beta_samples, X_test.T)  # shape: (n_draws, n_test)
    y_pred_mean = y_pred_samples.mean(axis=0)  # Média sobre os draws

    from sklearn.metrics import mean_squared_error, r2_score
    mse = mean_squared_error(y_test, y_pred_mean)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred_mean)

    print(f"PyMC (Nuclear) -> RMSE: {rmse:.4f}, R²: {r2:.4f}")
    print("Posterior predictive samples shape (from trace):", y_pred_samples.shape)

######################################
# MAIN
######################################
if __name__ == "__main__":
    print("Iniciando modelagem...", flush=True)
    model_hospital()
    model_bancario()
    model_nuclear()
    model_nuclear_pymc()  # opcional, se PyMC estiver instalado

    print("\n[FINAL] Modelagem concluída para todos os datasets.\n", flush=True)
