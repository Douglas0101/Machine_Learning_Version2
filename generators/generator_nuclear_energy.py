import numpy as np
import pandas as pd


def generate_nuclear_dataset(n_samples=10000, seed=42):
    np.random.seed(seed)
    data = {}

    # --- Variáveis independentes ---
    data["potencia_nominal"] = np.random.uniform(800, 1200, n_samples).round(2)
    data["vazao_refrigerante"] = np.random.normal(5000, 200, n_samples).round(2)
    data["nivel_combustivel"] = np.random.uniform(80, 100, n_samples).round(2)
    data["temperatura"] = np.clip(np.random.normal(300, 50, n_samples) + 0.5 * np.arange(n_samples), 200, 1000).round(2)
    data["pressao"] = np.random.weibull(2, n_samples) * 15
    data["vibracao_x"] = np.random.normal(0, 0.5, n_samples).round(3)
    data["vibracao_y"] = np.random.normal(0, 0.5, n_samples).round(3)
    data["vibracao_z"] = np.random.normal(0, 0.5, n_samples).round(3)
    data["fluxo_neutrons"] = np.random.poisson(1000, n_samples).astype(float)  # Convertido para float!
    data["temperatura_externa"] = np.random.normal(25, 5, n_samples).round(2)
    data["umidade"] = np.random.uniform(30, 90, n_samples).round(2)
    data["pressao_atmosferica"] = np.random.normal(1013, 10, n_samples).round(2)
    data["ultima_manutencao"] = np.random.randint(0, 365, n_samples)
    data["horas_operacao"] = np.random.poisson(5000, n_samples).astype(float)  # Convertido para float!
    data["falhas_passadas"] = np.random.poisson(0.5, n_samples)

    # --- Variáveis dependentes ---
    data["pressao_temperatura"] = (data["pressao"] / 10) * (data["temperatura"] / 100)
    data["fluxo_neutrons_potencia"] = data["fluxo_neutrons"] * data["potencia_nominal"] / 1000
    data["eficiencia_termica"] = np.random.normal(85, 5, n_samples).round(2) - 0.1 * (
                data["temperatura"] - 300) ** 2 - 0.05 * data["falhas_passadas"]
    data["falha"] = np.where((data["pressao"] > 30) & (data["temperatura"] > 350), 1, 0)

    # --- Missing Data (garantindo colunas como float) ---
    cols_para_nan = ["vazao_refrigerante", "vibracao_x", "fluxo_neutrons"]
    for col in cols_para_nan:
        # Converte para float antes de inserir NaNs
        data[col] = data[col].astype(float)
        data[col][np.random.choice(n_samples, int(n_samples * 0.05), replace=False)] = np.nan

    # --- DataFrame e salvamento (CSV) ---
    df = pd.DataFrame(data)
    df.to_csv("../data/nuclear_dataset.csv", index=False)

    # --- DataFrame e salvamento ---
    df = pd.DataFrame(data)
    df.to_parquet("../data/nuclear_dataset.parquet", engine='pyarrow', index=False)
    return df

nuclear_df = generate_nuclear_dataset()