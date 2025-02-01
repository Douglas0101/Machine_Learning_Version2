import numpy as np
import pandas as pd

def generate_finance_dataset(n_samples=10000, seed=42):
    np.random.seed(seed)
    data = {}

    # --- Variáveis independentes ---
    data["idade"] = np.clip(np.random.normal(35, 10, n_samples), 18, 70).astype(int)
    data["escolaridade"] = np.random.choice(["Fundamental", "Médio", "Superior", "Pós"], n_samples)
    data["estado_civil"] = np.random.choice(["Solteiro", "Casado", "Divorciado"], n_samples)
    data["numero_filhos"] = np.random.poisson(0.8, n_samples)
    data["renda_mensal"] = np.round(np.exp(np.random.normal(8.5, 0.6, n_samples)), 2)
    data["valor_emprestimo"] = np.random.gamma(2, 5000, n_samples).round(2)
    data["taxa_juros_nominal"] = (np.random.beta(2, 5, n_samples) * 20 + 5).round(2)
    data["valor_garantia"] = np.random.lognormal(10, 0.5, n_samples).round(2)
    data["PIB_regiao"] = np.random.uniform(1e6, 5e6, n_samples).round(2)
    data["taxa_desemprego"] = np.random.beta(2, 5, n_samples).round(2)
    data["inflacao_mensal"] = np.random.normal(0.5, 0.1, n_samples).round(2)
    data["trimestre_solicitacao"] = np.random.randint(1, 5, n_samples)
    data["dia_util"] = np.random.binomial(1, 0.7, n_samples)

    # --- Variáveis dependentes ---
    data["renda_idade"] = (data["renda_mensal"] / 1000) * data["idade"]
    data["divida_renda"] = np.random.gamma(2, 1, n_samples)
    data["taxa_juros_efetiva"] = data["taxa_juros_nominal"] * (1 + data["inflacao_mensal"] / 100)
    data["score_credito"] = np.clip(np.random.normal(600, 100, n_samples), 300, 850).astype(int)
    data["score_credito"] -= 50 * (data["taxa_desemprego"] > 0.1)
    data["inadimplente"] = np.random.binomial(1, 0.15, n_samples)
    data["score_credito"] -= 50 * data["inadimplente"]

    # --- Missing Data ---
    for col in ["valor_garantia", "escolaridade", "PIB_regiao"]:
        data[col][np.random.choice(n_samples, int(n_samples * 0.03), replace=False)] = np.nan

    # --- Dataframe e salvamento (CSV) ---
    df = pd.DataFrame(data)
    df.to_csv("../data/finance_dataset.csv", index=False)

    # --- DataFrame e salvamento ---
    df = pd.DataFrame(data)
    df.to_parquet("../data/finance_dataset.parquet", engine='pyarrow', index=False)
    return df

finance_df = generate_finance_dataset()