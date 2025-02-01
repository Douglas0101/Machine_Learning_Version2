import numpy as np
import pandas as pd

def generate_hospital_dataset(n_samples=10000, seed=42):
    np.random.seed(seed)

    # Inicializa o dicionário 'data'
    data = {}

    # --- Variáveis independentes ---
    data["idade"] = np.clip(np.random.normal(55, 15, n_samples), 18, 90).astype(int)
    data["genero"] = np.random.choice(["M", "F", "Outro"], n_samples, p=[0.48, 0.50, 0.02])
    data["altura"] = np.round(np.random.normal(1.7, 0.1, n_samples), 2)
    data["peso"] = np.round(np.random.normal(70, 15, n_samples), 1)
    data["comorbidades"] = np.random.poisson(1.5, n_samples)
    data["fumante"] = np.random.binomial(1, 0.2, n_samples)
    data["tipo_sanguineo"] = np.random.choice(["A+", "A-", "B+", "B-", "O+", "O-"], n_samples)
    data["tipo_tratamento"] = np.random.choice(["Cirúrgico", "Clínico", "Urgência"], n_samples, p=[0.3, 0.5, 0.2])
    data["dose_medicamento"] = np.random.gamma(2, 1, n_samples).round(2)
    data["tempo_cirurgia"] = np.random.normal(2, 0.5, n_samples).clip(0.5, 4)
    data["equipe_medica"] = np.random.choice(["Equipe_A", "Equipe_B", "Equipe_C"], n_samples)
    data["hospital_id"] = np.random.choice([f"HOSP_{i}" for i in range(1, 21)], n_samples)
    data["numero_leitos"] = np.random.poisson(200, n_samples)
    data["orcamento_anual"] = np.random.lognormal(20, 0.3, n_samples).round(2)
    data["regiao_hospital"] = np.random.choice(["Norte", "Sul", "Leste", "Oeste"], n_samples)
    data["dia_semana"] = np.random.choice(["Seg", "Ter", "Qua", "Qui", "Sex", "Sáb", "Dom"], n_samples)
    data["hora_internacao"] = np.random.randint(0, 24, n_samples)
    data["mes_internacao"] = np.random.randint(1, 13, n_samples)

    # --- Variáveis dependentes (calculadas após inicialização) ---
    data["idade_tipo_tratamento"] = np.where(
        (data["tipo_tratamento"] == "Cirúrgico") & (data["idade"] > 60), 1, 0
    )
    data["comorbidades_dose"] = 0.1 * data["comorbidades"] * data["dose_medicamento"]
    data["custo_tratamento"] = np.random.lognormal(8, 0.5, n_samples).round(2) + 100 * data["comorbidades"] + 50 * data[
        "idade_tipo_tratamento"]
    data["readmitido_30d"] = np.where(
        (data["comorbidades"] > 3) & (data["fumante"] == 1),
        1,
        np.random.binomial(1, 0.2, n_samples)
    )

    # --- Missing Data e Outliers ---
    for col in ["dose_medicamento", "tipo_sanguineo", "equipe_medica"]:
        data[col][np.random.choice(n_samples, int(n_samples * 0.05), replace=False)] = np.nan

    outlier_idx = np.random.choice(n_samples, int(n_samples * 0.03), replace=False)
    data["custo_tratamento"][outlier_idx] *= 10

    # --- DataFrame e salvamento (CSV) ---
    df = pd.DataFrame(data)
    df.to_csv("../data/hospital_dataset.csv", index=False)

    # --- Criar DataFrame e salvar ---
    df = pd.DataFrame(data)
    df.to_parquet("../data/hospital_dataset.parquet", engine='pyarrow', index=False)
    return df

hospital_df = generate_hospital_dataset()