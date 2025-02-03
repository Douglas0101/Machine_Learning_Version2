import unittest
import pandas as pd
import numpy as np

# Importar as funções do seu arquivo preprocessing
# Ajuste o import conforme o caminho real do seu projeto:
from preprocessing import (
    preprocess_hospital_data,
    preprocess_bancario_data,
    preprocess_nuclear_data
)

class TestPreprocessingHospital(unittest.TestCase):
    def setUp(self):
        """
        Cria um pequeno DataFrame sintético para testar
        a função preprocess_hospital_data.
        Contém:
          - missing values
          - datas inconsistentes
          - outliers
        """
        data = {
            "PacienteID": ["HOSP-00001", "HOSP-00002", "HOSP-00003", "HOSP-00004"],
            "Nome": ["Maria Silva", "João Santos", "Ana Pereira", None],  # 1 missing
            "CPF": ["123.456.789-10", "111.222.333-44", None, "333.222.111-55"],
            "Data_Nascimento": ["1980-05-10", "1970-01-01", None, "2000-12-31"],
            "Genero": ["Feminino", None, "Masculino", "Feminino"],
            "Data_Admissao": ["2022-01-10", "2022-01-15", "2022-02-01", "2022-02-05"],
            "Data_Alta": ["2022-01-12", "2022-01-20", "2022-03-01", "2022-02-20"],
            "Diagnostico": ["Pneumonia", "Fratura", None, "Covid-19"],
            "Procedimento": ["Cirurgia", "Observação", "Fisioterapia", "Antibióticos"],
            "Quantidade_Exames": [5, np.nan, 50, 2],
            "Custo_Total": [1000.0, 200000.0, 500.0, np.nan],  # note 200000 é um outlier
            "Medico_Responsavel": ["Dr(a). Martins", "Dr(a). Lemos", "Dr(a). Costa", "Dr(a). Xavier"],
            # Falta a coluna Dias_Internacao para ver se o pipeline cria
        }
        self.df_hosp = pd.DataFrame(data)

    def test_preprocessing_hospital(self):
        """Testa a função preprocess_hospital_data com DataFrame sintético."""
        # Salvar DataFrame em CSV para simular arquivo real
        test_csv = "hospital_test.csv"
        self.df_hosp.to_csv(test_csv, sep=";", index=False)

        # Chamar a função
        df_result = preprocess_hospital_data(test_csv)

        # Verificar se não dá erro e retorna um DataFrame
        self.assertIsInstance(df_result, pd.DataFrame)

        # Verificar se removeu ou winsorizou outliers etc.
        # Ao menos checar se não há valores NaN em colunas numéricas transformadas
        numeric_check = df_result.select_dtypes(include=[np.number])
        self.assertFalse(numeric_check.isnull().values.any(), "Ainda há valores numéricos nulos pós-transformação.")

        # Verificar se 'Dias_Internacao' foi criado (ou existe)
        self.assertIn("Dias_Internacao", df_result.columns, "Dias_Internacao não foi criado ou passou despercebido.")

        # Verificar se shape é coerente (após one-hot e remainder)
        # Não sabemos exatamente quantas colunas extra o OneHotEncoder criará, mas ao menos checamos se não está vazio:
        self.assertGreater(df_result.shape[0], 0, "Não retornou linhas no DataFrame final.")
        self.assertGreater(df_result.shape[1], 5, "Poucas colunas, esperado mais após o pipeline.")

        print("\n[Hospital] df_result HEAD:")
        print(df_result.head())


class TestPreprocessingBancario(unittest.TestCase):
    def setUp(self):
        data = {
            "ContaID": ["10000-A", "20000-B", "30000-C"],
            "Agencia": ["1234", "9999", None],
            "Tipo_Conta": ["Corrente", "Poupança", None],
            "Nome_Cliente": ["Alice", None, "Carlos"],
            "CPF_Cliente": ["123.456.789-10", None, "987.654.321-00"],
            "Data_Abertura": ["2015-01-01", "2020-05-10", "2010-12-31"],
            "Saldo_Atual": [5000.50, -2000.10, 9999999.0],  # 9999999 é outlier
            "Limite_Credito": [1000.0, np.nan, 20000.0],
            "Data_Ultimo_Movimento": ["2023-01-01", None, "2023-04-10"],
            "Status_Conta": ["Ativa", "Em Atraso", None],
            "Score_Credito": [300, 800, 1200],  # 1200 pode ser outlier
            "Renda_Mensal": [3000.0, 1500.0, np.nan]
        }
        self.df_banco = pd.DataFrame(data)

    def test_preprocessing_bancario(self):
        test_csv = "bancario_test.csv"
        self.df_banco.to_csv(test_csv, sep=";", index=False)

        df_result = preprocess_bancario_data(test_csv)
        self.assertIsInstance(df_result, pd.DataFrame)

        # Checar se não há nulos em colunas numéricas
        numeric_check = df_result.select_dtypes(include=[np.number])
        self.assertFalse(numeric_check.isnull().values.any(), "Ainda há valores numéricos nulos no dataset bancário.")

        # Checar se 'Anos_Conta' foi criado
        self.assertIn("Anos_Conta", df_result.columns, "Anos_Conta não foi criado.")

        print("\n[Bancário] df_result HEAD:")
        print(df_result.head())


class TestPreprocessingNuclear(unittest.TestCase):
    def setUp(self):
        data = {
            "PlantaID": ["NUC-00001", "NUC-00002"],
            "Nome_Planta": ["Angra 1", "Reator Atlântico"],
            "Localizacao": ["RJ - Brasil", None],
            "Data_Inspecao": ["2023-01-01", None],
            "Nivel_Radiacao": [3.5, 999.0],  # 999 é outlier
            "Temperatura_Reator": [400.0, 2000.0],  # 2000 é outlier
            "Pressao_Reator": [150.0, 10.0],
            "Eficiência_Operacional": [95.0, 10.0],
            "Falhas_Detectadas": [0, 10],
            "Custo_Manutencao": [100.0, 999999.0],  # outlier
            "Status_Operacional": ["Operando", "Em Manutenção"],
            "Potencia_Gerada": [80.0, 999.0], # outro outlier
        }
        self.df_nuc = pd.DataFrame(data)

    def test_preprocessing_nuclear(self):
        test_csv = "nuclear_test.csv"
        self.df_nuc.to_csv(test_csv, sep=";", index=False)

        df_result = preprocess_nuclear_data(test_csv)
        self.assertIsInstance(df_result, pd.DataFrame)

        # Checar se colunas numéricas não têm nulos
        numeric_check = df_result.select_dtypes(include=[np.number])
        self.assertFalse(numeric_check.isnull().values.any(), "Há valores numéricos nulos no dataset nuclear.")

        # Verificar se foram criadas Ano_Inspecao e Mes_Inspecao
        self.assertIn("Ano_Inspecao", df_result.columns, "Ano_Inspecao não foi criado.")
        self.assertIn("Mes_Inspecao", df_result.columns, "Mes_Inspecao não foi criado.")

        print("\n[Nuclear] df_result HEAD:")
        print(df_result.head())

if __name__ == '__main__':
    unittest.main()