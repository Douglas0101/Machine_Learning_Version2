import csv
import random
from faker import Faker
from datetime import timedelta

def gerar_dataset_hospitalar_avancado(
        qtd_registros=5000,
        nome_arquivo='dataset_hospitalar_avancado.csv',
        seed=42,
        missing_rate=0.01,
        outlier_rate=0.01
):
    """
    Gera um dataset hospitalar sintético com parâmetros avançados:
      - qtd_registros: número de linhas (pacientes) no CSV.
      - nome_arquivo: nome do arquivo CSV gerado.
      - seed: semente de aleatoriedade para reprodutibilidade.
      - missing_rate: proporção de valores ausentes em colunas numéricas e de texto.
      - outlier_rate: proporção de outliers em colunas numéricas.
    """

    # Configurando semente para resultados reproduzíveis
    random.seed(seed)
    faker = Faker('pt_BR')

    colunas = [
        "PacienteID", "Nome", "CPF", "Data_Nascimento", "Genero",
        "Data_Admissao", "Data_Alta", "Diagnostico", "Procedimento",
        "Quantidade_Exames", "Custo_Total", "Medico_Responsavel", "Dias_Internacao"
    ]

    # Listas de possíveis diagnósticos e procedimentos
    diagnosticos = [
        "Pneumonia", "Fratura", "Infarto", "Diabetes Descompensada",
        "Covid-19", "Hipertensão", "Apendicite", "Trauma Crânio-Encefálico"
    ]
    procedimentos = [
        "Cirurgia", "Antibióticos", "Fisioterapia",
        "Cirurgia Cardíaca", "Tratamento Clínico", "Observação"
    ]

    max_dias_internacao = 30

    with open(nome_arquivo, mode='w', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file, delimiter=';')
        writer.writerow(colunas)

        for i in range(qtd_registros):
            paciente_id = f"HOSP-{i + 1:05d}"
            nome_paciente = faker.name()
            cpf = faker.cpf()
            data_nascimento = faker.date_of_birth(minimum_age=0, maximum_age=100)
            genero = random.choice(["Masculino", "Feminino"])

            # Gera data de admissão nos últimos 365 dias
            data_admissao = faker.date_between(start_date='-365d', end_date='today')
            # Data de alta
            dias_internacao = random.randint(1, max_dias_internacao)
            data_alta = data_admissao + timedelta(days=dias_internacao)

            diag = random.choice(diagnosticos)
            proc = random.choice(procedimentos)

            # Exemplo de correlação: mais exames => potencialmente maior custo
            quantidade_exames = random.randint(1, 10)
            custo_base = 500 + 2000 * (quantidade_exames ** 0.5)  # custo cresce com raiz do número de exames
            custo_aleatorio = random.uniform(0.8, 1.2) * custo_base
            custo_total = round(custo_aleatorio, 2)

            medico_responsavel = f"Dr(a). {faker.last_name()}"

            # Preparar linha
            row = [
                paciente_id,
                nome_paciente,
                cpf,
                data_nascimento.strftime("%Y-%m-%d"),
                genero,
                data_admissao.strftime("%Y-%m-%d"),
                data_alta.strftime("%Y-%m-%d"),
                diag,
                proc,
                quantidade_exames,
                custo_total,
                medico_responsavel,
                dias_internacao
            ]

            # Inserindo valores ausentes aleatoriamente
            for c in range(len(row)):
                if random.random() < missing_rate:
                    row[c] = ""  # valor ausente

            # Inserindo outliers em colunas numéricas (Quantidade_Exames, Custo_Total, Dias_Internacao)
            # Índices numéricos = 9 (exames), 10 (custo), 12 (dias)
            for idx_num in [9, 10, 12]:
                if row[idx_num] != "" and random.random() < outlier_rate:
                    # Exemplo simples: multiplicar por um fator grande
                    if idx_num == 9:  # Quantidade_Exames
                        row[idx_num] = row[idx_num] * random.randint(10, 50)
                    elif idx_num == 10:  # Custo_Total
                        row[idx_num] = round(row[idx_num] * random.uniform(5, 20), 2)
                    elif idx_num == 12:  # Dias_Internacao
                        row[idx_num] = row[idx_num] * random.randint(5, 15)

            writer.writerow(row)

    print(f"Dataset hospitalar avançado gerado: {nome_arquivo}")


def gerar_dataset_bancario_avancado(
        qtd_registros=5000,
        nome_arquivo='dataset_bancario_avancado.csv',
        seed=42,
        missing_rate=0.01,
        outlier_rate=0.01
):
    """
    Gera um dataset bancário sintético com parâmetros avançados.
      - Inclui correlação entre limite de crédito e score de crédito.
      - Possui inserção de missing e outliers.
    """

    random.seed(seed)
    faker = Faker('pt_BR')

    colunas = [
        "ContaID", "Agencia", "Tipo_Conta", "Nome_Cliente", "CPF_Cliente",
        "Data_Abertura", "Saldo_Atual", "Limite_Credito", "Data_Ultimo_Movimento",
        "Status_Conta", "Score_Credito", "Renda_Mensal"
    ]

    tipos_conta = ["Corrente", "Poupança", "Salário"]
    status_conta_choices = ["Ativa", "Inativa", "Encerrada", "Em Atraso"]

    with open(nome_arquivo, mode='w', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file, delimiter=';')
        writer.writerow(colunas)

        for i in range(qtd_registros):
            conta_id = f"{random.randint(10000, 99999)}-{chr(random.randint(65, 90))}"
            agencia = str(random.randint(1000, 9999))
            tipo_conta = random.choice(tipos_conta)
            nome_cliente = faker.name()
            cpf_cliente = faker.cpf()

            data_abertura = faker.date_between(start_date='-10y', end_date='today')
            saldo_atual = round(random.uniform(-1000, 50000), 2)

            # Correlação entre limite de crédito e score de crédito
            score_credito = random.randint(0, 1000)
            # Exemplo: limite é ~ score_credito * fator
            limite_credito_base = score_credito * random.uniform(10, 30) / 100.0
            limite_credito = round(limite_credito_base, 2)  # Em Milhares, por exemplo

            data_ultimo_movimento = faker.date_between(start_date=data_abertura, end_date='today')
            status_conta = random.choice(status_conta_choices)

            # Renda mensal também pode correlacionar com score
            renda_mensal = round(random.uniform(1000, 20000) + (score_credito * 0.5), 2)

            row = [
                conta_id,
                agencia,
                tipo_conta,
                nome_cliente,
                cpf_cliente,
                data_abertura.strftime("%Y-%m-%d"),
                saldo_atual,
                limite_credito,
                data_ultimo_movimento.strftime("%Y-%m-%d"),
                status_conta,
                score_credito,
                renda_mensal
            ]

            # Missing
            for c in range(len(row)):
                if random.random() < missing_rate:
                    row[c] = ""

            # Outliers em colunas numéricas: saldo, limite_cred, score_credito, renda_mensal
            # Índices: 6 (saldo_atual), 7 (limite_cred), 10 (score), 11 (renda)
            numeric_idxs = [6, 7, 10, 11]
            for idx in numeric_idxs:
                if row[idx] != "" and random.random() < outlier_rate:
                    if idx == 6:  # saldo
                        row[idx] = round(row[idx] * random.uniform(5, 10), 2)
                    elif idx == 7:  # limite
                        row[idx] = round(row[idx] * random.uniform(5, 15), 2)
                    elif idx == 10:  # score
                        row[idx] = min(1500, int(row[idx] * random.randint(2, 5)))  # cap em 1500
                    elif idx == 11:  # renda
                        row[idx] = round(row[idx] * random.uniform(3, 10), 2)

            writer.writerow(row)

    print(f"Dataset bancário avançado gerado: {nome_arquivo}")


def gerar_dataset_energia_nuclear_avancado(
        qtd_registros=5000,
        nome_arquivo='dataset_energia_nuclear_avancado.csv',
        seed=42,
        missing_rate=0.01,
        outlier_rate=0.01
):
    """
    Gera um dataset de energia nuclear avançado.
      - Simula plantas nucleares, inspeções, status operacional.
      - Inclui parâmetros de correlação (temp x pressão), missing e outliers.
    """

    random.seed(seed)
    faker = Faker()

    colunas = [
        "PlantaID", "Nome_Planta", "Localizacao", "Data_Inspecao", "Nivel_Radiacao",
        "Temperatura_Reator", "Pressao_Reator", "Eficiência_Operacional",
        "Falhas_Detectadas", "Custo_Manutencao", "Status_Operacional", "Potencia_Gerada"
    ]

    status_operacional_choices = ["Operando", "Em Manutenção", "Parcialmente Operando", "Desativada"]
    plantas_nomes = ["Angra 1", "Angra 2", "Central Nuclear Santa", "NucleoForte", "Reator Atlântico"]
    localizacoes = ["RJ - Brasil", "SP - Brasil", "MG - Brasil", "BA - Brasil", "SC - Brasil"]

    with open(nome_arquivo, mode='w', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file, delimiter=';')
        writer.writerow(colunas)

        for i in range(qtd_registros):
            planta_id = f"NUC-{i + 1:05d}"
            nome_planta = random.choice(plantas_nomes)
            localizacao = random.choice(localizacoes)

            # Data de inspeção
            data_inspecao = faker.date_between(start_date='-730d', end_date='today')

            # Radiação (mSv/h)
            nivel_radiacao = round(random.uniform(0.01, 5.0), 2)

            # Correlação: pressão depende um pouco da temperatura
            temperatura_reator = round(random.uniform(200, 450), 2)
            pressao_reator = round((temperatura_reator * random.uniform(0.1, 0.3)) + random.uniform(20, 40), 2)

            eficiencia_operacional = round(random.uniform(50, 100), 2)
            falhas_detectadas = random.randint(0, 5)
            custo_manutencao = round(random.uniform(10, 500), 2)
            status_operacional = random.choice(status_operacional_choices)

            # Potência gerada (MW), correlacionada com eficiência e status
            if status_operacional == "Operando":
                potencia_gerada = round(eficiencia_operacional * random.uniform(0.8, 1.2), 2)
            elif status_operacional == "Em Manutenção":
                potencia_gerada = round(random.uniform(0, 20), 2)
            elif status_operacional == "Parcialmente Operando":
                potencia_gerada = round(eficiencia_operacional * random.uniform(0.3, 0.6), 2)
            else:
                potencia_gerada = 0

            row = [
                planta_id,
                nome_planta,
                localizacao,
                data_inspecao.strftime("%Y-%m-%d"),
                nivel_radiacao,
                temperatura_reator,
                pressao_reator,
                eficiencia_operacional,
                falhas_detectadas,
                custo_manutencao,
                status_operacional,
                potencia_gerada
            ]

            # Inserindo missing
            for c in range(len(row)):
                if random.random() < missing_rate:
                    row[c] = ""

            # Inserindo outliers em colunas numéricas
            numeric_idxs = [4, 5, 6, 7, 8, 9, 11]
            for idx in numeric_idxs:
                if row[idx] != "" and random.random() < outlier_rate:
                    if idx in [4, 5, 6]:
                        # Aumentar radiação, temp ou pressão
                        row[idx] = round(row[idx] * random.uniform(5, 10), 2)
                    elif idx == 7:
                        row[idx] = min(100.0, round(row[idx] * random.uniform(2, 5), 2))  # efficiency cap
                    elif idx == 8:
                        row[idx] = row[idx] + random.randint(5, 20)  # falhas extras
                    elif idx == 9:
                        row[idx] = round(row[idx] * random.uniform(5, 15), 2)
                    elif idx == 11:
                        row[idx] = round(row[idx] * random.uniform(3, 10), 2)

            writer.writerow(row)

    print(f"Dataset de energia nuclear avançado gerado: {nome_arquivo}")


if __name__ == "__main__":
    # Exemplos de uso dos scripts avançados:
    gerar_dataset_hospitalar_avancado(
        qtd_registros=5000,
        nome_arquivo='dataset_hospitalar_avancado.csv',
        seed=123,
        missing_rate=0.02,
        outlier_rate=0.02
    )

    gerar_dataset_bancario_avancado(
        qtd_registros=5000,
        nome_arquivo='dataset_bancario_avancado.csv',
        seed=123,
        missing_rate=0.02,
        outlier_rate=0.02
    )

    gerar_dataset_energia_nuclear_avancado(
        qtd_registros=5000,
        nome_arquivo='dataset_energia_nuclear_avancado.csv',
        seed=123,
        missing_rate=0.02,
        outlier_rate=0.02
    )