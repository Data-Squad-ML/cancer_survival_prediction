# cancer_survival_prediction

Projeto de ciência de dados e machine learning para análise e predição de sobrevivência de pacientes com câncer, com foco em preparação de dados, exploração, codificação de variáveis, modelagem e avaliação de desempenho.

## Visão geral

Este repositório reúne notebooks e scripts voltados à construção de um pipeline analítico para dados oncológicos. A estrutura sugere um fluxo de trabalho típico de projeto de ML:

1. **Preparação dos dados**
2. **Análise exploratória**
3. **Encoding/feature engineering**
4. **Treinamento e avaliação de modelos**

Além disso, o repositório contém um conjunto de dados em CSV, documentação de dicionário de dados e dependências para reproduzir o ambiente.

## Objetivo do projeto

O objetivo principal é explorar dados clínicos e demográficos para entender fatores associados à sobrevivência e apoiar a construção de modelos preditivos para câncer.

## Estrutura do repositório

```text
.
├── data/
│   ├── global_cancer_patients_2015_2024.csv
│   └── dicionario_inca_registro_hospitalar.pdf
├── src/
│   ├── 01_data_preparation.ipynb
│   ├── 02_exploratory_data_analysis.ipynb
│   ├── 03_feature_encoding.ipynb
│   ├── 04_modeling_and_evaluation.ipynb
│   ├── cancer_brasil.ipynb
│   └── scripts/
├── check_nans.py
├── test_nans.py
├── requirements.txt
└── README.md
```

## Principais arquivos e pastas

### `src/`
Contém os notebooks principais do projeto:

- `01_data_preparation.ipynb`: preparação e tratamento inicial dos dados
- `02_exploratory_data_analysis.ipynb`: análise exploratória
- `03_feature_encoding.ipynb`: transformação e codificação de variáveis
- `04_modeling_and_evaluation.ipynb`: treinamento e avaliação de modelos
- `cancer_brasil.ipynb`: notebook adicional relacionado ao domínio do projeto
- `scripts/`: espaço para scripts auxiliares ou automações

### `data/`
Contém os dados e a documentação de suporte:

- `global_cancer_patients_2015_2024.csv`: base principal de pacientes
- `dicionario_inca_registro_hospitalar.pdf`: dicionário/guia de referência dos dados

### Scripts auxiliares

- `check_nans.py`: verifica colunas com valores ausentes em um parquet de treino
- `test_nans.py`: imprime o total de NaNs e colunas com ausências

## Tecnologias e dependências

O projeto foi preparado para Python 3.12 e usa bibliotecas de análise e modelagem como:

- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `xgboost`
- `lightgbm`
- `scikit-survival`
- `autogluon.tabular`
- `shap`
- `pyarrow`

Veja a lista completa em `requirements.txt`.

## Como executar localmente

### 1. Clonar o repositório

```bash
git clone https://github.com/Data-Squad-ML/cancer_survival_prediction.git
cd cancer_survival_prediction
```

### 2. Criar e ativar um ambiente virtual

```bash
python3.12 -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows
```

### 3. Instalar as dependências

```bash
pip install -r requirements.txt
```

### 4. Abrir os notebooks

Você pode usar Jupyter Notebook, JupyterLab ou VS Code:

```bash
jupyter lab
```

ou

```bash
jupyter notebook
```

## Fluxo sugerido de uso

1. Comece por `01_data_preparation.ipynb` para entender a origem e o tratamento dos dados.
2. Em seguida, use `02_exploratory_data_analysis.ipynb` para identificar padrões, distribuições e problemas de qualidade.
3. Aplique codificação e engenharia de atributos em `03_feature_encoding.ipynb`.
4. Treine e compare modelos em `04_modeling_and_evaluation.ipynb`.
5. Utilize `check_nans.py` e `test_nans.py` para inspeções rápidas de valores ausentes.

## Verificação de dados faltantes

Os scripts incluídos mostram como inspecionar NaNs em um arquivo parquet de treino:

```python
import pandas as pd
train_pdf = pd.read_parquet("/Volumes/workspace/default/cancer_data/03_train.parquet")
nans = train_pdf.isna().sum()
print(nans[nans > 0])
```

## Observações importantes

- O repositório é **privado**.
- Não há uma descrição formal do projeto no GitHub, então este README foi estruturado com base na organização atual dos arquivos.
- O caminho do parquet usado nos scripts (`/Volumes/workspace/default/cancer_data/03_train.parquet`) pode precisar ser ajustado ao seu ambiente.

## Próximos passos recomendados

- Adicionar uma seção de **objetivo do modelo** com definição clara da variável-alvo.
- Documentar o **dataset**: fonte, período, número de linhas, colunas e dicionário de variáveis.
- Incluir **métricas de avaliação** dos modelos usados no notebook final.
- Adicionar **figuras** com resultados de EDA e performance.
- Padronizar os notebooks com comentários e sumário executivo.

## Licença

Este projeto está sob a licença MIT. Consulte o arquivo `LICENSE` para detalhes.
