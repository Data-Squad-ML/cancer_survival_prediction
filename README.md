# cancer_survival_prediction

## Resumo

Este repositório apresenta um estudo aplicado de ciência de dados e aprendizado de máquina voltado à análise da sobrevivência de pacientes com câncer. A organização do projeto sugere uma metodologia em etapas, contemplando preparação de dados, exploração estatística, transformação de variáveis, modelagem preditiva e avaliação de desempenho.

O foco principal é investigar relações entre características clínicas, demográficas e terapêuticas e o desfecho de sobrevivência, por meio de modelos supervisionados clássicos e abordagens survival analysis.

## Contexto metodológico

A estrutura do projeto indica um fluxo analítico consistente com pesquisas em dados clínicos:

1. construção e refinamento do conjunto de dados;
2. análise exploratória para identificação de padrões, distribuições e inconsistências;
3. codificação de variáveis categóricas e engenharia de atributos;
4. treinamento de modelos de classificação e survival;
5. comparação de resultados por métricas apropriadas ao problema.

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

## Base de dados e documentação

A pasta `data/` reúne os insumos principais para a pesquisa:

- `global_cancer_patients_2015_2024.csv`: base tabular principal utilizada no projeto;
- `dicionario_inca_registro_hospitalar.pdf`: documento de referência para interpretação das variáveis.

## Notebooks principais

Os notebooks em `src/` organizam o processo analítico em etapas:

- `01_data_preparation.ipynb`: preparação inicial, limpeza e estruturação dos dados;
- `02_exploratory_data_analysis.ipynb`: exploração descritiva e visualização dos dados;
- `03_feature_encoding.ipynb`: transformação, codificação e seleção de atributos;
- `04_modeling_and_evaluation.ipynb`: treinamento e avaliação dos modelos;
- `cancer_brasil.ipynb`: notebook complementar, possivelmente voltado a análises específicas do conjunto brasileiro.

## Scripts auxiliares

Os scripts de apoio são usados para inspeções pontuais de qualidade dos dados:

- `check_nans.py`: identifica colunas com valores ausentes no conjunto de treino;
- `test_nans.py`: calcula o total de valores ausentes e lista as colunas afetadas.

Esses scripts operam sobre arquivos parquet, o que sugere um estágio intermediário de processamento anterior ao treinamento.

## Ambiente computacional

O projeto foi estruturado para **Python 3.12** e utiliza bibliotecas amplamente empregadas em pesquisa aplicada e modelagem preditiva:

- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `pyarrow`
- `scikit-learn`
- `xgboost`
- `lightgbm`
- `scikit-survival`
- `autogluon.tabular`
- `shap`

A lista completa está disponível em `requirements.txt`.

## Reprodutibilidade

### Clonagem do repositório

```bash
git clone https://github.com/Data-Squad-ML/cancer_survival_prediction.git
cd cancer_survival_prediction
```

### Criação do ambiente virtual

```bash
python3.12 -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows
```

### Instalação das dependências

```bash
pip install -r requirements.txt
```

### Execução dos notebooks

```bash
jupyter lab
```

ou

```bash
jupyter notebook
```

## Resultados dos modelos

Os resultados experimentais do projeto estão concentrados em `src/scripts/results/`, o que indica uma organização sistemática dos artefatos de avaliação.

### Resultados de classificação

Em `src/scripts/results/classification/`, encontram-se os outputs dos modelos voltados à tarefa de classificação, incluindo métricas de desempenho e importância de variáveis.

Pelos scripts do repositório, os modelos de classificação incluem:

- **Random Forest**
- **MLP / rede neural**
- **AutoGluon Tabular**

As saídas típicas geradas por esses experimentos incluem:

- score de validação cruzada;
- AUC em treino e teste;
- gap entre treino e teste;
- permutation importance;
- relatório consolidado em TXT ou CSV.

### Resultados de survival analysis

Em `src/scripts/results/survival/`, estão os resultados associados aos modelos de sobrevivência, com foco em métricas mais adequadas ao problema temporal.

Os scripts do repositório sugerem experimentos com:

- **Random Survival Forest**
- **LightGBM Survival (Cox)**
- **XGBoost Survival (Cox)**
- possivelmente outras variações de survival em notebooks e scripts complementares

Entre os artefatos produzidos estão:

- **C-index** de treino, validação e teste;
- **Uno C-index**;
- **AUC dinâmica** ao longo do tempo;
- **Brier score** e **Integrated Brier Score (IBS)**;
- importâncias por ganho, permutation importance e SHAP;
- relatórios em texto e tabelas auxiliares em CSV;
- gráficos de resumo SHAP.

### Interpretação acadêmica dos resultados

A presença simultânea de métricas de classificação e survival sugere uma comparação metodológica entre abordagens diferentes para o mesmo fenômeno clínico. Em termos acadêmicos, isso é relevante porque:

- métricas como **AUC** capturam apenas separação entre classes;
- métricas survival como **C-index**, **AUC dinâmica** e **IBS** incorporam tempo e censura;
- explicabilidade por **SHAP** e permutation importance permite discutir fatores potencialmente associados ao desfecho.

Assim, a seção de resultados do repositório não apenas documenta desempenho, mas também contribui para a interpretação clínica dos modelos.

## Modelos e métricas utilizadas

### Classificação

- **Random Forest**
- **MLP em PyTorch**
- **AutoGluon Tabular**

Métricas observadas nos scripts:

- ROC-AUC
- AUC em treino e teste
- Permutation importance

### Survival analysis

- **Random Survival Forest**
- **LightGBM Survival (Cox)**
- **XGBoost Survival (Cox)**

Métricas observadas nos scripts:

- C-index
- Uno C-index
- AUC dinâmica
- Brier score
- Integrated Brier Score (IBS)
- SHAP global
- permutation importance

## Estrutura dos experimentos em `src/scripts/results/`

A organização dos resultados foi desenhada para separar experimentos por família de modelo:

- `classification/`: saídas de modelos classificatórios;
- `survival/`: saídas dos modelos de análise de sobrevivência;
- `autogluon/`: resultados específicos do AutoGluon.

Essa separação favorece rastreabilidade, comparação e reprodutibilidade experimental.

## Licença

Este projeto está licenciado sob a **MIT License**. Consulte o arquivo `LICENSE` para os termos completos.
