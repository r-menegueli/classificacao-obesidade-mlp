# Classificação de obesidade com rede neural MLP

[![CI](https://github.com/r-menegueli/classificacao-obesidade-mlp/actions/workflows/ci.yml/badge.svg)](https://github.com/r-menegueli/classificacao-obesidade-mlp/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-MLP-F7931E?logo=scikitlearn&logoColor=white)

Projeto acadêmico de classificação multiclasse com Python e scikit-learn. O modelo prevê `Underweight`, `Normal Weight`, `Overweight` e `Obese` a partir de idade, gênero, altura e peso.

## Pipeline

```mermaid
flowchart LR
    CSV[108 amostras] --> Validacao[Validação do esquema]
    Validacao --> Split[Divisão estratificada]
    Split --> Pre[Escala numérica + one-hot]
    Pre --> MLP[MLP 24-12]
    MLP --> CV[Validação cruzada 4 x 5]
    MLP --> Holdout[Conjunto de teste]
    CV --> Metricas[Acurácia balanceada e F1 macro]
    Holdout --> Matriz[Matriz de confusão]
```

## Resultados reproduzíveis

| Métrica na validação cruzada | Média | Desvio-padrão |
| --- | ---: | ---: |
| Acurácia | 93,70% | 5,25% |
| Acurácia balanceada | 92,90% | 6,72% |
| F1 macro | 92,45% | 6,72% |

![Matriz de confusão do conjunto de teste](resultados/matriz_confusao.png)

O `holdout` de 27 amostras foi classificado integralmente, mas a validação cruzada repetida é a estimativa mais confiável por aproveitar diferentes divisões do conjunto pequeno.

## Decisões de modelagem

- caminho do CSV portátil;
- divisão estratificada e semente fixa;
- pré-processamento dentro do pipeline para evitar vazamento entre treino e teste;
- variáveis numéricas padronizadas e gênero codificado com `OneHotEncoder`;
- MLP com camadas ocultas de 24 e 12 neurônios;
- BMI removido das entradas, pois praticamente define a classe e inflaria artificialmente o resultado;
- métricas e previsões gravadas em formatos abertos.

## Execução

```bash
python -m pip install -r requirements.txt
python analise_obesidade.py
```

Os resultados são atualizados em `resultados/`. A integração contínua executa o pipeline e verifica a estrutura das métricas a cada `push` e `pull request`.

## Limitações

O conjunto possui somente 108 amostras e é desbalanceado. O projeto demonstra engenharia de atributos e avaliação de modelos; **não deve ser utilizado para diagnóstico ou orientação médica**.
