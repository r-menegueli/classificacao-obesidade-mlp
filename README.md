# Classificação de obesidade com rede neural MLP

Atividade acadêmica de análise de dados e classificação multiclasse com Python e scikit-learn. O objetivo é classificar as categorias `Underweight`, `Normal Weight`, `Overweight` e `Obese` a partir de idade, gênero, altura e peso.

## Melhorias da curadoria

- caminho do CSV tornado portátil;
- caracteres e títulos corrigidos;
- divisão estratificada e semente aleatória fixa;
- pré-processamento integrado ao modelo para evitar vazamento entre treino e teste;
- gênero codificado com `OneHotEncoder`;
- validação cruzada estratificada repetida;
- métricas balanceadas, F1 macro e matriz de confusão;
- BMI removido das entradas, pois ele praticamente define a classe e inflaria artificialmente o resultado.

## Execução

```bash
python -m pip install -r requirements.txt
python analise_obesidade.py
```

Os resultados são gravados em `resultados/`.

## Limitações

O conjunto possui apenas 108 amostras e é desbalanceado. As métricas servem para demonstrar o fluxo de modelagem, não para validar uso clínico. O projeto não deve ser usado para diagnóstico ou orientação médica.
