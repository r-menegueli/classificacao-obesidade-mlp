"""Treina e avalia uma MLP em um pequeno conjunto de classificação de obesidade."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RAIZ = Path(__file__).resolve().parent
ARQUIVO_DADOS = RAIZ / "dados" / "obesity_classification.csv"
PASTA_RESULTADOS = RAIZ / "resultados"
SEMENTE = 42


def criar_pipeline() -> Pipeline:
    """Cria o pré-processamento e o classificador de forma reproduzível."""
    pre_processamento = ColumnTransformer(
        transformers=[
            ("numericas", StandardScaler(), ["Age", "Height", "Weight"]),
            (
                "categorica",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ["Gender"],
            ),
        ]
    )

    classificador = MLPClassifier(
        hidden_layer_sizes=(24, 12),
        activation="relu",
        solver="lbfgs",
        alpha=1e-3,
        max_iter=5_000,
        random_state=SEMENTE,
    )
    return Pipeline(
        steps=[
            ("pre_processamento", pre_processamento),
            ("classificador", classificador),
        ]
    )


def main() -> None:
    dados = pd.read_csv(ARQUIVO_DADOS)
    colunas_obrigatorias = {
        "ID",
        "Age",
        "Gender",
        "Height",
        "Weight",
        "BMI",
        "Label",
    }
    ausentes = colunas_obrigatorias.difference(dados.columns)
    if ausentes:
        raise ValueError(f"Colunas ausentes: {sorted(ausentes)}")
    if dados[list(colunas_obrigatorias)].isna().any().any():
        raise ValueError("O conjunto contém valores ausentes.")

    # O BMI foi removido das entradas porque ele praticamente define a classe
    # alvo e produziria uma estimativa excessivamente otimista por vazamento.
    entradas = dados[["Age", "Gender", "Height", "Weight"]]
    alvo = dados["Label"]

    validacao = RepeatedStratifiedKFold(
        n_splits=4,
        n_repeats=5,
        random_state=SEMENTE,
    )
    pontuacoes = cross_validate(
        criar_pipeline(),
        entradas,
        alvo,
        cv=validacao,
        scoring={
            "acuracia": "accuracy",
            "acuracia_balanceada": "balanced_accuracy",
            "f1_macro": "f1_macro",
        },
        n_jobs=1,
    )

    x_treino, x_teste, y_treino, y_teste = train_test_split(
        entradas,
        alvo,
        test_size=0.25,
        random_state=SEMENTE,
        stratify=alvo,
    )
    modelo = criar_pipeline()
    modelo.fit(x_treino, y_treino)
    previsto = modelo.predict(x_teste)

    PASTA_RESULTADOS.mkdir(exist_ok=True)
    metricas = {
        "amostras": int(len(dados)),
        "classes": {str(k): int(v) for k, v in alvo.value_counts().items()},
        "validacao_cruzada": {
            nome.replace("test_", ""): {
                "media": float(valores.mean()),
                "desvio_padrao": float(valores.std()),
            }
            for nome, valores in pontuacoes.items()
            if nome.startswith("test_")
        },
        "relatorio_holdout": classification_report(
            y_teste,
            previsto,
            output_dict=True,
            zero_division=0,
        ),
    }
    (PASTA_RESULTADOS / "metricas.json").write_text(
        json.dumps(metricas, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    pd.DataFrame(
        {
            "classe_real": y_teste.reset_index(drop=True),
            "classe_prevista": previsto,
        }
    ).to_csv(PASTA_RESULTADOS / "previsoes_holdout.csv", index=False)

    figura, eixo = plt.subplots(figsize=(7, 6), constrained_layout=True)
    ConfusionMatrixDisplay.from_predictions(
        y_teste,
        previsto,
        cmap="Blues",
        xticks_rotation=25,
        colorbar=False,
        ax=eixo,
    )
    eixo.set_title("Matriz de confusão — conjunto de teste")
    figura.savefig(PASTA_RESULTADOS / "matriz_confusao.png", dpi=160)
    plt.close(figura)

    print(json.dumps(metricas["validacao_cruzada"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
