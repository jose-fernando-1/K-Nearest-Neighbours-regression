"""
k-NN Regressão para Previsão da Resistência à Compressão do Concreto

Este script implementa o algoritmo k-Nearest Neighbors (k-NN) para regressão,
aplicado ao dataset "Concrete Compressive Strength".

Utilizadas apenas as bibliotecas pandas e matplotlib.

O dataset contém as seguintes features:
- Cement (component 1) (kg in a m^3 mixture)
- Blast Furnace Slag (component 2) (kg in a m^3 mixture)
- Fly Ash (component 3) (kg in a m^3 mixture)
- Water  (component 4) (kg in a m^3 mixture)
- Superplasticizer (component 5) (kg in a m^3 mixture)
- Coarse Aggregate  (component 6) (kg in a m^3 mixture)
- Fine Aggregate (component 7) (kg in a m^3 mixture)
- Age (day)

Target:
- Concrete compressive strength (MPa, megapascals)

Obs.: Para melhor desempenho do k-NN (que utiliza distância euclidiana),
recomenda-se normalizar as features para que todas tenham a mesma escala.
"""

import matplotlib.pyplot as plt
import pandas as pd

url = "http://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/Concrete_Data.xls"
dataset = pd.read_excel(url)

# print(dataset.head())
# print(dataset.info())
# print(dataset.describe())

features = [
    'Cement (component 1)(kg in a m^3 mixture)',
    'Blast Furnace Slag (component 2)(kg in a m^3 mixture)',
    'Fly Ash (component 3)(kg in a m^3 mixture)',
    'Water  (component 4)(kg in a m^3 mixture)',
    'Superplasticizer (component 5)(kg in a m^3 mixture)',
    'Coarse Aggregate  (component 6)(kg in a m^3 mixture)',
    'Fine Aggregate (component 7)(kg in a m^3 mixture)',
    'Age (day)'
]

target = ['Concrete compressive strength(MPa, megapascals) ']

# Extração dos dados de entrada (X) e saída (y) a partir do DataFrame
X = dataset[features]
y = dataset[target]

# Normalização das Features (opcional, mas é recomendada)
X_norm = (X - X.min()) / (X.max() - X.min())

# 80% para treino e 20% para teste
tamanho_treino = int(0.8 * len(X_norm))
X_treino, X_teste = X_norm.iloc[:tamanho_treino], X_norm.iloc[tamanho_treino:]
y_treino, y_teste = y.iloc[:tamanho_treino], y.iloc[tamanho_treino:]

def distancia_euclidiana(x: pd.Series, y: pd.Series) -> float:
    return (sum((x[col] - y[col]) ** 2 for col in x.index)) ** 0.5

def erro_medio_absoluto(y_esperado, y_previsto) -> float:
    return (sum(abs(e - p) for e, p in zip(y_esperado, y_previsto))) / len(y_esperado)

def raiz_do_erro_quadratico_medio(y_esperado, y_previsto) -> float:
    return (sum((e - p) ** 2 for e, p in zip(y_esperado, y_previsto)) / len(y_esperado)) ** 0.5

def k_nearest_neighbors(X_treino: pd.DataFrame, y_treino: pd.DataFrame, X_teste: pd.DataFrame, k: int) -> list:
    '''
    Implementa o algoritmo k-NN para regressão.

    Para cada ponto de teste, calcula a distância euclidiana em relação
    a todos os pontos de treino, seleciona os k vizinhos mais próximos e
    retorna a média dos valores alvo desses vizinhos.

    Parâmetros:
    ------------
    X_treino : pd.DataFrame
        DataFrame com as features dos dados de treino.
    y_treino : pd.DataFrame
        DataFrame com os valores alvo dos dados de treino.
    X_teste : pd.DataFrame
        DataFrame com as features dos dados de teste.
    k : int
        Número de vizinhos a serem considerados.

    Retorna:
    ------------
    list
        Lista com os valores previstos para os dados de teste.
    '''
    previsoes = []
    # Itera por cada exemplo no conjunto de teste
    for _, ponto_teste in X_teste.iterrows():
        distancias = []
        # Calcula a distância entre o ponto de teste e cada ponto de treino.
        for i, ponto_treino in X_treino.iterrows():
            dist = distancia_euclidiana(ponto_teste, ponto_treino)
            # y_treino.iloc[i].iloc[0] extrai o valor alvo correspondente.
            distancias.append((dist, y_treino.iloc[i].iloc[0]))

        # Ordena as distâncias em ordem crescente e seleciona os k vizinhos mais próximos(k primeiros elementos da lista)
        distancias.sort(key=lambda x: x[0])
        vizinhos_mais_proximos = distancias[:k]

        # A previsão é a média dos valores dos k vizinhos
        previsao = sum(v[1] for v in vizinhos_mais_proximos) / k
        previsoes.append(previsao)

    return previsoes

# Aplicando o modelo para diferentes valores de k
k_values = [1, 3, 5, 7, 9]

for k in k_values:
    previsoes = k_nearest_neighbors(X_treino, y_treino, X_teste, k)
    ema = erro_medio_absoluto(y_teste.values.flatten(), previsoes)
    reqm = raiz_do_erro_quadratico_medio(y_teste.values.flatten(), previsoes)

    print(f"\nPara k = {k}:")
    print(f" - Erro Médio Absoluto (MAE): {ema:.4f}")
    print(f" - Raiz do Erro Quadrático Médio (RMSE): {reqm:.4f}")

    plt.figure(figsize=(10, 6))
    plt.scatter(range(len(y_teste)), y_teste, color='blue', label='Valores Reais', alpha=0.6)
    plt.scatter(range(len(previsoes)), previsoes, color='red', label='Previsões', alpha=0.6)
    plt.xlabel('Amostras')
    plt.ylabel('Concrete Compressive Strength')
    plt.title(f'k-NN (k={k}) - Valores Reais vs Previsões')
    plt.legend()
    plt.show()
