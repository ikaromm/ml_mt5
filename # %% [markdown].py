# %% [markdown]
# # Imports

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import MetaTrader5 as mt5

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, precision_recall_curve

from imblearn.over_sampling import SMOTE

# %% [markdown]
# # Download candles

# %%
# Inicializa a conexão com o MetaTrader 5
if not mt5.initialize():
    print("Falha ao inicializar o MT5")
    mt5.shutdown()

# Escolha o símbolo e o período de tempo para coletar dados
symbol = "BTCUSDT"
timeframe = mt5.TIMEFRAME_M15  # Gráfico de 1 hora
bars = mt5.copy_rates_from_pos(symbol, timeframe, 0, 10000)  # Últimos 1000 candles

# Converte os dados em um DataFrame do pandas
data = pd.DataFrame(bars)
data['time'] = pd.to_datetime(data['time'], unit='s')

# Finaliza a conexão com o MT5
mt5.shutdown()


# %%
display(data)

# %% [markdown]
# # Funcs

# %%
# Função para calcular o CCI
def CCI(df, n):
    tp = (df['high'] + df['low'] + df['close']) / 3  # Preço típico
    sma = tp.rolling(n).mean()  # Média móvel simples
    mad = lambda x: np.mean(np.abs(x - np.mean(x)))  # Calculando o MAD manualmente
    rolling_mad = tp.rolling(n).apply(mad, raw=True)  # Aplicando o MAD manual
    cci = (tp - sma) / (0.015 * rolling_mad)  # Fórmula do CCI
    return cci

# Função para calcular RSI
def RSI(df, n):
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(n).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(n).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi
  
def BollingerBands(df, n=20, k=2):
    sma = df['close'].rolling(n).mean()
    std = df['close'].rolling(n).std()
    upper_band = sma + (k * std)
    lower_band = sma - (k * std)
    return upper_band, lower_band

# %% [markdown]
# # Organizing data

# %%
data['Momentum'] = data['close'].pct_change(5)  # Retorno percentual nos últimos 5 períodos
data['Upper_BB'], data['Lower_BB'] = BollingerBands(data, 20)
data['SMA'] = data['close'].rolling(20).mean()  # Média móvel simples de 20 períodos
data['RSI'] = RSI(data, 14)  # RSI com janela de 14 períodos
data['CCI'] = CCI(data, 20)  # CCI com janela de 20 períodos
# data['label'] = (data['close'].shift(-1) > data['close']).astype(int)  # Label baseado na variação de preço
data['Avg_Volume'] = data['tick_volume'].rolling(20).mean()

# Remover NaN criados pela função shift e CCI
data.dropna(inplace=True)


# Definir threshold para variação percentual futura
threshold = 0.0003  # Ajuste esse valor conforme necessário (0.5% de variação)

# Criar a label com base na variação percentual futura
data['future_price'] = data['close'].shift(-1)
data['price_change'] = (data['future_price'] - data['close']) / data['close']

# Se a variação for maior que o threshold, o label será 1 (compra), caso contrário, será 0 (venda)
data['label'] = (data['price_change'] > threshold).astype(int)

# Remover NaN criados pela função shift e indicadores
data.dropna(inplace=True)



X = data[['Avg_Volume','Momentum','Upper_BB','Lower_BB','RSI', 'SMA', 'CCI', 'open', 'high', 'low', 'close', 'tick_volume']]  # Features
y = data['label']  # Labels

# %%
data.label.value_counts()

# %% [markdown]
# # Randon Forest

# %%
split_point = int(len(X) * 0.7)
X_train, X_test = X[:split_point], X[split_point:]
y_train, y_test = y[:split_point], y[split_point:]

# Verificar se há desbalanceamento nas classes
print("Distribuição das classes antes do SMOTE:")
print(y_train.value_counts())

# Aplicar o SMOTE para balancear as classes
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Verificar a distribuição após o SMOTE
print("Distribuição das classes após o SMOTE:")
print(pd.Series(y_train_resampled).value_counts())

# Criar o modelo Random Forest com hiperparâmetros ajustados
model = RandomForestClassifier(
    n_estimators=200,  # Aumentando o número de estimadores
    max_depth=10,  # Limitando a profundidade da árvore
    min_samples_split=10,  # Impedindo splits muito pequenos
    max_features='sqrt',  # Selecionando sqrt do número total de features
    random_state=42
)

# Ajustar o modelo com os dados balanceados
model.fit(X_train_resampled, y_train_resampled)

# Prever no conjunto de teste
y_pred = model.predict(X_test)


# %%

# Avaliar o desempenho do modelo
accuracy = accuracy_score(y_test, y_pred)
print(f"Acurácia do modelo: {accuracy:.2f}")

# Mostrar métricas adicionais
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Validação cruzada com 5 dobras
cv_scores = cross_val_score(model, X, y, cv=5)
print(f"Acurácias nas 5 dobras: {cv_scores}")
print(f"Acurácia média na validação cruzada: {np.mean(cv_scores):.2f}")

# Calcular AUC-ROC
y_pred_proba = model.predict_proba(X_test)[:, 1]
roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f"AUC-ROC: {roc_auc:.2f}")

# Calcular precision-recall
precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)

# Plotar a curva precision-recall
plt.plot(recall, precision, marker='.')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.show()

# Importância das features
importances = model.feature_importances_
feature_names = X.columns

# Plotar a importância das features
plt.barh(feature_names, importances)
plt.xlabel('Importância')
plt.title('Importância das Features')
plt.show()


# %%
import pickle

# Salvar o modelo treinado
with open('modelo_rf.pkl', 'wb') as file:
    pickle.dump(model, file)



