import MetaTrader5 as mt5
import pandas as pd
import pickle
import numpy as np

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

# Função para calcular as Bandas de Bollinger
def BollingerBands(df, n=20, k=2):
    sma = df['close'].rolling(n).mean()
    std = df['close'].rolling(n).std()
    upper_band = sma + (k * std)
    lower_band = sma - (k * std)
    return upper_band, lower_band

# Carregar o modelo de machine learning treinado
with open('C:\\Users\\Ikaro\\AppData\\Roaming\\MetaQuotes\\Terminal\\1D66438650AC158D2CF94C521FB08859\\MQL5\\Experts\\gpt\\modelo_rf.pkl', 'rb') as file:
    model = pickle.load(file)

# Inicializa a conexão com o MetaTrader 5
if not mt5.initialize():
    print("Falha ao inicializar o MT5")
    mt5.shutdown()
    exit()

# Função para obter os dados mais recentes
def obter_dados_ao_vivo():
    # Obtém os dados de preços mais recentes para o par BTCUSDT no timeframe de 15 minutos (M15)
    rates = mt5.copy_rates_from_pos("BTCUSDT", mt5.TIMEFRAME_M15, 0, 21)
    if rates is None or len(rates) < 21:
        print("Erro ao obter os dados de mercado")
        return None
    data = pd.DataFrame(rates)
    
    # Adiciona os indicadores técnicos usados no modelo
    data['Momentum'] = data['close'].pct_change(5)  # Retorno percentual nos últimos 5 períodos
    data['Upper_BB'], data['Lower_BB'] = BollingerBands(data, 20)
    data['SMA'] = data['close'].rolling(20).mean()  # Média móvel simples de 20 períodos
    data['RSI'] = RSI(data, 14)  # RSI com janela de 14 períodos
    data['CCI'] = CCI(data, 20)  # CCI com janela de 20 períodos
    data['Avg_Volume'] = data['tick_volume'].rolling(20).mean()

    # Remover NaN criados pelos cálculos dos indicadores
    return data.dropna().tail(1)

# Coletar os dados ao vivo
dados_ao_vivo = obter_dados_ao_vivo()

# Verifica se os dados foram coletados corretamente
if dados_ao_vivo is None or dados_ao_vivo.empty:
    print("Falha ao obter os dados ao vivo")
    mt5.shutdown()
    exit()

# Fazer a previsão usando o modelo com todas as features usadas no treinamento
X_live = dados_ao_vivo[['Avg_Volume', 'Momentum', 'Upper_BB', 'Lower_BB', 'RSI', 'SMA', 'CCI', 'open', 'high', 'low', 'close', 'tick_volume']]

# Fazer a previsão
sinal = model.predict(X_live)

# Executar a ordem de acordo com o sinal do modelo
if sinal == 1:
    print("Sinal de Compra - Executando operação")
    # Executar uma ordem de compra
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": "BTCUSDT",
        "volume": 0.2,  # Define o volume que você deseja negociar
        "type": mt5.ORDER_TYPE_BUY,
        "price": mt5.symbol_info_tick("BTCUSDT").ask,
        "deviation": 20,
        "magic": 234000,
        "comment": "Trade de compra automatizado",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    result = mt5.order_send(request)
    print(result)
elif sinal == 0:
    print("Sinal de Venda - Executando operação")
    # Executar uma ordem de venda
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": "BTCUSDT",
        "volume": 0.2,  # Define o volume que você deseja negociar
        "type": mt5.ORDER_TYPE_SELL,
        "price": mt5.symbol_info_tick("BTCUSDT").bid,
        "deviation": 20,
        "magic": 234000,
        "comment": "Trade de venda automatizado",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    result = mt5.order_send(request)
    print(result)
else:
    print("Sinal desconhecido")

# Finaliza a conexão com o MT5
mt5.shutdown()
