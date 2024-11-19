import os
import wfdb
import numpy as np
# import pandas as pd
from scipy.signal import welch
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc, classification_report, confusion_matrix
import seaborn as sns
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def carrega_dados_mitbih(val):
    """
    Carrega os dados do banco de dados MIT-BIH e extrai segmentos de sinais.

    Retorno:
        X: Matriz com os segmentos de sinais.
        y: Vetor com as classes (0: normal, 1: anormal).
    """

    x1 = []
    y1 = []

    # Caminho para a pasta onde os arquivos do MIT-BIH estão localizados
    pasta = r'C:\Users\cydyq\Documents\Python\TCC\mitdb'

    if val == 1:
        records = ['100']
    else:
        records = [f[:-4] for f in os.listdir(pasta) if f.endswith('.dat')]

    for index, record in enumerate(records):
        print(f'Modelo atual[{index + 1}]: Record {record} de {len(records) - index} arquivos')

        # Carrega o sinal e as anotações do registro
        signals, fields = wfdb.rdsamp(record, pn_dir='mitdb/1.0.0')
        annotations = wfdb.rdann(record, 'atr', pn_dir='mitdb/1.0.0')

        # Extrai segmentos do sinal e as respectivas classes
        for i in range(0, len(signals) - 1000, 1000):
            segment = signals[i:i + 1000, 0]  # Seleciona o primeiro canal
            ann_idx = np.where((annotations.sample >= i) &
                               (annotations.sample < i + 1000))[0]

            if len(ann_idx) > 0:
                beat_type = 1 if annotations.symbol[ann_idx[0]] != 'N' else 0
                x1.append(segment)
                y1.append(beat_type)

    return np.array(x1), np.array(y1)


def extract_features(signals):
    """
    Extrai características do sinal de ECG.

    Args:
        signals: Segmento do sinal de ECG.

    Returns:
        features: Vetor com as características extraídas.
    """

    features = []

    # Características estatísticas
    features.extend([np.mean(signals), np.std(signals), np.max(signals), np.min(signals), np.median(signals)])

    # Características baseadas em frequência (utilizando o metodo de Welch)
    freqs, power = welch(signals, fs=360)  # Assumindo frequência de amostragem de 360 Hz
    features.extend([np.max(power), np.mean(power), np.argmax(power)])

    # Outras características (exemplos):
    # - Entropia de Shannon
    # - Fractal dimension
    # - Coeficientes de wavelet
    # - ...

    return np.array(features)


# Carregando os dados de todos os arquivos disponíveis
val1 = 10
X, y = carrega_dados_mitbih(val1)

# Extraindo as características
X_features = []
for signal in X:
    X_features.append(extract_features(signal))
X_features = np.array(X_features)

# Selecionando as 10 características mais relevantes utilizando o teste F
selector = SelectKBest(f_classif, k=10)
X_new = selector.fit_transform(X_features, y)

# Dividindo os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.2, random_state=42)

# Criando e treinando o modelo
model = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True)
model.fit(X_train, y_train)

# Fazendo previsões e avaliando o modelo
y_pred = model.predict(X_test)

######################### CURVA ROC #########################
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
print('Mostrando Matriz de confusão linha 121')
plt.show()
