import os
import wfdb
import openpyxl
import numpy as np
from scipy.signal import welch
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


######################### Carregamento dos Dados #########################
def carrega_dados_mitbih(ini, fim):
    """
    Carrega os dados do banco de dados MIT-BIH e extrai segmentos de sinais.

    Retorno:
        X: Matriz com os segmentos de sinais.
        y: Vetor com as classes (0: normal, 1: anormal).
    """

    x1 = []
    y1 = []
    casos = None
    for index, record in enumerate(range(100, fim+1)):
        casos = index  + 1
        print(f"Processando registro {casos}: {record}")

        # Carrega o sinal e as anotações do registro
        signals, fields = wfdb.rdsamp(str(record), pn_dir='mitdb/1.0.0')
        annotations = wfdb.rdann(str(record), 'atr', pn_dir='mitdb/1.0.0')

        # Extrai segmentos do sinal e as respectivas classes
        for i in range(0, len(signals) - 1000, 1000):
            segment = signals[i:i + 1000, 0]  # Seleciona o primeiro canal
            ann_idx = np.where((annotations.sample >= i) &
                               (annotations.sample < i + 1000))[0]

            if len(ann_idx) > 0:
                beat_type = 1 if annotations.symbol[ann_idx[0]] != 'N' else 0
                x1.append(segment)
                y1.append(beat_type)

    if casos:
        print(f'Exames analisados: {casos}')

    return np.array(x1), np.array(y1)
"""
Carregamento dos Dados: Lê os dados de ECG do banco de dados MIT-BIH.
"""
X, y = carrega_dados_mitbih(100, 109)

######################### Pré-processamento dos Dados #########################
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
    freqs, power = welch(signals, fs=360)
    features.extend([np.max(power), np.mean(power), np.argmax(power)])

    """
    # Outras características (exemplos):
    # - Entropia de Shannon
    # - Fractal dimension
    # - Coeficientes de wavelet
    # - ..."""

    return np.array(features)

"""
Extração de Características: Calcula características relevantes dos sinais de ECG, 
como estatísticas e características de frequência.
"""
X_features = []
for signal in X:
    X_features.append(extract_features(signal))
X_features = np.array(X_features)

"""
Seleção de Características: Seleciona as características mais importantes para a classificação, 
utilizando o método SelectKBest.
"""
selector = SelectKBest(f_classif, k='all')
X_new = selector.fit_transform(X_features, y)

######################### Treinamento do Modelo #########################
"""
Divisão dos Dados: Divide os dados em conjuntos de treinamento e teste.
"""
X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.2, random_state=42)

"""
Treinamento: Treina um modelo de Support Vector Machine (SVM) para classificar 
os sinais de ECG como normais ou anormais com base nas características selecionadas.
"""
model = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True)
model.fit(X_train, y_train)

######################### Avaliação do Modelo #########################
"""
Predição: Utiliza o modelo treinado para fazer previsões sobre os dados de teste.
"""
y_pred = model.predict(X_test)

"""
Métricas de Desempenho: Calcula diversas métricas, como acurácia, precisão, recall e
F1-score, para avaliar a qualidade das previsões.
"""
acuracia = accuracy_score(y_test, y_pred) if accuracy_score(y_test, y_pred) == '0' else '1'
precisao = precision_score(y_test, y_pred) if precision_score(y_test, y_pred) != '0' else '1'
recall = recall_score(y_test, y_pred) if recall_score(y_test, y_pred) != '0' else '1'
f1 = f1_score(y_test, y_pred) if f1_score(y_test, y_pred) != '0' else '1'
print(f'Acurácia: {acuracia} \nPrecisao: {precisao} \nRecall: {recall} \nF1: {f1}\n')

# Criando um DataFrame com as métricas
metricas = pd.DataFrame({'Metrica': ['Acuracia', 'Precisao', 'Recall', 'F1-Score'],
                         'Valor': [acuracia, precisao, recall, f1]})

"""
Matriz de Confusão: Visualiza a distribuição das previsões corretas e incorretas para cada classe.
"""
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.savefig('matriz_confusao.png')

"""
Curva ROC: Avalia o poder de discriminação do modelo, mostrando a 
relação entre a taxa de verdadeiros positivos e a taxa de falsos positivos.
"""
y_prob = model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, color='darkorange', label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('curva_roc.png')

"""
Importância das Features: Identifica quais características contribuem mais para a classificação.
"""
importances = selector.scores_
indices = np.argsort(importances)[::-1]

# Plotando a importância das features
plt.figure(figsize=(12, 6))
plt.title("Feature Importance")
plt.bar(range(X_new.shape[1]), importances[indices], color="r", align="center")
plt.xticks(range(X_new.shape[1]), indices)
plt.xlim([-1, X_new.shape[1]])
plt.savefig('importancia_features.png')

"""
Relatório de Classificação: Gera um relatório detalhado com todas as
métricas de desempenho, incluindo suporte, confiança e macro/micro médias.
"""
# Relatório de classificação em formato de string
report_str = classification_report(y_test, y_pred, output_dict=True)

# Convertendo o dicionário para DataFrame
classificacao = pd.DataFrame(report_str).transpose()

# Convertendo os valores para porcentagem e formatando
classificacao = classificacao.map(lambda x: '{:.2f}%'.format(x*100))

# Salvando os resultados em um único arquivo Excel
with pd.ExcelWriter('relatorio_final.xlsx', engine='openpyxl') as writer:
    metricas.to_excel(writer, sheet_name='Métricas', index=False)
    classificacao.to_excel(writer, sheet_name='Relatório de Classificação')
    
print('Resultados salvos em relatorio_final.xlsx')
