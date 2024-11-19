# Importando as bibliotecas necessárias
import numpy as np
import matplotlib.pyplot as plt
import wfdb
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Substitua 'mitdb/100' pelo caminho do seu arquivo
record = wfdb.rdrecord(r'mitdb/101', channels=[0], sampfrom=0, sampto=65000)
annotations = wfdb.rdann(r'mitdb/100', 'atr', sampfrom=0, sampto=65000)

# Extraindo os sinais e anotações
signals = record.p_signal[:, 0]
annotations = annotations.symbol

# Dividindo os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(signals, annotations, test_size=0.2, random_state=42)

# Criando o modelo SVM com kernel radial (RBF)
svm_model = SVC(kernel='rbf', C=1.0, gamma='scale')

# Treinando o modelo
svm_model.fit(X_train, y_train)

# Fazendo previsões com o conjunto de teste
y_pred = svm_model.predict(X_test)

# Avaliando o desempenho
accuracy = accuracy_score(y_test, y_pred)
print(f"Acurácia: {accuracy * 100:.2f}%")

# Relatório de classificação e matriz de confusão
print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred))

print("\nMatriz de Confusão:")
print(confusion_matrix(y_test, y_pred))

# Plotando alguns resultados (opcional)
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred, cmap='coolwarm', s=30)
plt.title('Classificação SVM de Sinais ECG')
plt.xlabel('Característica 1')
plt.ylabel('Característica 2')
plt.show()










# # Carregando os dados
# X, y = load_mitbih_data(arquivos)
#
# # Extraindo características (simplificado)
# # Você pode adicionar mais características relevantes
# X_features = np.column_stack((
#     np.mean(X, axis=1),
#     np.std(X, axis=1),
#     np.max(X, axis=1),
#     np.min(X, axis=1)
# ))
#
# # Normalizando os dados
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X_features)
#
# # Dividindo em treino e teste
# X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
#
# # Criando e treinando o modelo SVM
# svm_model = SVC(kernel='rbf', C=1.0, gamma='scale')
# svm_model.fit(X_train, y_train)
#
# # Fazendo previsões
# y_pred = svm_model.predict(X_test)
#
# # Avaliando o desempenho
# print(f"Acurácia: {accuracy_score(y_test, y_pred) * 100:.2f}%")
# print("\nRelatório de Classificação:")
# print(classification_report(y_test, y_pred))
# print("\nMatriz de Confusão:")
# print(confusion_matrix(y_test, y_pred))
#
# # Visualizando os resultados
# plt.figure(figsize=(12, 4))
#
# plt.subplot(121)
# plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred, cmap='coolwarm', s=30)
# plt.title('Classificação SVM de Sinais ECG')
# plt.xlabel('Média do Sinal')
# plt.ylabel('Desvio Padrão do Sinal')
#
# plt.subplot(122)
# # Plotando um exemplo de sinal ECG
# idx = np.random.randint(len(X))
# plt.plot(X[idx])
# plt.title(f'Exemplo de Sinal ECG (Classe {y[idx]})')
# plt.xlabel('Amostras')
# plt.ylabel('Amplitude')
#
# plt.tight_layout()
# plt.show()