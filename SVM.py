# Importando as bibliotecas necessárias
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# Gerando dados fictícios para ECG (substitua por seus dados reais)
# Aqui estou usando dados simulados. Quando você tiver os dados de ECG, carregue-os a partir de um arquivo .csv ou outra fonte.
# Por exemplo, `X = seus_dados_ecg` e `y = labels` (batimento normal ou arritmia)
X, y = datasets.make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=42)

# Dividindo os dados em treino e teste (80% treino e 20% teste)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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
