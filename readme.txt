Classificação de Arritmias Cardíacas utilizando SVM e o Banco de Dados MIT-BIH

Introdução
Objetivo do projeto:
    Este projeto tem como objetivo desenvolver e avaliar um modelo de aprendizado de máquina capaz de classificar sinais eletrocardiográficos (ECG) em normais e anormais, auxiliando no diagnóstico de arritmias cardíacas.

Breve explicação sobre o banco de dados MIT-BIH:
    O banco de dados MIT-BIH é um conjunto de dados amplamente utilizado na comunidade médica para pesquisa em eletrocardiografia. Ele contém registros de ECG de alta qualidade, anotados por cardiologistas, o que o torna ideal para o desenvolvimento e avaliação de algoritmos de classificação de arritmias.

Dataset
Descrição do dataset:
    O dataset utilizado neste projeto consiste em 205 registros do banco de dados MIT-BIH. Cada registro contém múltiplos canais de ECG, sendo utilizado apenas o primeiro canal neste trabalho. As classes presentes no dataset são:

    Normal: Sinais de ECG considerados normais.
    Anormal: Sinais de ECG com alguma forma de arritmia.

Pré-processamento realizado:
    Segmentação: Os sinais de ECG foram segmentados em segmentos de 48 amostras.
    Extração de características: Foram extraídas características estatísticas e de frequência de cada segmento, como média, desvio padrão, máximo, mínimo, mediana e potência espectral.

Extração de Características
    Foram extraídas as seguintes características dos sinais de ECG:

    Características estatísticas: Média, desvio padrão, máximo, mínimo e mediana.
    Características de frequência: Potência espectral máxima, média e frequência da potência máxima, calculadas utilizando o método de Welch.

Modelo
Algoritmo utilizado:
    Foi utilizado um classificador de Máquina de Vetores de Suporte (SVM) com kernel radial para realizar a classificação.

Hiperparâmetros escolhidos:
    Os hiperparâmetros do SVM foram ajustados utilizando [método de ajuste, e.g., GridSearchCV] para obter o melhor desempenho do modelo. Os valores finais dos hiperparâmetros foram:

    C = [valor]
    gamma = [valor]
    Avaliação
    Métricas utilizadas:
    Para avaliar o desempenho do modelo, foram utilizadas as seguintes métricas:

    Acurácia: Proporção de exemplos classificados corretamente.
    Precisão: Proporção de exemplos positivos classificados corretamente.
    Recall: Proporção de exemplos positivos corretamente identificados.
    F1-score: Média harmônica entre precisão e recall.
    Curva ROC: Visualização da relação entre a taxa de verdadeiros positivos e a taxa de falsos positivos.

Interpretação dos resultados:
    [Insira aqui uma interpretação concisa dos resultados obtidos. Por exemplo, comente sobre a acurácia do modelo, quais classes foram mais difíceis de classificar e quais características foram mais importantes para a classificação.]

Requisitos
    Para executar este código, são necessárias as seguintes bibliotecas Python:
        NumPy
        SciPy
        scikit-learn
        wfdb
        matplotlib
        pandas
        seaborn

Como Usar
    Clone o repositório: git clone https://[seu_repositorio]
    Instale as dependências: pip install -r requirements.txt (crie um arquivo requirements.txt com as bibliotecas necessárias)
    Execute o script: python seu_script.py

Contribuições
    Contribuições são bem-vindas! Você pode contribuir com este projeto através de:

    Issues: Reportando bugs ou sugerindo novas funcionalidades.
    Pull requests: Enviando código para melhorar o projeto.
