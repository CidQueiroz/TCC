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

    C = 10
    gamma = 0.1

Avaliação    
Métricas utilizadas:
    Para avaliar o desempenho do modelo, foram utilizadas as seguintes métricas:

    * Acurácia: 0.92 (92%)
    * Precisão: 0.88 (88%)
    * Recall: 0.95 (95%)
    * F1-score: 0.91 (91%)
    * AUC: 0.97

Interpretação dos resultados:
    O modelo de classificação de arritmias cardíacas apresentou um desempenho bastante satisfatório. A acurácia de 92% indica que o modelo classificou corretamente 92% dos casos, seja como normais ou anormais. Isso sugere que o modelo é capaz de generalizar bem para novos dados.

    Ao analisar a precisão de 88%, podemos concluir que quando o modelo previu que um sinal era anormal, ele estava correto em 88% dos casos. Isso indica uma baixa taxa de falsos positivos, ou seja, o modelo não tende a classificar erroneamente sinais normais como anormais.

    O recall de 95% indica que o modelo conseguiu identificar 95% de todos os casos de arritmia presentes nos dados. Isso é um resultado positivo, pois significa que o modelo é sensível em detectar a presença de arritmias.

    O F1-score de 91% representa um bom equilíbrio entre precisão e recall, indicando que o modelo tem um bom desempenho geral.

    Por fim, a AUC de 0.97 indica que o modelo tem uma excelente capacidade de discriminar entre sinais normais e anormais. Um valor próximo de 1 indica que o modelo é capaz de distinguir perfeitamente entre as duas classes.

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
    Clone o repositório: git clone https://github.com/CidQueiroz/TCC.git
    Instale as dependências: pip install -r requirements.txt
    Execute o script: python SVM.py

Contribuições
    Contribuições são bem-vindas! Você pode contribuir com este projeto através de:

    Issues: Reportando bugs ou sugerindo novas funcionalidades.
    Pull requests: Enviando código para melhorar o projeto.
