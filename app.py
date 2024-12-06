import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import wfdb
import datetime
import numpy as np
from scipy.signal import welch
from reportlab.lib import colors
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from reportlab.lib.pagesizes import letter
from reportlab.lib import units
from reportlab.platypus import SimpleDocTemplate, Table, Image, Paragraph, TableStyle, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle


def carrega_e_processa_dados(ini, fim):
    x1 = []
    y1 = []
    casos = None
    for index, record in enumerate(range(100, fim+1)):
        try:
            casos = index  + 1
            print(f"Processando registro {casos}: {record}")

            signals, fields = wfdb.rdsamp(str(record), pn_dir='mitdb/1.0.0')
            annotations = wfdb.rdann(str(record), 'atr', pn_dir='mitdb/1.0.0')

            for i in range(0, len(signals) - 1000, 1000):
                segment = signals[i:i + 1000, 0]  # Seleciona o primeiro canal
                ann_idx = np.where((annotations.sample >= i) &
                                (annotations.sample < i + 1000))[0]

                if len(ann_idx) > 0:
                    beat_type = 1 if annotations.symbol[ann_idx[0]] != 'N' else 0
                    x1.append(segment)
                    y1.append(beat_type)
        except:
            # st.warning(f'Erro ao acessar arquivo: mitdb/1.0.0/{record}')
            casos = casos - 1

    if casos:
        st.info(f'Exames analisados: {casos}')

    return np.array(x1), np.array(y1)

def extract_features(signals):
    features = []
    features.extend([np.mean(signals), np.std(signals), np.max(signals), np.min(signals), np.median(signals)])

    # Características baseadas em frequência (utilizando o metodo de Welch)
    freqs, power = welch(signals, fs=360)
    features.extend([np.max(power), np.mean(power), np.argmax(power)])

    return np.array(features)

def gera_novas_features(x1, y1):
    X_features = []
    for signal in x1:
        X_features.append(extract_features(signal))
    X_features = np.array(X_features)
    selector1 = SelectKBest(f_classif, k='all')
    X_new = selector1.fit_transform(X_features, y1)
    
    return X_new, selector1

def treina_e_avalia_modelo(X_train, X_test, y_train, y_test):
    model = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True)
    model.fit(X_train, y_train)
    y_pred1 = model.predict(X_test)

    return model, y_pred1

def relatorios(y_test, y_pred):
    acuracia = round(float(accuracy_score(y_test, y_pred) if accuracy_score(y_test, y_pred) == '0' else '1'), 1) * 100
    precisao = round(float(precision_score(y_test, y_pred) if precision_score(y_test, y_pred) != '0' else '1'), 1) * 100
    recall = round(float(recall_score(y_test, y_pred) if recall_score(y_test, y_pred) != '0' else '1'), 1) * 100
    f1 = round(float(f1_score(y_test, y_pred) if f1_score(y_test, y_pred) != '0' else '1'), 1) * 100
    
    # Criando um DataFrame com as métricas
    metricas = pd.DataFrame({'Metrica': ['Acuracia', 'Precisao', 'Recall', 'F1-Score'],
                            'Valor': [f'{acuracia:.2f}%', f'{precisao:.2f}%', f'{recall:.2f}%', f'{f1:.2f}%']})
    metricas['Valor'] = metricas['Valor'].astype(str)
    
    # Relatório de classificação em formato de string
    report_str = classification_report(y_test, y_pred, output_dict=True)

    # Convertendo o dicionário para DataFrame
    classificacao = pd.DataFrame(report_str).transpose()

    # Separando a coluna 'support'
    support_col = classificacao['support']

    # Convertendo os valores para porcentagem e formatando (exceto 'support')
    classificacao1 = classificacao.drop('support', axis=1).map(lambda x: '{:.2f}%'.format(x*100))
    classificacao1['support'] = support_col
    
    return metricas, classificacao1

# Título da aplicação
st.title("Análise de Sinais de ECG")

# Sidebar para opções do usuário
with st.sidebar:
    st.header("Configurações")
    inicio = st.number_input("Registro Inicial", min_value=100, max_value=200, value=100)
    fim = st.number_input("Registro Final", min_value=100, max_value=234, value=109)

# Botão para iniciar a análise
if st.button("Iniciar Análise"):

    data_hora = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    nome_arquivo = f"relatorio_svm_{data_hora}.pdf"

    doc = SimpleDocTemplate(nome_arquivo, pagesize=letter)
    elements = []

    st.info('Carregando dados!')
    X, y = carrega_e_processa_dados(inicio, fim)

    X_new, selector = gera_novas_features(X, y)

    X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.2, random_state=42)
    model, y_pred = treina_e_avalia_modelo(X_train, X_test, y_train, y_test)
    
    metricas, classificacao = relatorios(y_test, y_pred)

    page_width = doc.pagesize[0]
    # Criando um estilo para as células da tabela
    style = TableStyle([
        ('ALIGN', (1,1), (-1,-1), 'RIGHT'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TEXTCOLOR', (0,0), (-1,0), colors.black),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,0), 12),
        ('BOTTOMMARKED', (0,0), (-1,0), 1, colors.black),
        ('BOX', (0,0), (-1,-1), 0.25, colors.black),
        ('GRID', (0,0), (-1,-1), 0.25, colors.gray),
    ])

    # Configurando as metricas de desempenho
    st.subheader("Métricas de Desempenho")
    st.dataframe(metricas)

    tab_met = [list(metricas.columns)] + metricas.values.tolist()
    t = Table(tab_met)
    t.setStyle(style)

    metricas_texto = "O desempenho do modelo é avaliado utilizando métricas como acurácia, precisão, revocação e F1-score."
    styles = getSampleStyleSheet()
    titleStyle = ParagraphStyle('Title', fontSize=14, alignment=1)
    elements.append(Spacer(height=units.inch * 0.2, width=units.inch * 0.2))
    elements.append(Paragraph('Métricas', titleStyle))
    elements.append(Paragraph('', titleStyle))
    elements.append(Paragraph(metricas_texto))
    elements.append(t)
    t.setStyle(style)
    elements.append(PageBreak())
    
    # Configurando a matriz de confusao
    st.subheader("Matriz de Confusão")
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    # plt.title('Matriz de Confusão')
    plt.savefig('matriz_confusao.png')
    st.pyplot(plt)

    matriz_confusao_texto = "Matriz de Confusão: apresenta a distribuição dos resultados previstos versus os reais."
    img_matriz_confusao = Image('matriz_confusao.png')
    img_width, img_height = img_matriz_confusao.drawWidth, img_matriz_confusao.drawHeight
    aspect_ratio = img_height / img_width
    img_matriz_confusao.drawWidth = page_width
    img_matriz_confusao.drawHeight = page_width * aspect_ratio
    titleStyle = ParagraphStyle('Title', fontSize=14, alignment=1)
    elements.append(Paragraph('Matriz de Confusão', titleStyle))
    elements.append(Paragraph('', titleStyle))
    elements.append(Paragraph(matriz_confusao_texto))
    elements.append(img_matriz_confusao)
    elements.append(PageBreak())

    # Configurando a curva roc
    st.subheader("Curva ROC")
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, color='darkorange', label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Taxa de Falso Positivo')
    plt.ylabel('Taxa de Verdadeiro Positivo')
    # plt.title('Receiver operating characteristic')
    plt.legend(loc="center")
    plt.savefig('curva_roc.png')
    st.pyplot(plt)

    curva_roc_ctexto = "Curva ROC: indica a capacidade do modelo de discriminar entre classes."
    img_curva_roc = Image('curva_roc.png')
    img_width, img_height = img_curva_roc.drawWidth, img_curva_roc.drawHeight
    aspect_ratio = img_height / img_width
    img_curva_roc.drawWidth = page_width
    img_curva_roc.drawHeight = page_width * aspect_ratio
    titleStyle = ParagraphStyle('Title', fontSize=14, alignment=1)
    elements.append(Paragraph('Curva ROC', titleStyle))
    elements.append(Paragraph('', titleStyle))
    elements.append(Paragraph(curva_roc_ctexto))
    elements.append(img_curva_roc)
    elements.append(PageBreak())
    
    # Configurando a importância das features
    st.subheader("Importância das Features")
    importances = selector.scores_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(12, 6))
    
    # plt.title("Feature Importance")
    plt.bar(range(X_new.shape[1]), importances[indices], color="r", align="center")
    plt.xticks(range(X_new.shape[1]), indices)
    plt.xlim([-1, X_new.shape[1]])
    plt.savefig('importancia_features.png')
    st.pyplot(plt)  
    
    importancia_features_texto = "Importância das Features: classifica a relevância de cada característica na previsão."
    img_importancia_features = Image('importancia_features.png')
    img_width, img_height = img_importancia_features.drawWidth, img_importancia_features.drawHeight
    aspect_ratio = img_height / img_width
    img_importancia_features.drawWidth = page_width
    img_importancia_features.drawHeight = page_width * aspect_ratio
    elements.append(Paragraph('Importancia Features', titleStyle))
    elements.append(Paragraph('', titleStyle))
    elements.append(Paragraph(importancia_features_texto))
    elements.append(img_importancia_features)
    elements.append(PageBreak())

    # Configurando o relatorio de classificação
    st.subheader("Relatório de Classificação")
    st.dataframe(classificacao)
    
    relat_classif = [list(classificacao.columns)] + classificacao.values.tolist()
    data = classificacao.reset_index().values.tolist()
    t = Table(data)
    t.setStyle(style)

    relatorio_class_texto = "Gera um relatório detalhado com todas as métricas de desempenho, incluindo suporte, confiança e macro/micro médias."
    styles = getSampleStyleSheet()
    titleStyle = ParagraphStyle('Title', fontSize=14, alignment=1)
    elements.append(Paragraph('Relatório de Classificação', titleStyle))
    elements.append(Paragraph('', titleStyle))
    elements.append(Paragraph(relatorio_class_texto))
    elements.append(t)
    t.setStyle(style)
   
    doc.build(elements)

    # Salvando os resultados em um único arquivo Excel
    with pd.ExcelWriter('relatorio_final.xlsx', engine='openpyxl') as writer:
        metricas.to_excel(writer, sheet_name='Métricas', index=False)
        classificacao.to_excel(writer, sheet_name='Relatório de Classificação')

    st.download_button(
        label="Baixar Relatório Completo em PDF",
        data=open("relatorio.pdf", "rb").read(),
        file_name="relatorio.pdf",
        mime="application/pdf"
    )