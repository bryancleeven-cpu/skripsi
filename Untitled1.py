!pip install seaborn

import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier

# ========== KONFIGURASI DASAR ==========
st.set_page_config(page_title="Sentiment Analysis App", page_icon="💬", layout="wide")

# ========== CUSTOM STYLE ==========
st.markdown("""
    <style>
        /* Background umum */
        .main {
            background-color: var(--background-color);
            padding: 20px;
        }

        /* Box metric (card kecil di atas) */
        div[data-testid="stMetricValue"], div[data-testid="stMetricLabel"] {
            color: var(--text-color) !important;
        }

        /* Tambahan styling card */
        [data-testid="stMetric"] {
            background-color: var(--box-bg);
            border-radius: 10px;
            padding: 15px;
            box-shadow: 0 0 8px rgba(0,0,0,0.05);
        }

        /* Sentimen hasil prediksi */
        .result-box {
            font-size: 20px;
            font-weight: bold;
            text-align: center;
            border-radius: 10px;
            padding: 15px;
            margin-top: 10px;
        }
        .positive {background-color: #d4edda; color: #155724;}
        .negative {background-color: #f8d7da; color: #721c24;}
        .neutral {background-color: #fff3cd; color: #856404;}

        /* Warna mengikuti tema */
        :root {
            --background-color: #0E1117;
            --text-color: #ffffff;
            --box-bg: #1E1E1E;
        }

        @media (prefers-color-scheme: light) {
            :root {
                --background-color: #f8f9fa;
                --text-color: #000000;
                --box-bg: #ffffff;
            }
        }
    </style>
""", unsafe_allow_html=True)

# ========== FUNGSI PEMROSESAN ==========
@st.cache_data
def load_data():
    return pd.read_csv("Hasil_Labelling_Dataset_Lokal.csv", delimiter=';', low_memory=False)

def load_model(dataset, model_type='SVM'):
    X = dataset['stemmed_text']
    y = dataset['Sentiment']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    vectorizer = CountVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    if model_type == 'SVM':
        model = SVC(kernel='linear', random_state=42)
    elif model_type == 'Naive Bayes':
        model = MultinomialNB()
    elif model_type == 'Random Forest':
        model = RandomForestClassifier(random_state=42, n_estimators=100)

    model.fit(X_train_vec, y_train)
    y_pred = model.predict(X_test_vec)
    accuracy = accuracy_score(y_test, y_pred)
    return model, vectorizer, accuracy, y_test, y_pred

# ========== SIDEBAR ==========
st.sidebar.header("⚙️ Pengaturan Analisis")
model_choice = st.sidebar.selectbox("Pilih Model:", ["SVM", "Naive Bayes", "Random Forest"])
st.sidebar.info("Aplikasi ini menggunakan **dataset hasil scraping platform X** untuk analisis sentimen otomatis.")

# ========== KONTEN UTAMA ==========
st.title("💬 Sentiment Analysis Web App ")
st.caption("Analisis sentimen otomatis menggunakan algoritma Machine Learning")

data = load_data()
model, vectorizer, accuracy, y_test, y_pred = load_model(data, model_choice)

# ========== METRIC (CARD ATAS) ==========
col1, col2, col3 = st.columns(3)
sentiment_count = data['Sentiment'].value_counts()
col1.metric("Jumlah Data", f"{len(data)} Tweet")
col2.metric("Akurasi Model", f"{accuracy*100:.2f}%")
col3.metric("Jumlah Kategori", f"{len(sentiment_count)} Sentimen")

# ========== VISUALISASI (TAB) ==========
tab1, tab2, tab3 = st.tabs(["📊 Distribusi Sentimen", "📈 Confusion Matrix", "🧾 Laporan Klasifikasi"])

with tab1:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Pie Chart")
        fig_pie, ax_pie = plt.subplots(figsize=(6, 6))
        ax_pie.pie(sentiment_count, labels=sentiment_count.index, autopct='%1.1f%%', startangle=90,
                   colors=['#66b3ff', '#ff6666', '#99ff99'])
        ax_pie.axis('equal')
        st.pyplot(fig_pie)
    with c2:
        st.subheader("Bar Chart")
        fig_bar, ax_bar = plt.subplots(figsize=(6, 5))
        sns.barplot(x=sentiment_count.index, y=sentiment_count.values, palette='pastel', ax=ax_bar)
        ax_bar.set_title('Distribusi Sentimen')
        st.pyplot(fig_bar)

with tab2:
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig_cm, ax_cm = plt.subplots(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Negatif', 'Netral', 'Positif'],
                yticklabels=['Negatif', 'Netral', 'Positif'],
                ax=ax_cm, square=True)
    ax_cm.set_xlabel('Predicted')
    ax_cm.set_ylabel('Actual')
    st.pyplot(fig_cm)

with tab3:
    st.subheader("Classification Report")
    report = classification_report(y_test, y_pred, target_names=['Negatif', 'Netral', 'Positif'], output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose())

# ========== INPUT USER ==========
st.markdown("---")
st.subheader("💡 Coba Analisis Sentimen Sendiri")
user_input = st.text_area("Masukkan tweet atau kalimat:")

if user_input:
    # Preprocessing input (untuk memastikan data sesuai dengan format pelatihan)
    user_input_vec = vectorizer.transform([user_input])

    # Prediksi dari model
    prediction = model.predict(user_input_vec)[0]

    # Tampilkan hasil prediksi sesuai dengan kelas sentimen
    if prediction == "Positif":
        st.markdown(f"<div class='result-box positive'>😊 Sentimen: <b>{prediction}</b></div>", unsafe_allow_html=True)
    elif prediction == "Negatif":
        st.markdown(f"<div class='result-box negative'>😞 Sentimen: <b>{prediction}</b></div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='result-box neutral'>😐 Sentimen: <b>{prediction}</b></div>", unsafe_allow_html=True)
