"""
============================================================================
Streamlit Dashboard - Bank Transaction Pipeline
============================================================================
Proje sonuclarini gorsel olarak sunar:
- Model performans karsilastirma
- Feature importance
- EDA gorselleri
- Zaman serisi trendleri
- Veri dagilim grafikleri
============================================================================
"""

import streamlit as st
import pandas as pd
import json
import os
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ==================== SAYFA AYARLARI ====================
st.set_page_config(
    page_title="Bank Transaction Pipeline Dashboard",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== CUSTOM CSS ====================
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1a5276;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(135deg, #1a5276 0%, #2980b9 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #2980b9;
        margin-bottom: 1rem;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #1a5276;
        border-bottom: 3px solid #2980b9;
        padding-bottom: 0.5rem;
        margin: 2rem 0 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

OUTPUT_PATH = '/app/output'
ML_PATH = f'{OUTPUT_PATH}/ml'
EDA_PATH = f'{OUTPUT_PATH}/eda'


def load_ml_results():
    """ML sonuclarini ve gorsel verilerini yukle"""
    data = {}
    try:
        with open(f'{ML_PATH}/model_results.json', 'r') as f:
            data['results'] = json.load(f)
        with open(f'{ML_PATH}/feature_importance.json', 'r') as f:
            data['importance'] = json.load(f)
        data['plot_data'] = pd.read_csv(f'{ML_PATH}/plot_data.csv')
        return data
    except Exception:
        return None


def load_eda_data():
    """EDA verilerini yukle"""
    data = {}
    try:
        data['domain'] = pd.read_csv(f'{EDA_PATH}/domain_stats.csv')
    except Exception:
        data['domain'] = None
    try:
        data['location'] = pd.read_csv(f'{EDA_PATH}/location_stats.csv')
    except Exception:
        data['location'] = None
    try:
        data['monthly'] = pd.read_csv(f'{EDA_PATH}/monthly_stats.csv')
    except Exception:
        data['monthly'] = None
    try:
        with open(f'{EDA_PATH}/eda_results.json', 'r') as f:
            data['stats'] = json.load(f)
    except Exception:
        data['stats'] = None
    return data


# ==================== HEADER ====================
st.markdown('<div class="main-header">🏦 Bank Transaction Big Data Pipeline Dashboard</div>',
            unsafe_allow_html=True)

# ==================== SIDEBAR ====================
st.sidebar.title("📊 Navigation")
page = st.sidebar.radio("Sayfa Sec:", [
    "🏠 Genel Bakis",
    "📊 EDA Analizi",
    "🤖 ML Model Karsilastirma",
    "📈 Zaman Serisi",
    "🔍 Feature Importance"
])

# ==================== GENEL BAKIS ====================
if page == "🏠 Genel Bakis":
    st.markdown('<div class="section-header">Pipeline Ozeti</div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    eda = load_eda_data()

    if eda.get('stats'):
        stats = eda['stats']
        col1.metric("Toplam Satir", f"{stats.get('total_rows', 'N/A'):,}")
        col2.metric("Domain Sayisi", stats.get('unique_domains', 'N/A'))
        col3.metric("Lokasyon Sayisi", stats.get('unique_locations', 'N/A'))
        col4.metric("Event Tipleri", stats.get('unique_event_types', 'N/A'))

        st.markdown("### Sayisal Degisken Istatistikleri")
        if 'numeric' in stats:
            num_df = pd.DataFrame(stats['numeric']).T
            st.dataframe(num_df, use_container_width=True)
    else:
        st.warning("EDA verileri henuz mevcut degil. Pipeline'i calistirin.")

    st.markdown("### Pipeline Mimarisi")
    st.markdown("""
    ```
    CSV Dataset → Kafka Producer → Kafka Broker → Spark Structured Streaming
                                                          ↓
                                                    Bronze Layer (Raw)
                                                          ↓
                                                    Silver Layer (Clean)
                                                          ↓
                                                    Gold Layer (Aggregated)
                                                          ↓
                                                Feature Engineering → ML Models → MLflow
                                                                                    ↓
                                                                              Dashboard
    ```
    """)

# ==================== EDA ====================
elif page == "📊 EDA Analizi":
    st.markdown('<div class="section-header">Kesifsel Veri Analizi (EDA)</div>', unsafe_allow_html=True)
    eda = load_eda_data()

    if eda.get('domain') is not None:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Domain Dagilimi")
            fig = px.pie(eda['domain'], values='count', names='domain',
                         color_discrete_sequence=px.colors.qualitative.Set2,
                         title='Domain Bazli Islem Dagilimi')
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("#### Domain Bazli Ortalama Deger")
            fig = px.bar(eda['domain'], x='avg_value', y='domain', orientation='h',
                         color='avg_value', color_continuous_scale='Blues',
                         title='Domain Bazli Ortalama Islem Degeri')
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

        if eda.get('location') is not None:
            st.markdown("#### Top 15 Lokasyon (Toplam Deger)")
            fig = px.bar(eda['location'], x='total_value', y='location', orientation='h',
                         color='total_value', color_continuous_scale='Viridis',
                         title='Top 15 Lokasyon - Toplam Islem Degeri')
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)

        # EDA gorselleri
        st.markdown("#### EDA Gorselleri")
        eda_images = [f for f in os.listdir(EDA_PATH) if f.endswith('.png')] if os.path.exists(EDA_PATH) else []
        if eda_images:
            cols = st.columns(2)
            for i, img in enumerate(sorted(eda_images)):
                with cols[i % 2]:
                    st.image(f'{EDA_PATH}/{img}', caption=img.replace('.png','').replace('_',' '), use_column_width=True)
    else:
        st.warning("EDA verileri henuz mevcut degil.")

# ==================== ML KARSILASTIRMA ====================
elif page == "🤖 ML Model Karsilastirma":
    st.markdown('<div class="section-header">ML Model Karsilastirmasi</div>', unsafe_allow_html=True)
    ml_data = load_ml_results()

    if ml_data:
        # Yeni JSON formatina gore veriyi hazirla
        results = ml_data['results']
        df_list = []
        for name, metrics in results.items():
            df_list.append({
                "model_name": name,
                "rmse": metrics.get("rmse", 0),
                "mae": metrics.get("mae", 0),
                "r2": metrics.get("r2", 0)
            })
        df = pd.DataFrame(df_list)

        st.markdown("#### Model Metrikleri")
        st.dataframe(df[['model_name','rmse','mae','r2']].style.highlight_min(
            subset=['rmse','mae'], color='#a8e6cf').highlight_max(
            subset=['r2'], color='#a8e6cf'), use_container_width=True)

        # Grafik Karsilastirmasi
        st.markdown("#### Performans Karsilastirmasi")
        fig = make_subplots(rows=1, cols=2, subplot_titles=('RMSE (dusuk=iyi)', 'R² (yuksek=iyi)'))
        fig.add_trace(go.Bar(x=df['model_name'], y=df['rmse'], name='RMSE', marker_color='#e74c3c'), row=1, col=1)
        fig.add_trace(go.Bar(x=df['model_name'], y=df['r2'], name='R²', marker_color='#2ecc71'), row=1, col=2)
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        # Zorunlu Analiz Grafikleri
        st.markdown("#### 🔍 Detaylı Regresyon Analizi")
        plot_data = ml_data['plot_data']
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(px.scatter(plot_data, x="actual", y="predicted", trendline="ols", title="Gerçek vs Tahmin"), use_container_width=True)
        with col2:
            st.plotly_chart(px.histogram(plot_data, x="residual", nbins=50, title="Residual (Artık) Dağılımı"), use_container_width=True)
        # Gorseller
        ml_images = [f for f in os.listdir(ML_PATH) if f.endswith('.png')] if os.path.exists(ML_PATH) else []
        if ml_images:
            st.markdown("#### Detaylı Görseller")
            cols = st.columns(2)
            for i, img in enumerate(sorted(ml_images)):
                with cols[i % 2]:
                    st.image(f'{ML_PATH}/{img}', caption=img.replace('.png','').replace('_',' '), use_column_width=True)
    else:
        st.warning("⚠️ ML sonuçları henüz mevcut değil. Lütfen eğitimi tamamlayın.")


# ==================== ZAMAN SERISI ====================
elif page == "📈 Zaman Serisi":
    st.markdown('<div class="section-header">Günlük İşlem Trend Analizi (31 Gün)</div>', unsafe_allow_html=True)
    
    try:
        daily_df = pd.read_csv(f'{EDA_PATH}/daily_stats.csv')
        
        col1, col2 = st.columns(2)
        with col1:
            fig = px.line(daily_df, x='day', y='total_value', markers=True,
                          title='Günlük Toplam İşlem Değeri',
                          color_discrete_sequence=['#e74c3c'])
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.line(daily_df, x='day', y='transaction_count', markers=True,
                          title='Günlük İşlem Sayısı',
                          color_discrete_sequence=['#3498db'])
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

        fig = px.area(daily_df, x='day', y='avg_value', 
                      title='Günlük Ortalama İşlem Değeri Trendi',
                      color_discrete_sequence=['#2ecc71'])
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    except Exception:
        st.warning("Günlük trend verileri henüz mevcut değil.")


# ==================== FEATURE IMPORTANCE ====================
elif page == "🔍 Feature Importance":
    st.markdown('<div class="section-header">Feature Importance Analizi</div>', unsafe_allow_html=True)
    results = load_ml_results()

    if results:
        for r in results:
            fi = r.get('feature_importance', {})
            if fi:
                st.markdown(f"#### {r['model_name']}")
                fi_df = pd.DataFrame(list(fi.items()), columns=['Feature','Importance'])
                fi_df = fi_df.sort_values('Importance', ascending=True)

                fig = px.bar(fi_df, x='Importance', y='Feature', orientation='h',
                             color='Importance', color_continuous_scale='Blues',
                             title=f'{r["model_name"]} - Feature Importance')
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("ML sonuclari henuz mevcut degil.")

# ==================== FOOTER ====================
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#95a5a6;'>"
    "🏦 Bank Transaction Big Data Pipeline | "
    "Docker + Kafka + Spark + Delta Lake + MLflow + Streamlit"
    "</div>",
    unsafe_allow_html=True
)
