import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import plotly.express as px
from io import StringIO
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Configuración de la página
st.set_page_config(page_title="Análisis Energético Colombia", page_icon="⚡", layout="wide")

# Título principal
st.title("📊 Análisis del Balance Energético Colombiano")

# Función para cargar datos
def load_data(uploaded_file):
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file)
        return df
    except Exception as e:
        st.error(f"Error al cargar el archivo: {e}")
        return None

# Carga de archivos en el sidebar
st.sidebar.header("Carga de Datos")
uploaded_file = st.sidebar.file_uploader("Sube tu archivo (CSV o Excel)", type=["csv", "xlsx"])

if uploaded_file is not None:
    df = load_data(uploaded_file)
    
    if df is not None:
        st.sidebar.success("Archivo cargado correctamente!")
        
        # Sidebar con controles
        st.sidebar.header("Opciones de Análisis")
        analysis_type = st.sidebar.selectbox(
            "Seleccione el tipo de análisis",
            ["Exploración de Datos", "Correlaciones", "Modelos Predictivos", "Clustering"]
        )

        # Análisis Exploratorio
        if analysis_type == "Exploración de Datos":
            st.header("Exploración Inicial de Datos")
            
            # Mostrar dataframe
            if st.checkbox("Mostrar datos crudos"):
                st.dataframe(df)
            
            # Estadísticas descriptivas
            st.subheader("Estadísticas Descriptivas")
            st.dataframe(df.describe())
            
            # Selección de variable para histograma
            var_hist = st.selectbox("Seleccione variable para histograma", df.columns)
            fig, ax = plt.subplots()
            sns.histplot(df[var_hist], kde=True, ax=ax)
            st.pyplot(fig)

        # Análisis de Correlación
        elif analysis_type == "Correlaciones":
            st.header("Análisis de Correlaciones")
            
            # Matriz de correlación
            corr_matrix = df.corr()
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
            st.pyplot(fig)
            
            # Gráfico de dispersión interactivo
            st.subheader("Gráfico de Dispersión Interactivo")
            col1, col2 = st.columns(2)
            x_axis = col1.selectbox("Eje X", df.columns)
            y_axis = col2.selectbox("Eje Y", df.columns)
            fig = px.scatter(df, x=x_axis, y=y_axis, hover_data=[df.index])
            st.plotly_chart(fig, use_container_width=True)

        # Modelos Predictivos
        elif analysis_type == "Modelos Predictivos":
            st.header("Modelos de Machine Learning")
            
            # Selección de características
            st.subheader("Configuración del Modelo")
            features = st.multiselect(
                "Seleccione variables predictoras",
                df.columns,
                default=df.columns[:3] if len(df.columns) >= 3 else df.columns
            )
            target = st.selectbox("Variable objetivo", df.columns)
            
            if st.button("Entrenar Modelos") and features and target:
                try:
                    # Preparación de datos
                    X = df[features]
                    y = df[target]
                    
                    # Escalado y división
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)
                    X_train, X_test, y_train, y_test = train_test_split(
                        X_scaled, y, test_size=0.2, random_state=42
                    )
                    
                    # Entrenamiento de modelos
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Regresión Lineal")
                        lr = LinearRegression()
                        lr.fit(X_train, y_train)
                        y_pred = lr.predict(X_test)
                        
                        st.metric("R²", f"{r2_score(y_test, y_pred):.3f}")
                        st.metric("MSE", f"{mean_squared_error(y_test, y_pred):.2e}")
                        
                        # Coeficientes
                        coeff_df = pd.DataFrame({
                            'Variable': features,
                            'Coeficiente': lr.coef_
                        }).sort_values('Coeficiente', ascending=False)
                        st.dataframe(coeff_df)
                    
                    with col2:
                        st.subheader("Random Forest")
                        rf = RandomForestRegressor(n_estimators=100, random_state=42)
                        rf.fit(X_train, y_train)
                        y_pred_rf = rf.predict(X_test)
                        
                        st.metric("R²", f"{r2_score(y_test, y_pred_rf):.3f}")
                        st.metric("MSE", f"{mean_squared_error(y_test, y_pred_rf):.2e}")
                        
                        # Importancia de características
                        importances = pd.DataFrame({
                            'Variable': features,
                            'Importancia': rf.feature_importances_
                        }).sort_values('Importancia', ascending=False)
                        
                        fig, ax = plt.subplots()
                        sns.barplot(x='Importancia', y='Variable', data=importances, ax=ax)
                        st.pyplot(fig)
                
                except Exception as e:
                    st.error(f"Error al entrenar modelos: {e}")

        # Análisis de Clustering
        else:
            st.header("Análisis de Clustering")
            
            # Selección de características para clustering
            cluster_features = st.multiselect(
                "Variables para clustering",
                df.columns,
                default=df.columns[:2] if len(df.columns) >= 2 else df.columns
            )
            
            if cluster_features and st.button("Ejecutar Clustering"):
                try:
                    # Escalado
                    X = df[cluster_features]
                    X_scaled = StandardScaler().fit_transform(X)
                    
                    # Reducción de dimensionalidad
                    pca = PCA(n_components=2)
                    principal_components = pca.fit_transform(X_scaled)
                    
                    # Clustering
                    kmeans = KMeans(n_clusters=3, random_state=42)
                    clusters = kmeans.fit_predict(X_scaled)
                    
                    # Visualización
                    fig, ax = plt.subplots(figsize=(10, 6))
                    scatter = ax.scatter(
                        principal_components[:, 0], 
                        principal_components[:, 1], 
                        c=clusters, 
                        cmap='viridis',
                        alpha=0.6
                    )
                    plt.colorbar(scatter)
                    ax.set_xlabel('Componente Principal 1')
                    ax.set_ylabel('Componente Principal 2')
                    ax.set_title('Resultados de Clustering (K=3)')
                    st.pyplot(fig)
                    
                    # Estadísticas por cluster
                    df_cluster = df.copy()
                    df_cluster['Cluster'] = clusters
                    st.dataframe(df_cluster.groupby('Cluster')[cluster_features].mean())
                
                except Exception as e:
                    st.error(f"Error en clustering: {e}")

    else:
        st.sidebar.error("Error al procesar el archivo. Verifica el formato.")
else:
    st.info("Por favor, sube un archivo CSV o Excel para comenzar el análisis")

# Notas finales
st.sidebar.markdown("---")
st.sidebar.info(
    "🔍 Herramienta de análisis de datos energéticos\n\n"
    "Creada con Streamlit | Python")
