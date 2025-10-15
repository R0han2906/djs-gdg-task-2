import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="F1 DNF Classification Dashboard", layout="wide")
st.title('🏎️ F1 DNF Classification Dashboard')


@st.cache_data
def load_and_process_data():
 
    # path = kagglehub.dataset_download("pranay13257/f1-dnf-classification")

    
    # csv_file_path = os.path.join(path, 'f1_dnf.csv')
    # df = pd.read_csv("f1_dnf.csv")
    df = pd.read_csv('f1_dnf.csv')
     

    df = df.drop_duplicates()
    df = df.replace('\\N', np.nan)
    df = df.dropna()

    df["dob"] = pd.to_datetime(df["dob"], errors='coerce')
    df["date"] = pd.to_datetime(df["date"], errors='coerce')

    cols = ['milliseconds', 'rank', 'fastestLap']
    for col in cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df['positionOrder'] = pd.to_numeric(df['positionOrder'], errors='coerce').astype('Int64')
    df['fastestLapSpeed'] = pd.to_numeric(df['fastestLapSpeed'], errors='coerce').astype(float)

    df["DriverAge"] = 2025 - df["dob"].dt.year
    df["status"] = np.where(df["points"] > 0, 'Finished', 'DNF')

    return df

df = load_and_process_data()


tab1, tab2, tab3, tab4 = st.tabs([" Overview", " Charts", " Stats", " Filter Data"])


with tab1:
    st.subheader('Dataset Info')

    import io
    buffer = io.StringIO()
    df.info(buf=buffer)
    info_str = buffer.getvalue()
    st.text(info_str)

    st.subheader('Cleaned Data Sample')
    st.dataframe(df.head(10), use_container_width=True)


with tab2:
    st.subheader('Visual Analysis')
    col1, col2 = st.columns(2)

    
    with col1:
        st.write("### 🏁 Finish Status Distribution")
        status_counts = df['status'].value_counts()
        fig1, ax1 = plt.subplots()
        ax1.pie(status_counts, labels=status_counts.index, autopct='%1.1f%%',
                startangle=90, colors=["#2ecc71", "#e74c3c"])
        ax1.axis('equal')
        st.pyplot(fig1)

    
    with col2:
        st.write("### 👨‍✈️ Driver Age Distribution")
        fig2, ax2 = plt.subplots()
        sns.histplot(df["DriverAge"], bins=20, kde=True, color="#3498db", ax=ax2)
        st.pyplot(fig2)

    
    st.write("### ⚡ Fastest Lap Speed by Status")
    fig3, ax3 = plt.subplots()
    sns.boxplot(data=df, x="status", y="fastestLapSpeed", palette="Set2", ax=ax3)
    st.pyplot(fig3)


with tab3:
    st.subheader('Statistics Summary')
    st.dataframe(df.describe(include='all'), use_container_width=True)


with tab4:
    st.subheader("🔎 Filter Data by Team or Driver")
    col1, col2 = st.columns(2)
    with col1:
        team = st.selectbox("Select Constructor", options=sorted(df["constructorRef"].unique()))
    with col2:
        driver = st.selectbox("Select Driver", options=sorted(df["driverRef"].unique()))

    filtered_df = df[(df["constructorRef"] == team) & (df["driverRef"] == driver)]
    st.write(f"Showing results for **{driver} ({team})**:")
    st.dataframe(filtered_df, use_container_width=True)
