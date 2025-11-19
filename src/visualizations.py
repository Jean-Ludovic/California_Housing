import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

def show_visualizations(df):
    st.write("Distribution des prix")
    fig, ax = plt.subplots()
    sns.histplot(df["median_house_value"], bins=30, ax=ax)
    st.pyplot(fig)
