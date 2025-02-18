import streamlit as st

def show_about_page():
    st.title("About This Project")
    st.markdown("""
    **Professional Macroeconomic Forecasting Tool**

    This website is designed to predict macroeconomic variables using various modeling techniques including VARNN, Hybrid VARNN, and VAR. It integrates data preprocessing, augmentation, normalization, and advanced model training to provide a comprehensive forecasting solution.

    ---
    ### Project Team
    - **21110370 – Huỳnh Thị Ngọc Ánh**  
      *Group 1 – Dự báo dữ liệu kinh tế vĩ mô bằng mô hình VARNN*  
      **Advisor:** Nguyễn Thành Sơn
    - **21110716 – Nguyễn Thị Thanh Tuyền**

    ---
    This tool is built with Streamlit and leverages Python’s robust data science libraries. The goal is to offer a user-friendly and professional interface for macroeconomic forecasting.
    """)