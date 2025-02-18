import streamlit as st
from streamlit_option_menu import option_menu
from custom_pages.about import show_about_page
from custom_pages.predict import show_predict_page
from custom_pages.home import show_home_page

def main():
    with st.sidebar:
        selected = option_menu(
            menu_title="Navigation",  # required
            options=["Home", "Predict", "About"],  # required
            icons=["house", "graph-up-arrow", "info-circle"],  # optional
            menu_icon="cast",
            default_index=0,
            styles={
                "container": {
                    "padding": "10px 10px",
                    "background-color": "#ffffff",
                    "height": "100vh",
                    "border-right": "1px solid #eee",
                    "box-shadow": "2px 2px 5px rgba(0, 0, 0, 0.1)"
                },
                "icon": {
                    "color": "#6b7280", 
                    "font-size": "20px"
                },
                "nav-link": {
                    "font-size": "16px",
                    "font-family": "Inter, sans-serif",
                    "margin": "10px",
                    "padding": "10px 10px",
                    "color": "#374151",
                    "transition": "all 0.3s ease",
                    "--hover-color": "#f3f4f6",
                    "border-radius": "0px"
                },
                "nav-link-selected": {
                    "background-color": "#c8d6e5",
                    "border-left": "4px solid #6366F1",
                    "font-weight": "600",
                    "color": "#111"
                }
            }
        )

    
    if selected == "Home":
        show_home_page()
    elif selected == "Predict":
        show_predict_page()
    elif selected == "About":
        show_about_page()


if __name__ == "__main__":
    main()