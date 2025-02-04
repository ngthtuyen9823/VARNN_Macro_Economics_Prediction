# Data visualization
import matplotlib.pyplot as plt

# Streamlit for UI
import streamlit as st


def visualize_predictions(actual, predicted, columns):
    fig, axes = plt.subplots(len(columns), 1, figsize=(12, len(columns) * 4), sharex=True)
    
    if len(columns) == 1:
        axes = [axes]
    
    for i, column_name in enumerate(columns):
        axes[i].plot(actual.index, actual[column_name], label="Actual", color="blue", linestyle="--", linewidth=2)
        axes[i].plot(actual.index, predicted[:, i], label="Predicted", color="red", linestyle=":", linewidth=2)
        
        axes[i].set_title(f"Actual vs Predicted for {column_name}", fontsize=14, weight="bold")
        axes[i].set_xlabel("Time", fontsize=12)
        axes[i].set_ylabel(column_name, fontsize=12)
        axes[i].legend(fontsize=10)
        axes[i].grid(visible=True, linestyle='--', alpha=0.6)  

    plt.tight_layout()
    
    st.pyplot(fig)


def visualize_column(data, column, description=None):
    if column not in data.columns:
        st.error(f"Column '{column}' not found in the dataset.")
        return

    st.subheader(f"Visualization for: {column}")
    if description:
        st.write(description)

    st.line_chart(data[column])
    
    st.write(f"Displaying {len(data[column].dropna())} non-null values from the column '{column}'.")


