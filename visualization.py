# Data visualization
import matplotlib.pyplot as plt

# Streamlit for UI
import streamlit as st


import matplotlib.pyplot as plt
import streamlit as st
import textwrap

def visualize_predictions(actual, predicted, columns, wrap_width=20):
    num_plots = len(columns)
    fig, axes = plt.subplots(num_plots, 1, figsize=(12, num_plots * 4), sharex=True)

    # Ensure axes is iterable even if there's only one subplot
    if num_plots == 1:
        axes = [axes]

    # Loop through each column and plot both actual and predicted values
    for i, column_name in enumerate(columns):
        # Wrap long column names for better display in the title
        wrapped_name = "\n".join(textwrap.wrap(column_name, width=wrap_width))
        ax = axes[i]
        ax.plot(actual.index, actual[column_name], label="Actual", 
                color="#1f77b4", linestyle="--", linewidth=2)
        ax.plot(actual.index, predicted[:, i], label="Predicted", 
                color="#ff7f0e", linestyle=":", linewidth=2)

        ax.set_title(f"Actual vs. Predicted:\n{wrapped_name}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Time", fontsize=12)
        ax.set_ylabel(wrapped_name, fontsize=12)
        ax.legend(loc="best", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.6)

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


