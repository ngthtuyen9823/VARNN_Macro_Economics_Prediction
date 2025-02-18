import matplotlib.pyplot as plt
import textwrap
import streamlit as st

def visualize_predictions(actual, predicted, columns, wrap_width=20):
    """Visualizes actual vs. predicted values for multiple time series columns.

    Args:
        actual (pd.DataFrame): DataFrame containing actual values.
        predicted (np.ndarray): Array of predicted values (shape should match actual values).
        columns (list): List of column names corresponding to the actual and predicted data.
        wrap_width (int, optional): Width for wrapping long column names in the title. Defaults to 20.

    Returns:
        None: Displays plots using Streamlit.
    """
    num_plots = len(columns)
    fig, axes = plt.subplots(num_plots, 1, figsize=(12, num_plots * 4), sharex=True)

    # Ensure axes is iterable even if there's only one subplot
    if num_plots == 1:
        axes = [axes]

    # Loop through each column and plot both actual and predicted values
    for i, column_name in enumerate(columns):
        wrapped_name = "\n".join(textwrap.wrap(column_name, width=wrap_width))
        ax = axes[i]

        ax.plot(actual.index, actual[column_name], label="Actual", color="#1f77b4", 
                linestyle="--", linewidth=2)
        ax.plot(actual.index, predicted[:, i], label="Predicted", color="#ff7f0e", 
                linestyle=":", linewidth=2)

        ax.set_title(f"Actual vs. Predicted:\n{wrapped_name}", fontsize=14, fontweight="bold")
        ax.set_xlabel("Time", fontsize=12)
        ax.set_ylabel(wrapped_name, fontsize=12)
        ax.legend(loc="best", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    st.pyplot(fig)
    

def visualize_column(data, column, description=None):
    """Visualizes a specified column from a dataset using a line chart in Streamlit.

    Args:
        data (pd.DataFrame): The dataset containing the column to visualize.
        column (str): The name of the column to be visualized.
        description (str, optional): Additional context or description for the visualization.
    """
    if column not in data.columns:
        st.error(f"Error: The column {column} does not exist in the dataset.")
        return

    st.subheader(f"Visualization of {column}")

    if description:
        st.write(description)

    if data[column].dropna().empty:
        st.warning(f"No valid data available in column {column} for visualization.")
        return

    st.line_chart(data[column])

    st.write(f"Displaying {data[column].count()} non-null values from the column {column}.")
