import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime

def create_output_directory():
    """Create a directory to store all visualization outputs"""
    output_dir = f"visualizations"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    return output_dir

def read_csv_data(file_path):
    """Read data from a CSV file"""
    try:
        df = pd.read_csv(file_path)
        print(f"Successfully read data from {file_path}")
        print(f"Data shape: {df.shape}")
        return df
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return None

def visualize_correlation_matrix(df, output_dir):
    """Create and save correlation matrix visualization"""
    try:
        # Calculate correlation for numeric columns only
        numeric_df = df.select_dtypes(include=[np.number])
        
        if numeric_df.empty:
            print("No numeric columns found for correlation matrix")
            return
        
        # Create correlation matrix
        plt.figure(figsize=(12, 10))
        corr_matrix = numeric_df.corr()
        
        # Create heatmap
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
        plt.title('Correlation Matrix', fontsize=16)
        plt.tight_layout()
        
        # Save the visualization
        output_path = os.path.join(output_dir, 'correlation_matrix.png')
        plt.savefig(output_path, dpi=300)
        plt.close()
        print(f"Correlation matrix saved to {output_path}")
    except Exception as e:
        print(f"Error creating correlation matrix: {e}")

def visualize_column_frequencies(df, output_dir):
    """Create and save frequency plots for each column in a single figure with subplots"""
    try:
        num_columns = len(df.columns)
        num_rows = (num_columns + 2) // 4  # Arrange subplots in a grid with 3 columns
        
        fig, axes = plt.subplots(num_rows, 4, figsize=(15, 5 * num_rows))
        axes = axes.flatten()  # Flatten axes for easy iteration
        
        for i, column in enumerate(df.columns):
            ax = axes[i]
            value_counts = df[column].value_counts().sort_values(ascending=False)
            
            sns.barplot(x=value_counts.index, y=value_counts.values, ax=ax)
            ax.set_title(f'Frequency of {column}', fontsize=12)
            ax.set_xlabel(column)
            ax.set_ylabel('Count')
        
        # Hide any unused subplots
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])
        
        plt.tight_layout()
        
        # Save the visualization
        output_path = os.path.join(output_dir, 'frequency_subplots.png')
        plt.savefig(output_path, dpi=300)
        plt.close()
        print(f"Frequency plots saved to {output_path}")
    except Exception as e:
        print(f"Error creating frequency plots: {e}")
def visualize_float_column_plots(columns, df, output_dir):
    """Create and save line plots for float columns"""
    try:
        # Filter float columns
        float_columns = [col for col in columns if df[col].dtype == 'float64']
        
        if not float_columns:
            print("No float columns found for plotting")
            return
        
        num_columns = len(float_columns)
        num_rows = (num_columns + 2) // 3  # Arrange subplots in a grid with 3 columns
        
        fig, axes = plt.subplots(num_rows, 3, figsize=(15, 5 * num_rows))
        axes = axes.flatten()  # Flatten axes for easy iteration
        
        for i, column in enumerate(float_columns):
            ax = axes[i]
            ax.plot(df[column], marker='.', linestyle='-', label=column)
            ax.set_title(f'Plot of {column}', fontsize=12)
            ax.set_xlabel('Index')
            ax.set_ylabel(column)
            ax.legend()
        
        # Hide any unused subplots
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])
        
        plt.tight_layout()
        
        # Save the visualization
        output_path = os.path.join(output_dir, 'float_column_plots.png')
        plt.savefig(output_path, dpi=300)
        plt.close()
        print(f"Float column plots saved to {output_path}")
    except Exception as e:
        print(f"Error creating float column plots: {e}")

def visualize_boxplots(df, output_dir):
    """Create and save boxplots for numeric columns"""
    try:
        # Select numeric columns
        numeric_df = df.select_dtypes(include=[np.number])
        
        if numeric_df.empty:
            print("No numeric columns found for boxplots")
            return
        
        # Create boxplots for each numeric column
        for column in numeric_df.columns:
            plt.figure(figsize=(10, 6))
            sns.boxplot(x=numeric_df[column])
            plt.title(f'Boxplot of {column}', fontsize=14)
            plt.tight_layout()
            
            # Save the visualization
            safe_column_name = column.replace('/', '_').replace('\\', '_')
            output_path = os.path.join(output_dir, f'boxplot_{safe_column_name}.png')
            plt.savefig(output_path, dpi=300)
            plt.close()
            print(f"Boxplot for {column} saved to {output_path}")
    except Exception as e:
        print(f"Error creating boxplots: {e}")

def visualize_pairplot(df, output_dir):
    """Create and save pairplot for numeric columns"""
    try:
        # Select numeric columns
        numeric_df = df.select_dtypes(include=[np.number])
        
        if numeric_df.empty:
            print("No numeric columns found for pairplot")
            return
        
        # Limit to 5 columns if there are too many
        if numeric_df.shape[1] > 5:
            print(f"Limiting pairplot to first 5 numeric columns (from {numeric_df.shape[1]} total)")
            numeric_df = numeric_df.iloc[:, :5]
        
        # Create pairplot
        plt.figure(figsize=(12, 10))
        pairplot = sns.pairplot(numeric_df)
        plt.suptitle('Pairwise Relationships', y=1.02, fontsize=16)
        plt.tight_layout()
        
        # Save the visualization
        output_path = os.path.join(output_dir, 'pairplot.png')
        pairplot.savefig(output_path, dpi=300)
        plt.close()
        print(f"Pairplot saved to {output_path}")
    except Exception as e:
        print(f"Error creating pairplot: {e}")


def main():
    """Main function to run all visualization functions"""
    # Create output directory
    output_dir = create_output_directory()
    
    file_path = "MushroomDataset/Mushroomdataclean.csv"
    
    # Read data
    df = read_csv_data(file_path)
    
    if df is not None:
        print("\nGenerating visualizations...")
        # Generate all visualizations
        visualize_correlation_matrix(df, output_dir)
        visualize_column_frequencies(df, output_dir)
        #visualize_float_column_plots(["stem-width", "stem-height","cap-diameter"], df, output_dir)
        # visualize_boxplots(df, output_dir)
        # visualize_pairplot(df, output_dir)
        # visualize_missing_values(df, output_dir)
        
        print(f"\nVisualization complete! All images saved to the '{output_dir}' directory.")
    else:
        print("Could not generate visualizations due to errors with the CSV file.")

if __name__ == "__main__":
    main()