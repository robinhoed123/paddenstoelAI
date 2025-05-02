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
    """Create and save frequency plots for each column"""
    try:
        for column in df.columns:
            plt.figure(figsize=(10, 6))
            
            # Handle different column types
            if df[column].dtype in [np.number]:
                # For numeric columns, create a histogram
                sns.histplot(df[column], kde=True)
                plt.title(f'Frequency Distribution of {column}', fontsize=14)
                plt.xlabel(column)
                plt.ylabel('Frequency')
            else:
                # For categorical columns, create a bar plot
                value_counts = df[column].value_counts().sort_values(ascending=False)
                # Limit to top 20 values if there are too many
                if len(value_counts) > 20:
                    value_counts = value_counts.head(20)
                    plt.title(f'Top 20 Frequency Distribution of {column}', fontsize=14)
                else:
                    plt.title(f'Frequency Distribution of {column}', fontsize=14)
                
                sns.barplot(x=value_counts.index, y=value_counts.values)
                plt.xticks(rotation=45, ha='right')
                plt.xlabel(column)
                plt.ylabel('Count')
            
            plt.tight_layout()
            
            # Save the visualization
            safe_column_name = column.replace('/', '_').replace('\\', '_')
            output_path = os.path.join(output_dir, f'frequency_{safe_column_name}.png')
            plt.savefig(output_path, dpi=300)
            plt.close()
            print(f"Frequency plot for {column} saved to {output_path}")
    except Exception as e:
        print(f"Error creating frequency plots: {e}")

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

def visualize_missing_values(df, output_dir):
    """Create and save visualization of missing values"""
    try:
        plt.figure(figsize=(12, 8))
        
        # Calculate missing values
        missing = df.isnull().sum()
        missing = missing[missing > 0].sort_values(ascending=False)
        
        if missing.empty:
            print("No missing values found in the dataset")
            return
        
        # Create bar plot of missing values
        sns.barplot(x=missing.index, y=missing.values)
        plt.title('Missing Values by Column', fontsize=14)
        plt.xticks(rotation=45, ha='right')
        plt.ylabel('Count of Missing Values')
        plt.tight_layout()
        
        # Save the visualization
        output_path = os.path.join(output_dir, 'missing_values.png')
        plt.savefig(output_path, dpi=300)
        plt.close()
        print(f"Missing values plot saved to {output_path}")
    except Exception as e:
        print(f"Error creating missing values plot: {e}")

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
        visualize_boxplots(df, output_dir)
        visualize_pairplot(df, output_dir)
        visualize_missing_values(df, output_dir)
        
        print(f"\nVisualization complete! All images saved to the '{output_dir}' directory.")
    else:
        print("Could not generate visualizations due to errors with the CSV file.")

if __name__ == "__main__":
    main()