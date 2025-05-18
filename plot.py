import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def create_output_directory(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    return output_dir

def read_csv_data(file_path):
    try:
        df = pd.read_csv(file_path)
        print(f"Successfully read data from {file_path}")
        return df
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return None

def plot_correlation_matrix(df, output_dir):
    try:
        # correlation matrix
        plt.figure(figsize=(12, 10))
        corr_matrix = df.corr()
        
        # heatmap
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
        plt.title('Correlation Matrix', fontsize=16)
        plt.tight_layout()
        
        # Save  as png
        output_path = os.path.join(output_dir, 'correlation_matrix.png')
        plt.savefig(output_path, dpi=300)
        plt.close()
    except Exception as e:
        print(f"Error creating correlation matrix: {e}")

def plot_column_frequencies(df, output_dir):
    # subplots in een grind 3X4
    try:
        num_columns = len(df.columns)
        num_rows = (num_columns + 2) // 4  
        fig, axes = plt.subplots(num_rows, 4, figsize=(15, 5 * num_rows))
        axes = axes.flatten()
        
        for i, column in enumerate(df.columns):
            ax = axes[i]
            value_counts = df[column].value_counts().sort_values(ascending=False)
            
            sns.barplot(x=value_counts.index, y=value_counts.values, ax=ax)
            ax.set_title(f'Frequency of {column}', fontsize=12)
            ax.set_xlabel(column)
            ax.set_ylabel('Count')        
        plt.tight_layout()
        
        # Save png
        output_path = os.path.join(output_dir, 'frequency_Plot.png')
        plt.savefig(output_path, dpi=300)
        plt.close()
    except Exception as e:
        print(f"Error creating frequency plots: {e}")
def plot_float_column(columns, df, output_dir):
    try:
        num_columns = len(columns)
        num_rows = (num_columns + 2) // 3
        
        fig, axes = plt.subplots(num_rows, 3, figsize=(15, 5 * num_rows))
        axes = axes.flatten() 
        
        for i, column in enumerate(columns):
            ax = axes[i]
            ax.plot(df[column], marker='.', linestyle='-', label=column)
            ax.set_title(f'Plot of {column}', fontsize=12)
            ax.set_xlabel('Index')
            ax.set_ylabel(column)
            ax.legend()
        plt.tight_layout()
        
        # Save as png
        output_path = os.path.join(output_dir, 'columns_Plot.png')
        plt.savefig(output_path, dpi=300)
        plt.close()
    except Exception as e:
        print(f"Error creating float column plots: {e}")

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

def plot_mutual_information(df, output_dir):
    try:
        from sklearn.feature_selection import mutual_info_classif
        
        # Separate features and target
        X = df.drop('class', axis=1)
        y = df['class']
        
        # Calculate mutual information
        mi_scores = mutual_info_classif(X, y, discrete_features=True)
        
        # Create a DataFrame for better visualization
        mi_df = pd.DataFrame({
            'Feature': X.columns,
            'Mutual Information': mi_scores
        }).sort_values('Mutual Information', ascending=False)
        
        # Plot
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Mutual Information', y='Feature', data=mi_df)
        plt.title('Mutual Information with Target Class', fontsize=16)
        plt.tight_layout()
        
        # Save plot
        output_path = os.path.join(output_dir, 'mutual_information.png')
        plt.savefig(output_path, dpi=300)
        plt.close()
        
        print(f"Mutual information plot saved to {output_path}")
    except Exception as e:
        print(f"Error creating mutual information plot: {e}")

def main():
    output_dir = create_output_directory("plot data")
    
    #data inlezen om te plotten
    file_path = "MushroomDataset\Mushroomdataclean.csv"
    df = read_csv_data(file_path)
    
    if df is not None:
        plot_correlation_matrix(df, output_dir)
        plot_mutual_information(df, output_dir)
        # plot_column_frequencies(df, output_dir)
        plot_float_column(["stem-width", "stem-height","cap-diameter"], df, output_dir)
    else:
        print(f"Error: could not read data from '{file_path}'")

if __name__ == "__main__":
    main()