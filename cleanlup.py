import pandas as pd
import numpy as np

def fill_null(df):

    # Maak een kopie van het dataframe om het origineel niet te wijzigen
    df_clean = df.copy()
    
    # Vul null-waarden met forward fill methode
    df_clean = df_clean.ffill()
    
    # Als er nog steeds null-waarden zijn (bijv. aan het begin), gebruik backward fill
    df_clean = df_clean.bfill()


    return df_clean

def categorical_to_integer(df_column, file_path):
    """
    Deze functie zet categorische data om naar integers en slaat de mapping op in een bestand.
    
    Args:
        df_column: De pandas kolom met categorische data
        file_path: Het bestandspad waar de mapping wordt opgeslagen
    
    Returns:
        Een pandas Series met de getransformeerde integer data
    """
    # Verkrijg unieke categorieën
    categories = df_column.unique()
    
    # Maak een mapping van categorieën naar integers
    category_mapping = {category: i for i, category in enumerate(categories) if pd.notna(category)}
    
    # Open het bestand om de mapping op te slaan
    with open(file_path, 'a') as f:
        f.write(f"\nKolom: {df_column.name}\n")
        for category, value in category_mapping.items():
            f.write(f"{category} -> {value}\n")
    
    # Transformeer de kolom naar integers
    transformed_column = df_column.map(category_mapping)
    
    return transformed_column

def float_to_categorical(df_column, num_divisions, file_path):

    # Bepaal de minimum- en maximumwaarde in de kolom
    # Converteer de kolom naar floats, negeer niet-numerieke waarden
    df_column = pd.to_numeric(df_column, errors='coerce')
    
    # Bepaal de minimum- en maximumwaarde in de kolom
    min_val = df_column.min()
    max_val = df_column.max()
    
    # Bereken de breedte van elke range
    range_width = (max_val-min_val) / num_divisions
    
    # Maak de ranges aan
    ranges = []
    for i in range(num_divisions):
        lower = min_val + i * range_width
        upper = min_val + (i + 1) * range_width
        ranges.append((lower, upper))
    
    # Schrijf de ranges naar het bestand
    with open(file_path, 'a') as f:
        f.write(f"\nKolom: {df_column.name}\n")
        for i, (lower, upper) in enumerate(ranges):
            f.write(f"Range {i}: {lower:.2f} - {upper:.2f}\n")
    
    # Functie om de waarde te vervangen door de bijbehorende categorie-index
    def assign_category(value):
        for i, (lower, upper) in enumerate(ranges):
            if lower <= value <= upper:
                return i
        #float devison presion problem fix (value 62.34 rij 58318 in data frame)
        print(f"wat gebuert er wat is dit value:{value} range:{ranges}")
        return num_divisions-1
    
    # Transformeer de kolom naar categorieën
    transformed_column = df_column.apply(assign_category)
    
    return transformed_column

# Main functie
def main():
    """
    Hoofdfunctie die het dataframe inleest, verwerkt en opslaat.
    """
    # Maak een nieuw bestand aan voor de indices
    with open('mushroomindex.txt', 'w') as f:
        f.write("Mushroom Data Transformatie Index\n")
        f.write("================================\n")
    
    # Lees het dataframe in
    df = pd.read_csv('MushroomDataset/Mushroomdata.csv')
    
    # Vul alle null-waarden in
    df_clean = fill_null(df)
    
    # Verwerk float kolommen naar categorische data
    float_columns = [1, 9, 10]
    for col_idx in float_columns:
        col_name = df_clean.columns[col_idx]
        df_clean[col_name] = float_to_categorical(df_clean[col_name], 5, 'mushroomindex.txt')
    
    # Verwerk categorische kolommen naar integers
    categorical_columns = [0,2, 3, 4, 5, 6, 7, 8, 11, 12, 13, 15, 16, 17, 18, 19, 20]
    for col_idx in categorical_columns:
        col_name = df_clean.columns[col_idx]
        df_clean[col_name] = categorical_to_integer(df_clean[col_name], 'mushroomindex.txt')
    #de reden waarom dit gedropt wordt is dat alle waarde van deze feuter 0 zijn (heeft geen waarde)
    df_clean = df_clean.drop(["veil-type"], axis=1)
    # dataframe opslaan
    df_clean.to_csv('MushroomDataset/Mushroomdataclean.csv', index=False)
    print(df_clean.info())

if __name__ == "__main__":
    main()