import pandas as pd
import numpy as np

def fill_null(df):

    # Maak een kopie van het dataframe om het origineel niet te wijzigen
    df_clean = df.copy()
    
    # Vul null-waarden met forward and backward fill methode
    df_clean = df_clean.ffill()
    df_clean = df_clean.bfill()


    return df_clean

def categorical_to_int(df_column, file_path):
    # Verkrijg unieke categorieën en ik sorteer ze zodat dezelfde chars altijd dezelfde numerieke waarde krijgen (dit was nodig om mijn corelatie matrix correct te krijgen)
    categories = sorted(df_column.unique())
    
    # Maak een mapping van categorieën naar int
    category_mapping = {category: i for i, category in enumerate(categories)}
    
    # Open het bestand om de mapping op te slaan
    with open(file_path, 'a') as f:
        f.write(f"\nKolom: {df_column.name}\n")
        for category, value in category_mapping.items():
            f.write(f"{category} -> {value}\n")
    
    # Transformeer de kolom naar int
    transformed_column = df_column.map(category_mapping)
    
    return transformed_column

def float_to_categorical(df_column, num_divisions,max_val, file_path):

    # Bepaal de minimum- en maximumwaarde in de kolom
    # Converteer de kolom naar floats
    df_column = pd.to_numeric(df_column)
    
    # Bepaal de minimum- en maximumwaarde in de kolom
    min_val = df_column.min()    
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

def replace_color_with_cap_color(df, column_name):
    mask = df[column_name] == "f"
    
    # Vervang deze waarden met de overeenkomstige waarde uit de cap-color kolom
    df.loc[mask, column_name] = df.loc[mask, "cap-color"]
    
    return df[column_name]

def replace_surface_with_cap_surface(df, column_name):
    mask = df[column_name] == "f"
    
    # Vervang deze waarden met de overeenkomstige waarde uit de cap-surface kolom
    df.loc[mask, column_name] = df.loc[mask, "cap-surface"]
    
    return df[column_name]

# Main functie om de nieuwe functies toe te passen
def main():
    """
    Hoofdfunctie die het dataframe inleest, verwerkt en opslaat.
    """
    
    # Maak een nieuw txt bestand aan voor de transformaties
    with open('mushroomindex.txt', 'w') as f:
        f.write("Mushroom Data Transformatie Index\n")
        f.write("================================\n")
    
    # Lees het dataframe in
    df = pd.read_csv('MushroomDataset/Mushroomdata.csv')
    
    # Vul alle null-waarden in
    df_clean = fill_null(df)
    
    # Pas de functie toe op de kleurkolommen - vervang 'f' met cap-color
    df_clean['stem-color'] = replace_color_with_cap_color(df_clean, "stem-color")
    df_clean['veil-color'] = replace_color_with_cap_color(df_clean, "veil-color")
    df_clean['gill-color'] = replace_color_with_cap_color(df_clean, "gill-color")
    
    # Pas de functie toe op de oppervlaktekolom - vervang 'f' met cap-surface
    df_clean['stem-surface'] = replace_surface_with_cap_surface(df_clean, "stem-surface")
    
    # Verwerk float kolommen naar categorische data
    df_clean["cap-diameter"] = float_to_categorical(df_clean["cap-diameter"], 20, 24, 'mushroomindex.txt')
    df_clean["stem-height"] = float_to_categorical(df_clean["stem-height"], 20, 19, 'mushroomindex.txt')
    df_clean["stem-width"] = float_to_categorical(df_clean["stem-width"], 20, 42, 'mushroomindex.txt')
    
    # Verwerk categorische kolommen naar int
    categorical_columns = [0, 2, 3, 4, 5, 6, 7, 8, 11, 12, 13, 15, 16, 17, 18, 19, 20]
    for col_idx in categorical_columns:
        col_name = df_clean.columns[col_idx]
        df_clean[col_name] = categorical_to_int(df_clean[col_name], 'mushroomindex.txt')
    
    # Drop kolommen die niet nuttig zijn
    df_clean = df_clean.drop(["veil-type"], axis=1)  # alle waarden hier zijn hetzelfde
    
    # Dataframe opslaan
    df_clean.to_csv('MushroomDataset/MushroomdatacorlationMatrix.csv', index=False)
    print(df_clean.info())

if __name__ == "__main__":
    main()