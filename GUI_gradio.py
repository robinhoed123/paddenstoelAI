import gradio as gr
from joblib import load
import csv
# met deze code kun je heel gemakkelijk de modelen testen in een GUI
# en de resultaten worden opgeslagen in een csv bestand
production = None
confidence = 0.0
modelName="RandomForest"

# functie om een voorspelling te maken met een geselecteerd model
def make_prediction(input_data):
    global prodiction, confidence,model
    try:
        model = load(f"models\{modelName}\{modelName}_model.joblib")
        prediction = model.predict([input_data])
        probabilities = model.predict_proba([input_data])
        predicted_feature = prediction[0]
        confidence = max(probabilities[0]) * 100  #om zetten naar procent
        if predicted_feature == 1:
            prodiction = "edible"
        elif predicted_feature == 0:
            prodiction = "poisonous"
        return f"This mushroom is : {prodiction}, Confidence: {confidence:.2f}%"
    except Exception as e:
        return f"An error occurred: {str(e)}"

# functie om de string te converteren naar een int voor de modelen
# door een lookup tabel die de string omzet naar een int
def string_int_lookup(string):
    lookup_tabel = {
        "convex": 0,
        "flat": 1,
        "spherical": 2,
        "bell": 3,
        "conical": 4,
        "sunken": 5,
        "others": 6,
        "grooves": 0,
        "shiny": 1,
        "sticky": 2,
        "scaly": 3,
        "fleshy": 4,
        "smooth": 5,
        "leathery": 6,
        "dry": 7,
        "wrinkled": 8,
        "fibrous": 9,
        "silky": 10,
        "orange": 0,
        "red": 1,
        "brown": 2,
        "gray": 3,
        "green": 4,
        "white": 5,
        "yellow": 6,
        "pink": 7,
        "purple": 8,
        "buff": 9,
        "blue": 10,
        "black": 11,
        "free": 0,
        "adnate": 1,
        "decurrent": 2,
        "sinuate": 3,
        "adnexed": 4,
        "pores": 5,
        "No_attachment": 6,
        "swollen": 0,
        "bulbous": 1,
        "rooted": 2,
        "club": 3,
        "Filamentous": 4,
        "ring": 0,
        "grooved": 0,
        "pendant": 1,
        "evanescent": 2,
        "large": 3,
        "movable": 5,
        "flaring": 6,
        "zone": 7,
        "woods": 0,
        "meadows": 1,
        "grasses": 2,
        "heaths": 3,
        "leaves": 4,
        "paths": 5,
        "waste": 6,
        "urban": 7,
        "winter": 0,
        "summer": 1,
        "autumn": 2,
        "spring": 3,
        "close": 0,
        "distant": 1

}
    try: 
        x=lookup_tabel[string]
        print(f"look up found {string}=={x}")
        return x
    except:
        print(f"look up {string} not found in lookup tabel")

# functie om de waarde van de slider te converteren naar een int (een index van een bepaalde range)
# dit is nodig omdat de modelen zijn getraind op een bepaalde range van waarden)
def slider_int_corection(value,range_type):
    cap_diameter_max = [
    1.55, 2.73, 3.91, 5.09, 6.28, 7.46, 8.64, 9.82, 11.00, 12.18,
    13.36, 14.54, 15.72, 16.90, 18.08, 19.27, 20.45, 21.63, 22.81, 23.99
    ]

    stem_height_max = [
        0.94, 1.89, 2.84, 3.79, 4.74, 5.69, 6.64, 7.59, 8.54, 9.49,
        10.44, 11.39, 12.34, 13.29, 14.24, 15.19, 16.14, 17.09, 18.04, 18.99
    ]

    stem_width_max = [
        2.09, 4.19, 6.29, 8.39, 10.49, 12.59, 14.69, 16.79, 18.89, 20.99,
        23.09, 25.19, 27.29, 29.39, 31.49, 33.59, 35.69, 37.79, 39.89, 41.99
    ]
    if range_type==0:
        max=cap_diameter_max
    elif range_type==1:
        max=stem_height_max
    elif range_type==2:
        max=stem_width_max
    for index, value_max in enumerate(max):
        if value >= value_max:
            return index
    return len(max) - 1  # Return de laatste index als geen match is gevonden

# functie om de data te verzamelen van de interface
# en deze te converteren naar een lijst van waarden die de modelen kunnen gebruiken
# maakt gebruik van de eerder genoemde functies om de waarden te converteren
def collect_data(cap_diameter, stem_height, stem_width, gill_spacing, 
                does_bruise_bleed, has_ring, cap_shape, 
                surface, color, gill_attachment, stem_root, ring_type, 
                habitat, season):
    data = []
    data.append(slider_int_corection(cap_diameter,0))
    data.append(slider_int_corection(stem_height, 1))
    data.append(slider_int_corection(stem_width, 2))
    data.append(gill_spacing)
    data.append(cap_shape)
    data.append(surface)
    data.append(color)
    data.append(1 if does_bruise_bleed else 0)
    data.append(gill_attachment)
    data.append(stem_root)
    data.append(0 if has_ring else 1)
    data.append(ring_type if has_ring else 4)
    data.append(habitat)
    data.append(season)
    print(season)

    return make_prediction(data) 
def save_data_to_csv(cap_diameter, stem_height, stem_width, gill_spacing, 
                     does_bruise_bleed, has_ring, cap_shape, 
                     surface, color, gill_attachment, stem_root, ring_type, 
                     habitat, season):
    global modelName,prodiction,confidence
    # file path
    file_path = f"resultatenGUI/mushroom_data({modelName}).csv"
    print("test")
    print(modelName)
    # data
    data_row = [
        cap_diameter, stem_height, stem_width, gill_spacing, 
        does_bruise_bleed, has_ring, cap_shape, 
        surface, color, gill_attachment, stem_root, ring_type, 
        habitat, season, prodiction, confidence
    ]
    
    # Write CSV file
    try:
        with open(file_path, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            # Write header if the file is empty
            if file.tell() == 0:
                writer.writerow([
                    "Cap Diameter", "Stem Height", "Stem Width", "Gill Spacing", 
                    "Does Bruise/Bleed", "Has Ring", "Cap Shape", 
                    "Surface", "Color", "Gill Attachment", "Stem Root", "Ring Type", 
                    "Habitat", "Season", "Result", "Confidence"
                ])
            writer.writerow(data_row)
        return "Data saved successfully!"
    except Exception as e:
        return f"An error occurred while saving: {str(e)}"

###########################################################################################################        
# Gradio interface
with gr.Blocks() as app:
    gr.Markdown("# Mushroom Data Interface")
    
    # Dropdown to select the model
    model_selector = gr.Dropdown(
        ["AdaBoost", "DecisionTree", "KNN","MLP", "RandomForest", "SVM", "XGBoost"],
        label="Select Model",
        value="RandomForest"
    )

    # Update the global modelName variable when a model is selected
    model_selector.change(
        fn=lambda x: globals().update(modelName=x) or f"Model set to {x}",
        inputs=model_selector,
        outputs=None
    )

    # SLIDERS
    cap_diameter = gr.Slider(0.38, 62.34, value=10.0, label="Cap diameter (cm)")
    stem_height = gr.Slider(0.0, 33.92, value=5.0, label="Stem height (cm)")
    stem_width = gr.Slider(0.0, 103.91, value=10.0, label="Stem width (mm)")
    
    # RADIO BUTTONS
    gill_spacing = gr.Radio([("close", 0), ("distant", 1)], label="Gill spacing", value="close")
    
    # CHECKBOXES
    does_bruise_bleed = gr.Checkbox(label="Does it bruise or bleed ?")
    has_ring = gr.Checkbox(label="Has ring")
    
    # DROPDOWNS
    cap_shape = gr.Dropdown(
        [("bell", 3), ("conical", 4), ("convex", 0), ("flat", 1), ("sunken", 5), ("spherical", 2), ("others", 6)],
        label="Cap shape"
    )
    
    surface = gr.Dropdown(
        [("dry", 7), ("fibrous", 9), ("grooves", 0), ("scaly", 3), ("smooth", 5), ("shiny", 1), 
         ("leathery", 6), ("silky", 10), ("sticky", 2), ("wrinkled", 8), ("fleshy", 4)],
        label="Surface"
    )
    
    color = gr.Dropdown(
        [("brown", 2), ("buff", 9), ("gray", 3), ("green", 4), ("pink", 7), ("purple", 8), 
         ("red", 1), ("white", 5), ("yellow", 6), ("blue", 10), ("orange", 0), ("black", 11)],
        label="Color"
    )
    
    gill_attachment = gr.Dropdown(
        [("adnate", 1), ("adnexed", 4), ("decurrent", 2), ("free", 0), 
         ("sinuate", 3), ("pores", 5), ("No_attachment", 6)],
        label="Gill attachment"
    )
    
    stem_root = gr.Dropdown(
        [("bulbous", 1), ("swollen", 0), ("club", 3), ("Filamentous", 4), ("rooted", 2)],
        label="Stem root"
    )
    
    # Ring type dropdown (onzichtbaar)
    ring_type = gr.Dropdown(
        [("evanescent", 2), ("flaring", 6), ("grooved", 0), ("large", 3), 
         ("pendant", 1), ("zone", 7), ("movable", 5)],
        label="Ring type",
        visible=False
    )
    
    habitat = gr.Dropdown(
        [("grasses", 2), ("leaves", 4), ("meadows", 1), ("paths", 5), ("heaths", 3), 
         ("urban", 7), ("waste", 6), ("woods", 0)],
        label="Habitat"
    )
    
    season = gr.Dropdown(
        [("spring", 3), ("summer", 1), ("autumn", 2), ("winter", 0)],
        label="Season"
    )
    
    # Output
    output = gr.Textbox(label="Result")
    
    # Submit button
    submit_btn = gr.Button("Submit") #voor een of andere reden wordt submit vert
    submit_btn.click(
        fn=collect_data,
        inputs=[cap_diameter, stem_height, stem_width, gill_spacing, 
              does_bruise_bleed, has_ring, cap_shape, 
              surface, color, gill_attachment, stem_root, ring_type, 
              habitat, season],
        outputs=output
    )                       

    # save button
    save_btn = gr.Button("Save")
    save_btn.click(
        fn=save_data_to_csv,
        inputs=[cap_diameter, stem_height, stem_width, gill_spacing, 
                does_bruise_bleed, has_ring, cap_shape, 
                surface, color, gill_attachment, stem_root, ring_type, 
                habitat, season],
        outputs=None
    )
        
    
    #tonen/verbergen van ring_type
    has_ring.change(
        fn=lambda x: gr.update(visible=x),
        inputs=has_ring,
        outputs=ring_type
    )
    # snel testen van verschillende modelen met het zelfde voorbeeld
    examples = [
        [10.0, 5.0, 10.0, ("close", 0), True, True, ("convex", 0), ("smooth", 5), ("brown", 2), ("adnate", 1), ("bulbous", 1), ("grooved", 0), ("woods", 0), ("spring", 3)],#poisonous 71.0%
        [8.0, 3.0, 8.0, ("distant", 1), True, True, ("bell", 3), ("shiny", 1), ("yellow", 6), ("decurrent", 2), ("club", 3), ("pendant", 1), ("urban", 7), ("autumn", 2)],#poisonous  67.0%
        [5.0, 2.0, 5.0, ("close", 0), True, True, ("spherical", 2), ("scaly", 3), ("red", 1), ("sinuate", 3), ("rooted", 2), ("flaring", 6), ("heaths", 3), ("winter", 0)],#poisonous 69.0%
    ]

    gr.Examples(
        examples=examples,
        inputs=[cap_diameter, stem_height, stem_width, gill_spacing, 
                does_bruise_bleed, has_ring, cap_shape, 
                surface, color, gill_attachment, stem_root, ring_type, 
                habitat, season]
    )

# Start de  app
if __name__ == "__main__":
     app.launch()