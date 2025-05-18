import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
import datetime
import kerastuner as kt

# Constante voor reproduceerbaarheid
randstate = 42
np.random.seed(randstate)
tf.random.set_seed(randstate)

def load_data():

    # Laad de dataset
    df = pd.read_csv('MushroomDataset/Mushroomdataclean.csv')
    print("Dataset geladen met vorm:", df.shape)
    
    # Split target en features
    y = df.iloc[:, 0]  # Target kolom (eetbaar of giftig)
    X = df.drop(df.columns[0], axis=1)  # Feature kolommen
    
    print(f"Features vorm: {X.shape}")
    print(f"Target vorm: {y.shape}")
    
    # Split data in train+validatie en test sets (80/20)
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.2, random_state=randstate
    )
    
    # Split train/validatie verder in train en validatie sets (75/25)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.25, random_state=randstate
    )
    
    print(f"Training data: {X_train.shape}")
    print(f"Validatie data: {X_val.shape}")
    print(f"Test data: {X_test.shape}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def create_model(hp):
    model = keras.Sequential()
    
    # Input laag
    model.add(layers.Input(shape=(14,)))
    
    # Voeg verborgen lagen toe met instelbare eenheden
    for i in range(hp.Int('num_layers', 1, 3)):
        units = hp.Int(f'units_{i}', min_value=16, max_value=128, step=16)
        model.add(layers.Dense(units, activation='relu'))
        
        # Batch normalization
        model.add(layers.BatchNormalization())
        
        # Dropout
        dropout_rate = hp.Float(f'dropout_{i}', min_value=0.1, max_value=0.5, step=0.1)
        model.add(layers.Dropout(dropout_rate))
    
    # Output laag (binaire classificatie)
    model.add(layers.Dense(1, activation='sigmoid'))
    
    #learning rate
    learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log')
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def build_standard_model():
    model = keras.Sequential([
        layers.Input(shape=(14,)),
        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(32, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_model(model, X_train, y_train, X_val, y_val, batch_size=32, epochs=50):
    """
    # Model trainen met callbacks
    """
    # Maak callback voor TensorBoard logs
    log_dir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, 
        histogram_freq=1,
        write_graph=True,
        update_freq='epoch'
    )
    
    # Early stopping callback
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
    
    # ModelCheckpoint callback om beste model op te slaan
    checkpoint_path = "model_checkpoints/best_model.h5"
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        save_best_only=True,
        monitor='val_accuracy',
        mode='max',
        verbose=1
    )
    
    # Train het model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=[tensorboard_callback, early_stopping, model_checkpoint],
        verbose=1
    )
    
    return history, model

def tune_hyperparameters(X_train, y_train, X_val, y_val):
    """
    # Hyperparameters optimaliseren met KerasTuner
    """
    print("Hyperparameter tuning starten...")
    
    # Maak tuner object
    tuner = kt.Hyperband(
        create_model,
        objective='val_accuracy',
        max_epochs=30,
        factor=3,
        directory='kerastuner_dir',
        project_name='mushroom_classification'
    )
    
    # Maak callback voor early stopping tijdens tuning
    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
    
    # Start tuning proces
    tuner.search(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=32,
        callbacks=[stop_early]
    )
    
    # Krijg beste hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    print("Beste hyperparameters:")
    print(f"- Aantal lagen: {best_hps.get('num_layers')}")
    for i in range(best_hps.get('num_layers')):
        print(f"- Laag {i+1}:")
        print(f"  - Eenheden: {best_hps.get(f'units_{i}')}")
        print(f"  - Dropout rate: {best_hps.get(f'dropout_{i}')}")
    print(f"- Learning rate: {best_hps.get('learning_rate')}")
    
    # Bouw model met beste hyperparameters
    best_model = tuner.hypermodel.build(best_hps)
    
    return best_model

def evaluate_model(model, X_test, y_test):
    """
    # Model evalueren op test set
    """
    # Evalueer het model
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=1)
    print(f"Test nauwkeurigheid: {test_acc:.4f}")
    print(f"Test verlies: {test_loss:.4f}")
    
    # Voorspellingen maken
    y_pred = model.predict(X_test)
    y_pred_classes = [1 if p > 0.5 else 0 for p in y_pred]
    
    return test_acc, test_loss, y_pred_classes

def plot_history(history):
    """
    # Leerproces visualiseren
    """
    plt.figure(figsize=(12, 5))
    
    # Plot nauwkeurigheid
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model nauwkeurigheid')
    plt.ylabel('Nauwkeurigheid')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validatie'], loc='lower right')
    
    # Plot verlies
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model verlies')
    plt.ylabel('Verlies')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validatie'], loc='upper right')
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

def save_model(model, filepath="mushroom_model"):
    """
    # Model opslaan
    """
    # Sla model op in SavedModel formaat
    model.save(filepath)
    print(f"Model opgeslagen in: {filepath}")
    
    # Sla model op in h5 formaat
    model.save(f"{filepath}.h5")
    print(f"Model opgeslagen in: {filepath}.h5")

def main():
    """
    # Hoofdfunctie die het hele proces uitvoert
    """
    # Laad en split data
    X_train, X_val, X_test, y_train, y_val, y_test = load_data()
    
    # Voer hyperparameter tuning uit (uitcommentariëren als je dit wilt overslaan)
    # best_model = tune_hyperparameters(X_train, y_train, X_val, y_val)
    
    # Of gebruik een standaard model zonder tuning
    print(X_train.shape[1])
    best_model = build_standard_model()
    
    # Train het model
    history, trained_model = train_model(best_model, X_train, y_train, X_val, y_val)
    
    # Evalueer het model
    test_acc, test_loss, y_pred_classes = evaluate_model(trained_model, X_test, y_test)
    
    # Visualiseer de trainingsgeschiedenis
    plot_history(history)
    
    # Sla het model op
    save_model(trained_model)
    
    print("Project succesvol uitgevoerd!")

if __name__ == "__main__":
    main()