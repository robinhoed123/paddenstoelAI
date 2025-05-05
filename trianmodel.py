import os
import time
import json
import numpy as np
import pandas as pd
from datetime import datetime
from joblib import dump, load

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split, KFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

# Seed voor reproduceerbaarheid
randstate = 42
np.random.seed(randstate)

# Map voor opslag van modellen
BASE_MODEL_DIR = 'models'
os.makedirs(BASE_MODEL_DIR, exist_ok=True)

# Functie om resultaten op te slaan
def save_model_info(clf_name, pipeline, scores, training_time, params, base_dir=BASE_MODEL_DIR):
    """Sla model en bijbehorende informatie op"""
    # Maak directory voor deze classifier
    model_dir = os.path.join(base_dir, clf_name)
    os.makedirs(model_dir, exist_ok=True)
    
    # Sla model op
    model_path = os.path.join(model_dir, f"{clf_name}_model.joblib")
    dump(pipeline, model_path)
    

    info = {
        "Method": clf_name,
        "Pre-processing": str(pipeline.named_steps.get('preprocessor', 'None')),
        "Data splits": "5-fold cross validation",
        "Feature extraction": str(pipeline.named_steps.get('feature_engineer', 'None')),
        "MAE on validation set (mm)": f"{scores['val_mae']:.4f}",
        "MAE on test set": f"{scores['test_mae']:.4f}",
        "Accuracy on validation set": f"{scores['val_accuracy']:.4f}",
        "Accuracy on test set": f"{scores['test_accuracy']:.4f}",
        "Cross-validation mean accuracy": f"{scores['cv_mean_accuracy']:.4f}",
        "Cross-validation std accuracy": f"{scores['cv_std_accuracy']:.4f}",
        "Training time (s)": f"{training_time:.2f}",
        "Tuned parameters": params,
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # opslaan in JSON formaat
    info_path = os.path.join(model_dir, f"{clf_name}_info.json")
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=4)
    
    print(f"Model en informatie opgeslagen in: {model_dir}")

def train_and_evaluate_classifier(X_train, X_val, X_test, y_train, y_val, y_test, 
                                 clf_name, classifier, param_grid, preprocessor='standard'):
    """Train en evalueer een classifier met pipeline en grid search"""
    
    print(f"\n{'='*50}")
    print(f"Training {clf_name} classifier...")
    print(f"{'='*50}")
    
    # Definieer preprocessing
    if preprocessor == 'standard':
        preprocessor_step = StandardScaler()
    elif preprocessor == 'minmax':
        preprocessor_step = MinMaxScaler()
    else:
        preprocessor_step = None
    
    # pipeline
    pipeline_steps = []
    if preprocessor_step:
        pipeline_steps.append(('preprocessor', preprocessor_step))
    
    # Voeg classifier toe
    pipeline_steps.append(('classifier', classifier))
    
    # pipeline
    pipeline = Pipeline(pipeline_steps)
    
    # KFold cross-validator
    kfold = KFold(n_splits=5, shuffle=True, random_state=randstate)
    
    # Grid search met cross-validation
    start_time = time.time()
    
    # Pas grid search toe
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=kfold,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    
    # Train het model
    grid_search.fit(X_train, y_train)
    
    # Bereken training tijd
    training_time = time.time() - start_time
    
    # Beste model en parameters
    best_pipeline = grid_search.best_estimator_
    best_params = grid_search.best_params_
    
    # Cross-validation scores
    cv_scores = cross_val_score(best_pipeline, X_train, y_train, cv=kfold, scoring='accuracy')
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()
    
    print(f"\nBeste parameters: {best_params}")
    print(f"Cross-validation accuracy: {cv_mean:.4f} ± {cv_std:.4f}")
    
    # Voorspellingen
    y_val_pred = best_pipeline.predict(X_val)
    y_test_pred = best_pipeline.predict(X_test)
    
    # Bereken metrics
    val_mae = mean_absolute_error(y_val, y_val_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    
    print(f"\nValidatie MAE: {val_mae:.4f}")
    print(f"Test MAE: {test_mae:.4f}")
    print(f"Validatie accuracy: {val_accuracy:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")
    
    # Sla scores op
    scores = {
        'val_mae': val_mae,
        'test_mae': test_mae,
        'val_accuracy': val_accuracy,
        'test_accuracy': test_accuracy,
        'cv_mean_accuracy': cv_mean,
        'cv_std_accuracy': cv_std
    }
    
    # Sla model en informatie op
    save_model_info(clf_name, best_pipeline, scores, training_time, best_params)
    
    return best_pipeline, scores


def load_data():

    # Laad de dataset
    df = pd.read_csv('MushroomDataset/Mushroomdataclean.csv')
    print(df)
    y = df.iloc[:, 0]
    X = df.drop(df.columns[0], axis=1)
    print(X)
    print(y)
    # Split data
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.2, random_state=randstate
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.25, random_state=randstate
    )
    
    print(f"Training data: {X_train.shape}")
    print(f"Validatie data: {X_val.shape}")
    print(f"Test data: {X_test.shape}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def main():
    """Hoofdfunctie om alle classifiers te trainen"""
    
    # Laad data
    print("Data wordt geladen en gesplitst...")
    X_train, X_val, X_test, y_train, y_val, y_test = load_data()
    
    # Definieer classifiers en parameter grids
    classifiers = {
        "RandomForest": {
            "clf": RandomForestClassifier(random_state=randstate),
            "params": {
                "classifier__n_estimators": [50, 100, 200],
                "classifier__max_depth": [None, 10, 20],
            }
        },
        # "SVM": {
        #     "clf": SVC(probability=True, random_state=randstate),
        #     "params": {
        #         "classifier__C": [0.1, 1, 10],
        #         "classifier__kernel": ["linear", "rbf"],
        #     }
        # },
        "AdaBoost": {
            "clf": AdaBoostClassifier(random_state=randstate),
            "params": {
                "classifier__n_estimators": [50, 100, 200],
                "classifier__learning_rate": [0.01, 0.1, 1.0],
            }
        },
        "DecisionTree": {
            "clf": DecisionTreeClassifier(random_state=randstate),
            "params": {
                "classifier__max_depth": [None, 10, 20, 30],
                "classifier__min_samples_split": [2, 5, 10],
            }
        },
        "KNN": {
            "clf": KNeighborsClassifier(),
            "params": {
                "classifier__n_neighbors": [3, 5, 7, 10],
                "classifier__weights": ["uniform", "distance"],
            }
        },
        "MLP": {
            "clf": MLPClassifier(random_state=randstate, max_iter=200),
            "params": {
                "classifier__hidden_layer_sizes": [(50,), (100,), (50, 50)],
                "classifier__alpha": [0.0001, 0.001, 0.01],
                "classifier__batch_size": [32, 64, 128],  # Batch size voor batch normalization effect
            }
        }
    }
    
    # Train en evalueer elke classifier
    results = {}
    for name, config in classifiers.items():
        print(f"\nTraining {name} classifier...")
        
        # Bepaal geschikte preprocessor
        preprocessor = 'standard'
        if name == 'KNN' or name == 'SVM':
            preprocessor = 'minmax'
        elif name == 'MLP':  # Gebruik batch normalization voor MLP
            preprocessor = 'batch_norm'
        
        # Train en evalueer
        best_pipeline, scores = train_and_evaluate_classifier(
            X_train, X_val, X_test, y_train, y_val, y_test,
            name, config["clf"], config["params"], preprocessor
        )
        
        results[name] = scores
    
    # Overzicht van resultaten
    print("\n\n" + "="*80)
    print("OVERZICHT VAN RESULTATEN")
    print("="*80)
    
    # Maak DataFrame voor makkelijke vergelijking
    results_df = pd.DataFrame({
        'Classifier': list(results.keys()),
        'Val MAE': [results[k]['val_mae'] for k in results],
        'Test MAE': [results[k]['test_mae'] for k in results],
        'Val Accuracy': [results[k]['val_accuracy'] for k in results],
        'Test Accuracy': [results[k]['test_accuracy'] for k in results],
        'CV Mean Accuracy': [results[k]['cv_mean_accuracy'] for k in results],
    })
    
    # Sorteer op Test Accuracy
    results_df = results_df.sort_values('Test Accuracy', ascending=False)
    
    print(results_df.to_string(index=False))
    print("\nAlle modellen zijn getraind en opgeslagen in:", BASE_MODEL_DIR)

if __name__ == "__main__":
    main()