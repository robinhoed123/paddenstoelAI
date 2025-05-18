import os
import time
import json
import numpy as np
import pandas as pd
from joblib import dump

from sklearn.tree import plot_tree

from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split, KFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.svm import SVC


# Seed
randstate = 42 #het was een goede film :)
np.random.seed(randstate)

# Map voor opslag van modellen
BASE_MODEL_DIR = 'models'
os.makedirs(BASE_MODEL_DIR, exist_ok=True)

# Functie om decision trees op te slaan
def save_decision_trees(clf_name, pipeline, model_dir, max_trees=3, tree_plot_max_depth=12):
    if clf_name not in ["RandomForest", "DecisionTree", "AdaBoost", "XGBoost", "LightGBM"]:
        return  

    classifier = pipeline.named_steps.get('classifier')
    trees_dir = os.path.join(model_dir, "decision_trees")
    os.makedirs(trees_dir, exist_ok=True)

    feature_names = None
    try:
        if hasattr(pipeline, 'feature_names_in_'):
            feature_names = list(pipeline.feature_names_in_)
    except Exception as e:
        print(f"Kon feature names niet ophalen voor {clf_name}: {e}")

    class_names = [str(c) for c in getattr(classifier, 'classes_', [])] if hasattr(classifier, 'classes_') else None

    trees_to_plot_info = []  # Lijst van (bestandsnaam, boom_estimator)

    if isinstance(classifier, DecisionTreeClassifier):
        trees_to_plot_info.append(('', classifier)) # Enkele boom
    elif hasattr(classifier, 'estimators_') and isinstance(classifier.estimators_, list):
        # RandomForest of has more trees
        plotted_count = 0
        for i, estimator in enumerate(classifier.estimators_):
            if plotted_count >= max_trees:
                break
            if isinstance(estimator, DecisionTreeClassifier):
                trees_to_plot_info.append((f'_tree_{i}', estimator))
                plotted_count += 1
    if not trees_to_plot_info:
        print(f"Geen bomen gevonden om te plotten voor {clf_name}.")
        return 
    plotted_actual_count = 0
    for tree_name, tree_estimator in trees_to_plot_info:
        plt.figure(figsize=(40, 20))
        try:
            plot_tree(tree_estimator,
                      filled=True,
                      feature_names=feature_names,
                      class_names=class_names,
                      rounded=True,
                      fontsize=8,
                      max_depth=tree_plot_max_depth)
            plt.title(f"{clf_name}{tree_name}")
            file_path = os.path.join(trees_dir, f"{clf_name.replace(' ', '_')}{tree_name}.svg")
            plt.savefig(file_path, format="svg", bbox_inches='tight')
            plotted_actual_count +=1
        except Exception as e:
            print(f"Fout bij het opslaan van tree {clf_name}{tree_name}: {e}")
        finally:
            plt.close()

# Functie om resultaten op te slaan
def save_model_info(clf_name, pipeline,y_test, y_test_pred, scores, training_time, params, base_dir=BASE_MODEL_DIR):
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
        "MAE on test set": f"{scores['test_mae']:.4f}",
        "Accuracy on test set": f"{scores['test_accuracy']:.4f}",
        "Cross-validation mean accuracy": f"{scores['cv_mean_accuracy']:.4f}",
        "Cross-validation std accuracy": f"{scores['cv_std_accuracy']:.4f}",
        "Training time (s)": f"{training_time:.2f}",
        "Tuned parameters": params,
    }
    
    # opslaan in JSON formaat
    info_path = os.path.join(model_dir, f"{clf_name}_info.json")
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=4)
    
    #confusion matrix 
    cm = confusion_matrix(y_test, y_test_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y_test))
    disp.plot(cmap='viridis')
    cm_path = os.path.join(model_dir, f"{clf_name}_confusion_matrix.png")
    disp.figure_.savefig(cm_path)
    plt.close()
    print(f"Confusion matrix opgeslagen in: {cm_path}")
    # Classification report
    report = classification_report(y_test, y_test_pred, target_names=np.unique(y_test).astype(str))
    report_path = os.path.join(model_dir, f"{clf_name}_classification_report.txt")
    with open(report_path, 'w') as f:
        f.write(report)
    
    # Decision trees
    save_decision_trees(clf_name, pipeline, model_dir)  
    print(f"Model en informatie opgeslagen in: {model_dir}")
    
def train_and_evaluate_classifier(X_train, X_test, y_train, y_test, 
                                 clf_name, classifier, param_grid):

    #Train en evalueer een classifier met pipeline en grid search
    
    print(f"\n{'='*50}")
    print(f"Training {clf_name} classifier...")
    print(f"{'='*50}")
    
    pipeline_steps = []
    
    
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
        verbose=2
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
    y_test_pred = best_pipeline.predict(X_test)
    
    # Bereken metrics
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    
    print(f"Test MAE: {test_mae:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")
    
    # Sla scores op
    scores = {
        'test_mae': test_mae,
        'test_accuracy': test_accuracy,
        'cv_mean_accuracy': cv_mean,
        'cv_std_accuracy': cv_std
    }
    
    # Sla model en informatie op
    save_model_info(clf_name,best_pipeline,y_test,y_test_pred,scores,training_time,best_params)
    
    
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
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=randstate
    )
    
    print(f"Training data: {X_train.shape}")
    print(f"Test data: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test


def main():
    """Hoofdfunctie om alle classifiers te trainen"""
    # Laad data
    print("Data wordt geladen en gesplitst...")
    X_train, X_test, y_train, y_test = load_data()
    
    # Definieer classifiers en parameter grids
    classifiers = {
        "RandomForest": {
            "clf": RandomForestClassifier(random_state=randstate),
            "params": {
                "classifier__n_estimators": [50, 100, 200],
                "classifier__max_depth": [ 10, 20, 30, 40, 50],
                "classifier__min_samples_split": [2, 5, 10],
                "classifier__min_samples_leaf": [1, 2, 4],
                "classifier__max_features": ["sqrt", "log2"],
                "classifier__bootstrap": [True, False],
            }
        },
        "SVM": { #staat in commentaar omdat het te traag is
            "clf": SVC(probability=True, random_state=randstate),
            "params": {
                "classifier__C": [0.1, 1, 10],
                "classifier__kernel": ["linear", "rbf"],
            }
        },
        "AdaBoost": {
            "clf": AdaBoostClassifier(random_state=randstate),
            "params": {
                "classifier__n_estimators": [50, 100, 200, 300, 400, 500],
                "classifier__learning_rate": [0.01, 0.1,0.5, 1.0],
            }
        },
        "DecisionTree": {
            "clf": DecisionTreeClassifier(random_state=randstate),
            "params": {
                "classifier__max_depth": [10, 20, 30,40,50],  
                "classifier__min_samples_leaf": [1, 2, 4],  # Minimum samples per blad(leaf): hogere waarden maken de boom minder gedetailleerd/complex
                "classifier__min_samples_split": [2, 5, 10],  # Minimum samples voor een splitsing: hogere waarden geven minder splitsingen(kleiner boom)

            }
        },
        "KNN": {
            "clf": KNeighborsClassifier(),
            "params": {
                "classifier__n_neighbors": [3, 5, 7, 10],  
                "classifier__weights": ["uniform", "distance"],
                "classifier__leaf_size": [10, 20, 30],  # Bladgrootte: beïnvloedt snelheid van zoeken naar buren
            }
        },
        "XGBoost": {
            "clf": XGBClassifier(random_state=randstate),
            "params": {
                "classifier__n_estimators": [50, 100, 200, 300, 400],
                "classifier__max_depth": [3, 6, 10],
                "classifier__learning_rate": [0.01, 0.1, 0.5, 1.0],  
            }
        },
        "LightGBM": {
            "clf": LGBMClassifier(random_state=randstate),
            "params": {
                "classifier__n_estimators": [50, 100, 200, 300, 400],
                "classifier__max_depth": [3, 6, 10],
                "classifier__learning_rate": [0.01, 0.1, 0.5, 1.0],
            }
        },

    }
    
    
    # Train en evalueer elke classifier
    results = {}
    for name, config in classifiers.items():
        print(f"\nTraining {name} classifier...")
        
        # Train en evalueer
        scores = train_and_evaluate_classifier(
            X_train, X_test, y_train, y_test,
            name, config["clf"], config["params"]
        )
        

if __name__ == "__main__":
    main()