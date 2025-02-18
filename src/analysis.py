from src.data_preparation import code_categorical_data_multiclass, divide_data_in_train_test, scale_data_train_test
from sklearn.feature_selection import mutual_info_classif, SelectKBest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



# --- FUNCIONES LA LIGA ----------------------------------------------------------------------------------------------------------------------------
def laliga_best_model(matches_in_laliga):
    matches_df = matches_in_laliga.copy()
    X, y, encoder, match_ids = _preprocessing(matches_df)
    X_train, X_test, y_train, y_test, match_ids_train, match_ids_test = divide_data_in_train_test(X, y, match_ids)

    # selección de características usando Mutual Information
    selector = SelectKBest(lambda X, y: mutual_info_classif(X, y, random_state=666), k=50)
    X_train_reduced = selector.fit_transform(X_train, y_train)
    X_test_reduced = selector.transform(X_test)
    # obtenemos los nombres de las características seleccionadas
    selected_columns = X.columns[selector.get_support()]

    # entrenamiento del modelo (Logistic Regression, C=0.29354310869235306, penalty='l1', solver='saga')
    X_train_reduced, X_test_reduced = scale_data_train_test(X_train_reduced, X_test_reduced, "standard")
    best_model = LogisticRegression(random_state=42, max_iter=1000, C=0.29354310869235306, penalty='l1', solver='saga')
    best_model.fit(X_train_reduced, y_train)

    # predicciones en el conjunto de prueba reducido
    y_pred_reduced = best_model.predict(X_test_reduced)

    # calculamos las métricas de evaluación y mostramos los resultados
    evaluation_metrics = _show_metrics("Logistic Regression MI", best_model, X_train_reduced, X_test_reduced, y_train, y_test, y_pred_reduced)

    return best_model, evaluation_metrics, X_train_reduced, X_test_reduced, selected_columns, encoder, match_ids_test


def laliga_global_analysis(best_model_LaLiga, feature_names_reduced_LaLiga, encoder_LaLiga):
    coef_matrix = best_model_LaLiga.coef_  # matriz de coeficientes (n_clases * n_features)
    class_labels = encoder_LaLiga.classes_
    num_features = len(feature_names_reduced_LaLiga)

    for idx, class_name in enumerate(class_labels):
        coef_importance = coef_matrix[idx]  # coeficientes de la clase actual
        sorted_indices = np.argsort(coef_importance)[::-1]
        sorted_features = [feature_names_reduced_LaLiga[i] for i in sorted_indices]
        sorted_importance = coef_importance[sorted_indices]
        # creamos gráfico para la clase
        plt.figure(figsize=(15, max(8, num_features * 0.3)))
        plt.barh(sorted_features, sorted_importance, color='darkorange')
        plt.xlabel("Coefficient value", fontsize=14)
        plt.ylabel("Features", fontsize=14)
        plt.title(f"Importance of characteristics when the winner is: {class_name}", fontsize=16)
        plt.gca().invert_yaxis()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.show()


# --- FUNCIONES AUXILIARES ----------------------------------------------------------------------------------------------------------------------------
def _preprocessing(matches_df_copy):
    matches_df_copy = matches_df_copy.copy()
    match_ids = matches_df_copy["match_id"].values
    X = matches_df_copy.drop(columns=["winner_team", "match_id"])
    y = matches_df_copy["winner_team"]
    y, encoder = code_categorical_data_multiclass(y)
    return X, y, encoder, match_ids


def _show_metrics(model_name, best_model, X_train, X_test, y_train, y_test, y_pred):
    train_accuracy = best_model.score(X_train, y_train)
    test_accuracy = best_model.score(X_test, y_test)
    precision_macro = precision_score(y_test, y_pred, average='macro')
    precision_weighted = precision_score(y_test, y_pred, average='weighted')
    recall_macro = recall_score(y_test, y_pred, average='macro')
    recall_weighted = recall_score(y_test, y_pred, average='weighted')
    f1_macro = f1_score(y_test, y_pred, average='macro')
    f1_weighted = f1_score(y_test, y_pred, average='weighted')
    metrics = {'Train Accuracy': train_accuracy, 'Test Accuracy': test_accuracy, 'Precision Macro': precision_macro, 'Precision Weighted': precision_weighted,
            'Recall Macro': recall_macro, 'Recall Weighted': recall_weighted, 'F1 Macro': f1_macro, 'F1 Weighted': f1_weighted}
    models = [model_name]
    results_df = pd.DataFrame(metrics, index=models)
    return results_df

