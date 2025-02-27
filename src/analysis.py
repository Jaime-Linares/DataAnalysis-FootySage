from src.fetch_data import get_competition_id_and_season_id, get_match_info
from src.data_preparation import code_categorical_data_multiclass, divide_data_in_train_test, scale_data_train_test
from sklearn.feature_selection import mutual_info_classif, SelectKBest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score
import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



# --- FUNCIONES COMUNES ----------------------------------------------------------------------------------------------------------------------------
def compute_shap_values(model, X_train, X_test, feature_names):
    '''
    Compute SHAP values for a multiclass classification model.
    params:
        model (object): Trained model.
        X_train (DataFrame): Training data.
        X_test (DataFrame): Test data.
        feature_names (list): List of feature names.
    returns:
        shap.Explanation: SHAP values computed for X_test.
    '''
    explainer = shap.Explainer(model.predict_proba, X_train, feature_names=feature_names)
    shap_values = explainer(X_test)
    return shap_values


def plot_shap_summary(shap_values, feature_names, encoder, threshold=0.005):
    '''
    Generate SHAP summary plot for each class in a multiclass classification model.
    params:
        shap_values (shap.Explanation): Computed SHAP values.
        feature_names (list): List of feature names.
        encoder (LabelEncoder): Encoder used to transform target labels.
        threshold (float): Importance threshold to filter features. Default is 0.005.
    returns:
        None : Prints feature importance and plots beeswarm charts
    '''
    for i in range(shap_values.shape[2]):
        class_name = encoder.inverse_transform([i])[0]
        print(f"Class {i}: {class_name}")

        # calculamos la media absoluta de cada característicae identificamos las características importantes
        shap_importance = np.abs(shap_values.values[:, :, i]).mean(axis=0)
        important_features = np.where(shap_importance > threshold)[0]
        unimportant_features = np.where(shap_importance <= threshold)[0]
        print(f"Important features ({len(important_features)}):")
        print([feature_names[idx] for idx in important_features])
        print(f"Unimportant features ({len(unimportant_features)}):")
        print([feature_names[idx] for idx in unimportant_features])

        # si hay características importantes, creamos un gráfico beeswarm
        if len(important_features) > 0:
            shap.plots.beeswarm(shap_values[:, important_features, i], max_display=important_features.size)
        else:
            print(f"No features with an impact greater than {threshold} for class {i}.")


def plot_shap_dependence_plots(shap_values, feature_names, X_test_original, encoder, num_features_to_plot=12, n_cols=3):
    '''
    Generate SHAP dependence plots for each class in a multiclass classification model.
    params:
        shap_values (shap.Explanation): SHAP values computed for the model.
        feature_names (list): List of feature names.
        X_test_original (DataFrame): Test dataset with original (unscaled) feature values.
        encoder (LabelEncoder): Encoder used to transform target labels.
        num_features_to_plot (int, optional): Number of top features to plot per class. Default is 12.
        n_cols (int, optional): Number of columns in the subplot grid. Default is 3.
    returns:
        None: Displays SHAP dependence plots for each class
    '''
    # calculamos el número de filas necesarias para mostrar todas las características en el número de columnas especificado
    n_rows = (num_features_to_plot // n_cols) + (num_features_to_plot % n_cols > 0)

    for class_idx in range(shap_values.shape[2]):
        class_name = encoder.inverse_transform([class_idx])[0]  
        print(f"\nGraphs for class: {class_name}\n")

        # seleccionamos las características más importantes para la clase actual
        shap_importance = np.abs(shap_values.values[:, :, class_idx]).mean(axis=0)
        top_features = np.argsort(shap_importance)[-num_features_to_plot:]

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        axes = axes.flatten()

        for i, feature_idx in enumerate(top_features):
            feature_name = feature_names[feature_idx]
            ax = axes[i]
            shap.dependence_plot(
                feature_name, 
                shap_values.values[:, :, class_idx],  
                X_test_original,  # usamos el conjunto de prueba original (sin escalar)
                feature_names=feature_names, 
                ax=ax,
                show=False
            )
            ax.set_title(f"{feature_name}")

        plt.suptitle(f"SHAP Dependence Plots - {class_name}", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()


def filter_dfs_by_team(X_test, X_test_orig, match_ids_test, team_name, competition_name, season_name, competition_gender):
    '''
    Filter the test set by a specific team.
    params:
        X_test (ndarray): Test feature set.
        X_test_orig (ndarray): Original test feature set before scaling.
        match_ids_test (ndarray): Array of match IDs for the test set.
        team_name (str): Name of the team to filter by.
        competition_name (str): Name of the competition.
        season_name (str): Name of the season
        competition_gender (str): The gender category of the competition (e.g., 'male', 'female').
    returns:
        X_test_team (ndarray): Test feature set filtered by the team.
        X_test_orig_team (ndarray): Original test feature set filtered by the team.
        team_match_ids (list): List of match IDs for the team.
    '''
    competition_id, season_id = get_competition_id_and_season_id(competition_name, competition_gender, season_name)
    team_match_ids = []
    for match_id in match_ids_test:
        match_info = get_match_info(competition_id, season_id, match_id)
        home_team = match_info['home_team'].values[0]
        away_team = match_info['away_team'].values[0]
        if team_name in [home_team, away_team]:
            team_match_ids.append(match_id)
    # creamos una máscara booleana y la aplicamos a los arrays
    mask = np.isin(match_ids_test, list(team_match_ids))
    X_test_team = X_test[mask]
    X_test_orig_team = X_test_orig[mask]
    return X_test_team, X_test_orig_team, team_match_ids


def force_plot_shap_team_matches(model, X_train, X_test_team, X_test_orig_team, feature_names, match_ids_test_team, encoder, team_name, competition_name, season_name, competition_gender):
    '''
    Generate SHAP force plots for matches of a specific team.
    params:
        model (object): Trained model.
        X_train (ndarray): Training data.
        X_test_team (ndarray): Test data for the team.
        X_test_orig_team (ndarray): Original test feature set before scaling for the team.
        feature_names (list): List of feature names.
        match_ids_test_team (list): List of match IDs for the team.
        encoder (LabelEncoder): Encoder used to transform target labels.
        team_name (str): Name of the team.
        competition_name (str): Name of the competition.
        season_name (str): Name of the season.
        competition_gender (str): The gender category of the competition (e.g., 'male', 'female').
    returns:
        None: Displays SHAP force plots for the team's matches.
    '''
    competition_id, season_id = get_competition_id_and_season_id(competition_name, competition_gender, season_name)
    explainer = shap.Explainer(model.predict_proba, X_train, feature_names=feature_names)
    shap_values_team = explainer(X_test_team)
    print(f"**Team analysis for {team_name} in {competition_name} {season_name} ({competition_gender})**")

    for i in range(X_test_team.shape[0]):
        # mostramos cierta información sobre el partido
        match_id = match_ids_test_team[i]
        match_info = get_match_info(competition_id, season_id, match_id)
        home_team = match_info['home_team'].values[0]
        home_score = match_info['home_score'].values[0]
        away_team = match_info['away_team'].values[0]
        away_score = match_info['away_score'].values[0]
        predicted_probs = model.predict_proba(X_test_team[i].reshape(1, -1))
        predicted_class_idx = np.argmax(predicted_probs, axis=1)[0]
        predicted_class_name = encoder.inverse_transform([predicted_class_idx])[0]
        print(f"📊 Match analysis with id: {match_id}")
        print(f"🏟️ {home_team} 🆚 {away_team}")
        print(f"📌 Real result: {home_score}-{away_score}")
        print(f"🤖 Prediction of the winning team of the model: {predicted_class_name}")
        print(f"🤖 Probabilities for each class: {predicted_probs}")

        # seleccionamos las características importantes que se van a mostrar en el gráfico de fuerza SHAP
        # para mostrarlas en un tabla y así poder analizarlas mejor
        features_force_plots = {}  
        for class_idx in range(predicted_probs.shape[1]):
            shap_exp = shap.Explanation(
                values=shap_values_team.values[i, :, class_idx], 
                base_values=shap_values_team.base_values[i, class_idx],
                data=X_test_team[i],
                feature_names=feature_names
            )
            for j in range(len(feature_names)):
                if abs(shap_exp.values[j]) > 0.005:  
                    features_force_plots[feature_names[j]] = X_test_orig_team[i][j]
        selected_features_df = pd.DataFrame(list(features_force_plots.items()), columns=["Feature", "Actual Value"])
        selected_features_df = selected_features_df.sort_values(by="Actual Value", ascending=False)
        print("🔎 Key Features Displayed in Force Plots: ")
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.axis('tight')
        ax.axis('off')
        ax.table(cellText=selected_features_df.values, colLabels=selected_features_df.columns, cellLoc='center', loc='center')
        plt.show()

        # mostramos el gráfico de fuerza SHAP para cada clase
        for class_idx in range(predicted_probs.shape[1]):
            class_name = encoder.inverse_transform([class_idx])[0]
            shap_exp = shap.Explanation(
                values=shap_values_team.values[i, :, class_idx], 
                base_values=shap_values_team.base_values[i, class_idx],
                data=X_test_team[i],
                feature_names=feature_names
            )
            shap.force_plot(
                shap_exp.base_values,
                shap_exp.values,
                shap_exp.data,
                feature_names=shap_exp.feature_names,
                matplotlib=True,
                text_rotation=80,
                show=False
            )
            plt.title(f"SHAP Force Plot for class: {class_name}", fontsize=15, pad=90)
            plt.show()


# --- FUNCIONES LA LIGA ----------------------------------------------------------------------------------------------------------------------------
def laliga_best_model(matches_in_laliga):
    '''
    Train and evaluate the best model (chosen during experimentation) for La Liga matches.
    params:
        matches_in_laliga (DataFrame): DataFrame containing match data for La Liga.
    returns:
        best_model (LogisticRegression): Trained Logistic Regression model.
        evaluation_metrics (DataFrame): DataFrame containing evaluation metrics.
        X_train_reduced (ndarray): Reduced training feature set.
        X_test_reduced (ndarray): Reduced test feature set.
        X_test_reduced_orig (ndarray): Original reduced test feature set before scaling.
        selected_columns (list): List of selected feature names.
        encoder (LabelEncoder): Encoder used to transform target labels.
        match_ids_test (ndarray): Array of match IDs for the test set.
    '''
    matches_df = matches_in_laliga.copy()
    X, y, encoder, match_ids = _preprocessing(matches_df)
    X_train, X_test, y_train, y_test, match_ids_train, match_ids_test = divide_data_in_train_test(X, y, match_ids)

    # selección de características usando Mutual Information
    selector = SelectKBest(lambda X, y: mutual_info_classif(X, y, random_state=666), k=50)
    X_train_reduced = selector.fit_transform(X_train, y_train)
    X_test_reduced = selector.transform(X_test)
    X_test_reduced_orig = X_test_reduced.copy()
    # obtenemos los nombres de las características seleccionadas
    selected_columns = X.columns[selector.get_support()].tolist()

    # entrenamiento del modelo (Logistic Regression, C=0.29354310869235306, penalty='l1', solver='saga')
    X_train_reduced, X_test_reduced = scale_data_train_test(X_train_reduced, X_test_reduced, "standard")
    best_model = LogisticRegression(random_state=42, max_iter=1000, C=0.29354310869235306, penalty='l1', solver='saga')
    best_model.fit(X_train_reduced, y_train)

    # predicciones en el conjunto de prueba reducido
    y_pred_reduced = best_model.predict(X_test_reduced)

    # calculamos las métricas de evaluación y mostramos los resultados
    evaluation_metrics = _show_metrics("Logistic Regression MI", best_model, X_train_reduced, X_test_reduced, y_train, y_test, y_pred_reduced)

    return best_model, evaluation_metrics, X_train_reduced, X_test_reduced, X_test_reduced_orig, selected_columns, encoder, match_ids_test


def laliga_global_analysis(best_model_LaLiga, feature_names_reduced_LaLiga, encoder_LaLiga):
    '''
    Perform a global analysis of LaLiga using the provided best model, reduced features, and encoder.
    params:
        best_model_LaLiga (LogisticRegression): The best trained model for LaLiga.
        feature_names_reduced_LaLiga (list): List of reduced feature names.
        encoder_LaLiga (LabelEncoder): Label encoder for LaLiga classes.
    returns:
        None: This function does not return any value. It generates bar charts showing the importance of features for each class.
    '''
    coef_matrix = best_model_LaLiga.coef_  # matriz de coeficientes (n_clases * n_features)
    class_labels = encoder_LaLiga.classes_
    num_features = len(feature_names_reduced_LaLiga)

    for idx, class_name in enumerate(class_labels):
        print(f"Class {idx}: {encoder_LaLiga.inverse_transform([idx])}")
        coef_importance = coef_matrix[idx]
        nonzero_indices = np.where(coef_importance != 0)[0]
        if len(nonzero_indices) == 0:
            print(f"No significant features for class {class_name}")
            continue
        sorted_indices = nonzero_indices[np.argsort(coef_importance[nonzero_indices])[::-1]]
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
    '''
    Preprocesses the given DataFrame by copying it, extracting match IDs, and encoding the target variable.
    params:
        matches_df_copy (DataFrame): A DataFrame containing match data with columns "match_id" and "winner_team".
    returns:
        tuple: A tuple containing:
            X (DataFrame): The feature matrix.
            y (Series): The encoded target variable.
            encoder (object): The encoder used for encoding the target variable.
            match_ids (ndarray): An array of match IDs.
    '''
    matches_df_copy = matches_df_copy.copy()
    match_ids = matches_df_copy["match_id"].values
    X = matches_df_copy.drop(columns=["winner_team", "match_id"])
    y = matches_df_copy["winner_team"]
    y, encoder = code_categorical_data_multiclass(y)
    return X, y, encoder, match_ids


def _show_metrics(model_name, best_model, X_train, X_test, y_train, y_test, y_pred):
    '''
    Computes and displays various performance metrics for a given model.
    params:
        model_name (str): The name of the model.
        best_model (object): The trained model object.
        X_train (ndarray): Training data features.
        X_test (ndarray): Test data features.
        y_train (ndarray): Training data labels.
        y_test (ndarray): Test data labels.
        y_pred (ndarray): Predicted labels for the test data.
    returns:
        DataFrame: A DataFrame containing the computed metrics.
    '''
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

