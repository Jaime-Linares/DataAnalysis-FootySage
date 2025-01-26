from src.data_preparation import code_categorical_data_multiclass, divide_data_in_train_test
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
import pandas as pd
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns



class ExperimentLauncher:

    def __init__(self, matches_df):
        self.matches_df = matches_df


    def run(self):
        print("Starting experiment...")
        print("Random Forest")
        rf_grid_search, rf_best_model, rf_eval = self.__random_forest_train_and_evaluate()
        print("Random Forest Selected Features")
        rf_reduced_eval = self.__random_forest_selected_features_train_and_evaluate(rf_grid_search, rf_best_model)
        print("Decision Tree")
        dt_grid_search, dt_best_model, dt_eval = self.__decision_tree_train_and_evaluate()
        print("Decision Tree Selected Features")
        dt_reduced_eval = self.__decision_tree_selected_features_train_and_evaluate(dt_grid_search, dt_best_model)
        print("Logistic Regression")
        lr_grid_search, lr_best_model, lr_eval = self.__logistic_regression_train_and_evaluate()
        print("Logistic Regression Selected Features")
        lr_reduced_eval = self.__logistic_regression_selected_features_train_and_evaluate(lr_grid_search, lr_best_model)
        print("KNN")
        knn_grid_search, knn_best_model, knn_eval = self.__knn_train_and_evaluate()
        print("KNN Selected Features")
        knn_reduced_eval = self.__knn_selected_features_train_and_evaluate(knn_grid_search, knn_best_model)
        print("Experiment finished.")
        return self.__show_results(rf_eval, rf_reduced_eval, dt_eval, dt_reduced_eval, lr_eval, lr_reduced_eval, knn_eval, knn_reduced_eval)


    def __random_forest_train_and_evaluate(self):
        X, y, encoder = self.__preprocessing("multiclass")
        X_train, X_test, y_train, y_test = self.__divide_data_in_train_and_test(X, y)

        # definimos un pipeline para el modelo RandomForestClassifier
        rf_pipeline = Pipeline([
            ('classifier', RandomForestClassifier(class_weight='balanced', random_state=42))
        ])

        # definimos el espacio de búsqueda de hiperparámetros
        rf_param_grid = {
            'classifier__n_estimators': [15, 20, 35, 40, 50],
            'classifier__max_depth': [3, 4, 5, 6, 8],
            'classifier__criterion': ['gini', 'entropy'],
            'classifier__max_features': ['sqrt', 'log2', None]
        }

        # realizamos la búsqueda de hiperparámetros
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        grid_search = GridSearchCV(rf_pipeline, rf_param_grid, cv=skf, scoring='f1_macro', verbose=1, n_jobs=-1)
        grid_search.fit(X_train, y_train)

        # mejores parámetros
        print("Best hyperparameters:", grid_search.best_params_)
        # mejor modelo
        rf_best_model = grid_search.best_estimator_

        # evaluación en el conjunto de entrenamiento y prueba
        train_accuracy = rf_best_model.score(X_train, y_train)
        #print(f"Train Accuracy: {train_accuracy:.4f}")
        test_accuracy = rf_best_model.score(X_test, y_test)
        #print(f"Test Accuracy: {test_accuracy:.4f}")

        # validación cruzada con el mejor modelo
        cv_scores = cross_val_score(rf_best_model, X_train, y_train, cv=skf, scoring='accuracy')
        #print(f"Cross-Validation Accuracy: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

        # predicciones en el conjunto de prueba
        y_pred = rf_best_model.predict(X_test)

        # matriz de confusión
        conf_matrix = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=encoder.classes_, yticklabels=encoder.classes_)
        plt.xlabel('Predicted label')
        plt.ylabel('True label')
        plt.title('Confusion Matrix - RandomForest')
        plt.show()

        # reporte de clasificación
        print(classification_report(y_test, y_pred))    

        # evaluación del modelos
        eval = {
            "train_accuracy": train_accuracy,
            "test_accuracy": test_accuracy,
            "cv_scores_mean": cv_scores.mean(),
            "cv_scores_std": cv_scores.std()
        }
        
        return grid_search, rf_best_model, eval
    

    def __random_forest_selected_features_train_and_evaluate(self, rf_grid_search, rf_best_model):
        X, y, encoder = self.__preprocessing("multiclass")

        # importancia de características
        feature_importances = pd.DataFrame({
            'Feature': X.columns,
            'Importance': rf_best_model.named_steps['classifier'].feature_importances_
        }).sort_values(by='Importance', ascending=False)

        # filtramos la características con importancia mayor a un umbral
        important_features = feature_importances[feature_importances['Importance'] > 0.0]['Feature']
        X_reduced = X[important_features]

        # dividimos los datos reducidos en entrenamiento y prueba
        X_train_reduced, X_test_reduced, y_train, y_test = divide_data_in_train_test(X_reduced, y, test_size=0.2)

        # entrenamos el modelo con las características reducidas
        rf_grid_search.fit(X_train_reduced, y_train)
        # mejores parámetros
        print("Best hyperparameters:", rf_grid_search.best_params_)
        # mejor modelo reducido
        rf_best_model_reduced = rf_grid_search.best_estimator_

        # evaluamos el modelo
        train_accuracy_reduced = rf_best_model_reduced.score(X_train_reduced, y_train)
        #print(f"Train Accuracy (Reduced): {train_accuracy_reduced:.4f}")
        test_accuracy_reduced = rf_best_model_reduced.score(X_test_reduced, y_test)
        #print(f"Test Accuracy (Reduced): {test_accuracy_reduced:.4f}")

        # validación cruzada
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores_reduced = cross_val_score(rf_best_model_reduced, X_train_reduced, y_train, cv=skf, scoring='accuracy')
        #print(f"Cross-Validation Accuracy (Reduced): {cv_scores_reduced.mean():.4f} +/- {cv_scores_reduced.std():.4f}")

        # predicciones en el conjunto de prueba
        y_pred_reduced = rf_best_model_reduced.predict(X_test_reduced)

        # matriz de confusión
        conf_matrix = confusion_matrix(y_test, y_pred_reduced)
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=encoder.classes_, yticklabels=encoder.classes_)
        plt.xlabel('Predicted label')
        plt.ylabel('True label')
        plt.title('Confusion Matrix - RandomForest Reduced')
        plt.show()   

        # reporte de clasificación
        print(classification_report(y_test, y_pred_reduced))

        # evaluación del modelos
        eval = {
            "train_accuracy": train_accuracy_reduced,
            "test_accuracy": test_accuracy_reduced,
            "cv_scores_mean": cv_scores_reduced.mean(),
            "cv_scores_std": cv_scores_reduced.std()
        }
        
        return eval
    

    def __decision_tree_train_and_evaluate(self):
        X, y, encoder = self.__preprocessing("multiclass")
        X_train, X_test, y_train, y_test = self.__divide_data_in_train_and_test(X, y)

        # definimos un pipeline para el modelo DecisionTreeClassifier
        dt_pipeline = Pipeline([
            ('classifier', DecisionTreeClassifier(random_state=42))
        ])

        # definimos el espacio de búsqueda de hiperparámetros
        dt_param_grid = {
            'classifier__max_depth': [3, 5, 7, 9, 12],
            'classifier__criterion': ['gini', 'entropy'],
            'classifier__max_features': ['sqrt', 'log2', None]
        }

        # realizamos la búsqueda de hiperparámetros
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        grid_search = GridSearchCV(dt_pipeline, dt_param_grid, cv=skf, scoring='f1_macro', verbose=1, n_jobs=-1)
        grid_search.fit(X_train, y_train)

        # mejores parámetros
        print("Best hyperparameters:", grid_search.best_params_)
        # mejor modelo
        dt_best_model = grid_search.best_estimator_

        # evaluación en el conjunto de entrenamiento y prueba
        train_accuracy = dt_best_model.score(X_train, y_train)
        #print(f"Train Accuracy: {train_accuracy:.4f}")
        test_accuracy = dt_best_model.score(X_test, y_test)
        #print(f"Test Accuracy: {test_accuracy:.4f}")

        # validación cruzada con el mejor modelo
        cv_scores = cross_val_score(dt_best_model, X_train, y_train, cv=skf, scoring='accuracy')
        #print(f"Cross-Validation Accuracy: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

        # predicciones en el conjunto de prueba
        y_pred = dt_best_model.predict(X_test)

        # matriz de confusión
        conf_matrix = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=encoder.classes_, yticklabels=encoder.classes_)
        plt.xlabel('Predicted label')
        plt.ylabel('True label')
        plt.title('Confusion Matrix - DecisionTree')
        plt.show()

        # reporte de clasificación
        print(classification_report(y_test, y_pred))   

        # evaluación del modelos
        eval = {
            "train_accuracy": train_accuracy,
            "test_accuracy": test_accuracy,
            "cv_scores_mean": cv_scores.mean(),
            "cv_scores_std": cv_scores.std()
        }
        
        return grid_search, dt_best_model, eval
    

    def __decision_tree_selected_features_train_and_evaluate(self, dt_grid_search, dt_best_model):
        X, y, encoder = self.__preprocessing("multiclass")

        # calculamos la información mutua para variables continuas con random_state
        mi_continuous = mutual_info_regression(X, y, random_state=42)

        # creamos un DataFrame para mostrar los resultados junto con los nombres de las columnas
        mi_results_mutual_information = pd.DataFrame({
            'Feature': X.columns,
            'Mutual Information': mi_continuous
        }).sort_values(by='Mutual Information', ascending=False)

        # filtramos las características con coeficientes mayores a un umbral
        important_features = mi_results_mutual_information[mi_results_mutual_information['Mutual Information'] > 0.025]['Feature']
        X_reduced = X[important_features]

        # dividimos los datos reducidos en entrenamiento y prueba
        X_train_reduced, X_test_reduced, y_train, y_test = divide_data_in_train_test(X_reduced, y, test_size=0.2)

        # definimos un pipeline para el modelo DecisionTreeClassifier
        dt_pipeline = Pipeline([
            ('classifier', DecisionTreeClassifier(random_state=42))
        ])

        # definimos el espacio de búsqueda de hiperparámetros
        dt_param_grid = {
            'classifier__max_depth': [2, 3, 5, 7, 9],
            'classifier__criterion': ['gini', 'entropy'],
            'classifier__max_features': ['sqrt', 'log2', None]
        }

        # realizamos la búsqueda de hiperparámetros
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        dt_grid_search = GridSearchCV(dt_pipeline, dt_param_grid, cv=skf, scoring='f1_macro', verbose=1, n_jobs=-1)
        dt_grid_search.fit(X_train_reduced, y_train)

        # entrenamos el modelo con las características reducidas
        #dt_grid_search.fit(X_train_reduced, y_train)
        # mejores parámetros
        print("Best hyperparameters:", dt_grid_search.best_params_)
        # mejor modelo reducido
        dt_best_model_reduced = dt_grid_search.best_estimator_

        # evaluamos el modelo
        train_accuracy_reduced = dt_best_model_reduced.score(X_train_reduced, y_train)
        #print(f"Train Accuracy (Reduced): {train_accuracy_reduced:.4f}")
        test_accuracy_reduced = dt_best_model_reduced.score(X_test_reduced, y_test)
        #print(f"Test Accuracy (Reduced): {test_accuracy_reduced:.4f}")

        # validación cruzada
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores_reduced = cross_val_score(dt_best_model_reduced, X_train_reduced, y_train, cv=skf, scoring='accuracy')
        #print(f"Cross-Validation Accuracy (Reduced): {cv_scores_reduced.mean():.4f} +/- {cv_scores_reduced.std():.4f}")

        # predicciones en el conjunto de prueba
        y_pred_reduced = dt_best_model_reduced.predict(X_test_reduced)

        # matriz de confusión
        conf_matrix = confusion_matrix(y_test, y_pred_reduced)
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=encoder.classes_, yticklabels=encoder.classes_)
        plt.xlabel('Predicted label')
        plt.ylabel('True label')
        plt.title('Confusion Matrix - DecisionTree Reduced')
        plt.show()   

        # reporte de clasificación
        print(classification_report(y_test, y_pred_reduced))

        # evaluación del modelos
        eval = {
            "train_accuracy": train_accuracy_reduced,
            "test_accuracy": test_accuracy_reduced,
            "cv_scores_mean": cv_scores_reduced.mean(),
            "cv_scores_std": cv_scores_reduced.std()
        }
        
        return eval
    

    def __logistic_regression_train_and_evaluate(self):
        X, y, encoder = self.__preprocessing("one-hot")
        X_train, X_test, y_train, y_test = self.__divide_data_in_train_and_test(X, y)

        # definimos un pipeline para el modelo LogisticRegression con StandardScaler
        lr_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(random_state=42, max_iter=1000))
        ])

        # definimos el espacio de búsqueda de hiperparámetros
        lr_param_grid = [
            {'classifier__penalty': ['l1'], 'classifier__solver': ['liblinear', 'saga'], 'classifier__C': [0.01, 0.1, 1, 10, 100]},
            {'classifier__penalty': ['l2'], 'classifier__solver': ['lbfgs', 'liblinear', 'saga', 'newton-cg'], 'classifier__C': [0.01, 0.1, 1, 10, 100]},
            {'classifier__penalty': ['elasticnet'], 'classifier__solver': ['saga'], 'classifier__C': [0.01, 0.1, 1, 10, 100], 'classifier__l1_ratio': [0.1, 0.5, 0.9]},
            {'classifier__penalty': [None], 'classifier__solver': ['lbfgs', 'saga', 'newton-cg']}
        ]

        # realizamos la búsqueda de hiperparámetros
        grid_search = GridSearchCV(lr_pipeline, lr_param_grid, cv=5, scoring='accuracy', verbose=1, n_jobs=-1)
        grid_search.fit(X_train, y_train)

        # mejores parámetros
        print("Best hyperparameters:", grid_search.best_params_)
        # mejor modelo
        lr_best_model = grid_search.best_estimator_

        # evaluación en el conjunto de entrenamiento y prueba
        train_accuracy = lr_best_model.score(X_train, y_train)
        #print(f"Train Accuracy: {train_accuracy:.4f}")
        test_accuracy = lr_best_model.score(X_test, y_test)
        #print(f"Test Accuracy: {test_accuracy:.4f}")

        # validación cruzada con el mejor modelo
        cv_scores = cross_val_score(lr_best_model, X_train, y_train, cv=5, scoring='accuracy')
        #print(f"Cross-Validation Accuracy: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

        # predicciones en el conjunto de prueba
        y_pred = lr_best_model.predict(X_test)

        # matriz de confusión
        conf_matrix = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=encoder.classes_, yticklabels=encoder.classes_)
        plt.xlabel('Predicted label')
        plt.ylabel('True label')
        plt.title('Confusion Matrix - Logistic Regression')
        plt.show()

        # reporte de clasificación
        print(classification_report(y_test, y_pred))  

        # evaluación del modelos
        eval = {
            "train_accuracy": train_accuracy,
            "test_accuracy": test_accuracy,
            "cv_scores_mean": cv_scores.mean(),
            "cv_scores_std": cv_scores.std()
        }
        
        return grid_search, lr_best_model, eval
    

    def __logistic_regression_selected_features_train_and_evaluate(self, lr_grid_search, lr_best_model):
        X, y, encoder = self.__preprocessing("one-hot")

        # coeficientes del modelo
        feature_importances = pd.DataFrame({
            'Feature': X.columns,
            'Coefficient': lr_best_model.named_steps['classifier'].coef_[0]
        }).sort_values(by='Coefficient', ascending=False)

        # filtramos las características con coeficientes mayores a un umbral
        important_features = feature_importances[feature_importances['Coefficient'].abs() > 0.001]['Feature']
        X_reduced = X[important_features]

        # dividimos los datos reducidos en entrenamiento y prueba
        X_train_reduced, X_test_reduced, y_train, y_test = divide_data_in_train_test(X_reduced, y, test_size=0.2)

        # entrenamos el modelo con las características reducidas
        lr_grid_search.fit(X_train_reduced, y_train)
        # mejores parámetros
        print("Best hyperparameters:", lr_grid_search.best_params_)
        # mejor modelo reducido
        lr_best_model_reduced = lr_grid_search.best_estimator_

        # evaluación del modelo reducido
        train_accuracy_reduced = lr_best_model_reduced.score(X_train_reduced, y_train)
        #print(f"Train Accuracy (Reduced): {train_accuracy_reduced:.4f}")
        test_accuracy_reduced = lr_best_model_reduced.score(X_test_reduced, y_test)
        #print(f"Test Accuracy (Reduced): {test_accuracy_reduced:.4f}")

        # validación cruzada con el modelo reducido
        cv_scores_reduced = cross_val_score(lr_best_model_reduced, X_train_reduced, y_train, cv=5, scoring='accuracy')
        #print(f"Cross-Validation Accuracy (Reduced): {cv_scores_reduced.mean():.4f} +/- {cv_scores_reduced.std():.4f}")

        # predicciones en el conjunto de prueba reducido
        y_pred_reduced = lr_best_model_reduced.predict(X_test_reduced)

        # matriz de confusión para modelo reducido
        conf_matrix = confusion_matrix(y_test, y_pred_reduced)
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=encoder.classes_, yticklabels=encoder.classes_)
        plt.xlabel('Predicted label')
        plt.ylabel('True label')
        plt.title('Confusion Matrix - Logistic Regression Reduced')
        plt.show()   

        # reporte de clasificación
        print(classification_report(y_test, y_pred_reduced))

        # evaluación del modelos
        eval = {
            "train_accuracy": train_accuracy_reduced,
            "test_accuracy": test_accuracy_reduced,
            "cv_scores_mean": cv_scores_reduced.mean(),
            "cv_scores_std": cv_scores_reduced.std()
        }
        
        return eval
    

    def __knn_train_and_evaluate(self):
        X, y, encoder = self.__preprocessing("multiclass")
        X_train, X_test, y_train, y_test = self.__divide_data_in_train_and_test(X, y)

        # definimos un pipeline para el modelo KNeighborsClassifier con StandardScaler
        knn_pipeline = Pipeline([
            ('scaler', MinMaxScaler()),
            ('classifier', KNeighborsClassifier())
        ])

        # definimos el espacio de búsqueda de hiperparámetros
        knn_param_grid = {
            'classifier__n_neighbors': [3, 5, 7, 9, 11, 13],
            'classifier__weights': ['uniform', 'distance'],
            'classifier__metric': ['euclidean', 'manhattan', 'minkowski']
        }

        # realizamos la búsqueda de hiperparámetros
        grid_search = GridSearchCV(knn_pipeline, knn_param_grid, cv=5, scoring='accuracy', verbose=1, n_jobs=-1)
        grid_search.fit(X_train, y_train)

        # mejores parámetros
        print("Best hyperparameters:", grid_search.best_params_)
        # mejor modelo
        knn_best_model = grid_search.best_estimator_

        # evaluación en el conjunto de entrenamiento y prueba
        train_accuracy = knn_best_model.score(X_train, y_train)
        #print(f"Train Accuracy: {train_accuracy:.4f}")
        test_accuracy = knn_best_model.score(X_test, y_test)
        #print(f"Test Accuracy: {test_accuracy:.4f}")

        # validación cruzada con el mejor modelo
        cv_scores = cross_val_score(knn_best_model, X_train, y_train, cv=5, scoring='accuracy')
        #print(f"Cross-Validation Accuracy: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

        # predicciones en el conjunto de prueba
        y_pred = knn_best_model.predict(X_test)

        # Matriz de confusión
        conf_matrix = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=encoder.classes_, yticklabels=encoder.classes_)
        plt.xlabel('Predicted label')
        plt.ylabel('True label')
        plt.title('Confusion Matrix - KNN')
        plt.show()

        # reporte de clasificación
        print(classification_report(y_test, y_pred))  

        # evaluación del modelos
        eval = {
            "train_accuracy": train_accuracy,
            "test_accuracy": test_accuracy,
            "cv_scores_mean": cv_scores.mean(),
            "cv_scores_std": cv_scores.std()
        }
        
        return grid_search, knn_best_model, eval
    

    def __knn_selected_features_train_and_evaluate(self, knn_grid_search, knn_best_model):
        X, y, encoder = self.__preprocessing("one-hot")

        # calculamos la información mutua para variables continuas con random_state
        mi_continuous = mutual_info_regression(X, y, random_state=42)

        # creamos un DataFrame para mostrar los resultados junto con los nombres de las columnas
        mi_results_mutual_information = pd.DataFrame({
            'Feature': X.columns,
            'Mutual Information': mi_continuous
        }).sort_values(by='Mutual Information', ascending=False)

        # filtramos las características con coeficientes mayores a un umbral
        important_features = mi_results_mutual_information[mi_results_mutual_information['Mutual Information'] > 0.025]['Feature']
        X_reduced = X[important_features]

        # dividimos los datos reducidos en entrenamiento y prueba
        X_train_reduced, X_test_reduced, y_train, y_test = divide_data_in_train_test(X_reduced, y, test_size=0.2)

        # dividimos los datos reducidos en entrenamiento y prueba
        X_train_reduced, X_test_reduced, y_train, y_test = divide_data_in_train_test(X_reduced, y, test_size=0.2)

        # entrenamos el modelo con las características reducidas
        knn_grid_search.fit(X_train_reduced, y_train)
        # mejores parámetros
        print("Best hyperparameters:", knn_grid_search.best_params_)
        # mejor modelo reducido
        knn_best_model_reduced = knn_grid_search.best_estimator_

        # evaluación del modelo reducido
        train_accuracy_reduced = knn_best_model_reduced.score(X_train_reduced, y_train)
        print(f"Train Accuracy (Reduced): {train_accuracy_reduced:.4f}")
        test_accuracy_reduced = knn_best_model_reduced.score(X_test_reduced, y_test)
        print(f"Test Accuracy (Reduced): {test_accuracy_reduced:.4f}")

        # validación cruzada con el modelo reducido
        cv_scores_reduced = cross_val_score(knn_best_model_reduced, X_train_reduced, y_train, cv=5, scoring='accuracy')
        print(f"Cross-Validation Accuracy (Reduced): {cv_scores_reduced.mean():.4f} +/- {cv_scores_reduced.std():.4f}")

        # predicciones en el conjunto de prueba reducido
        y_pred_reduced = knn_best_model_reduced.predict(X_test_reduced)

        # matriz de confusión para modelo reducido
        conf_matrix = confusion_matrix(y_test, y_pred_reduced)
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=encoder.classes_, yticklabels=encoder.classes_)
        plt.xlabel('Predicted label')
        plt.ylabel('True label')
        plt.title('Confusion Matrix - Logistic Regression Reduced')
        plt.show()   

        # reporte de clasificación
        print(classification_report(y_test, y_pred_reduced))

        # evaluación del modelos
        eval = {
            "train_accuracy": train_accuracy_reduced,
            "test_accuracy": test_accuracy_reduced,
            "cv_scores_mean": cv_scores_reduced.mean(),
            "cv_scores_std": cv_scores_reduced.std()
        }
        
        return eval
    

    def __preprocessing(self, type):
        matches_df = self.__get_matches()
        X = matches_df.drop(columns=["winner_team"])
        y = matches_df["winner_team"]

        if type == "multiclass":
            y, encoder = code_categorical_data_multiclass(y)
        elif type == "one-hot":
            y, encoder = code_categorical_data_multiclass(y)

        return X, y, encoder
    

    def __divide_data_in_train_and_test(self, X, y):
        X_train, X_test, y_train, y_test = divide_data_in_train_test(X, y, test_size=0.2)
        return X_train, X_test, y_train, y_test


    def __get_matches(self):
        return self.matches_df.copy()
    

    def __show_results(self, rf_eval, rf_reduced_eval, dt_eval, dt_reduced_eval, lr_eval, lr_reduced_eval, knn_eval, knn_reduced_eval):
        results = {
            "RF": rf_eval, "RF_Reduced": rf_reduced_eval,
            "DT": dt_eval, "DT_Reduced": dt_reduced_eval,
            "LR": lr_eval, "LR_Reduced": lr_reduced_eval,
            "KNN": knn_eval, "KNN_Reduced": knn_reduced_eval
        }
        results_df = pd.DataFrame(results).T
        results_df.index.name = "Model"
        return results_df
    
