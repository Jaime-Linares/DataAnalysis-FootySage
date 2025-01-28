from src.data_preparation import code_categorical_data_multiclass, scale_data_train_test, divide_data_in_train_test
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, classification_report
import pandas as pd
from sklearn.feature_selection import mutual_info_regression, SelectFromModel
import matplotlib.pyplot as plt
import seaborn as sns



class ExperimentLauncher:

    def __init__(self, matches_df):
        self.matches_df = matches_df
        self.train_accuracy = [None] * 8
        self.test_accuracy = [None] * 8
        self.precision_macro = [None] * 8
        self.precision_weighted = [None] * 8
        self.recall_macro = [None] * 8
        self.recall_weighted = [None] * 8
        self.f1_macro = [None] * 8
        self.f1_weighted = [None] * 8
        self.hyperparameters = [None] * 8


    def run(self):
        print("Starting experiment...")
        print("Random Forest")
        rf_best_model = self.__random_forest_train_and_evaluate(0)
        print("Random Forest Selected Features")
        self.__random_forest_selected_features_train_and_evaluate(1, rf_best_model)
        print("Decision Tree")
        dt_best_model = self.__decision_tree_train_and_evaluate(2)
        print("Decision Tree Selected Features")
        self.__decision_tree_selected_features_train_and_evaluate(3, dt_best_model)
        print("Logistic Regression")
        lr_best_model = self.__logistic_regression_train_and_evaluate(4)
        print("Logistic Regression Selected Features")
        self.__logistic_regression_selected_features_train_and_evaluate(5, lr_best_model)
        print("KNN")
        knn_best_model = self.__knn_train_and_evaluate(6)
        print("KNN Selected Features")
        self.__knn_selected_features_train_and_evaluate(7, knn_best_model)
        print("Experiment finished.")
        return self.__show_results()


    def __random_forest_train_and_evaluate(self, position):
        X, y, encoder = self.__preprocessing()
        X_train, X_test, y_train, y_test = divide_data_in_train_test(X, y)

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
        best_params = {k.replace('classifier__', ''): v for k, v in grid_search.best_params_.items()}
        print("Best hyperparameters:", best_params)
        # mejor modelo
        rf_best_model = RandomForestClassifier(**best_params, class_weight='balanced', random_state=42)
        rf_best_model.fit(X_train, y_train)

        # predicciones en el conjunto de prueba
        y_pred = rf_best_model.predict(X_test)

        # calculamos las métricas de evaluación y las añadimos a las listas
        self.__calculate_and_add_metrics(position, rf_best_model, X_train, X_test, y_train, y_test, y_pred, best_params)

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
        
        return rf_best_model
    

    def __random_forest_selected_features_train_and_evaluate(self, position, rf_best_model):
        X, y, encoder = self.__preprocessing()

        # importancia de características
        feature_importances = pd.DataFrame({
            'Feature': X.columns,
            'Importance': rf_best_model.feature_importances_
        }).sort_values(by='Importance', ascending=False)
        # filtramos la características con importancia mayor a un umbral
        important_features = feature_importances[feature_importances['Importance'] > 0.0]['Feature']
        X_reduced = X[important_features]

        # dividimos los datos reducidos en entrenamiento y prueba
        X_train_reduced, X_test_reduced, y_train, y_test = divide_data_in_train_test(X_reduced, y)

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
        grid_search.fit(X_train_reduced, y_train)

        # mejores parámetros
        best_params_reduced = {k.replace('classifier__', ''): v for k, v in grid_search.best_params_.items()}
        print("Best hyperparameters:", best_params_reduced)
        # mejor modelo reducido
        rf_best_model_reduced = RandomForestClassifier(**best_params_reduced, class_weight='balanced', random_state=42)
        rf_best_model_reduced.fit(X_train_reduced, y_train)

        # predicciones en el conjunto de prueba
        y_pred_reduced = rf_best_model_reduced.predict(X_test_reduced)

        # calculamos las métricas de evaluación y las añadimos a las listas
        self.__calculate_and_add_metrics(position, rf_best_model_reduced, X_train_reduced, X_test_reduced, y_train, y_test, y_pred_reduced, best_params_reduced)

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
    

    def __decision_tree_train_and_evaluate(self, position):
        X, y, encoder = self.__preprocessing()
        X_train, X_test, y_train, y_test = divide_data_in_train_test(X, y)

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
        best_params = {k.replace('classifier__', ''): v for k, v in grid_search.best_params_.items()}
        print("Best hyperparameters:", best_params)
        # mejor modelo
        dt_best_model = DecisionTreeClassifier(**best_params, random_state=42)
        dt_best_model.fit(X_train, y_train)

        # predicciones en el conjunto de prueba
        y_pred = dt_best_model.predict(X_test)

        # calculamos las métricas de evaluación y las añadimos a las listas
        self.__calculate_and_add_metrics(position, dt_best_model, X_train, X_test, y_train, y_test, y_pred, best_params)

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
        
        return dt_best_model
    

    def __decision_tree_selected_features_train_and_evaluate(self, position, dt_best_model):
        X, y, encoder = self.__preprocessing()

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
        X_train_reduced, X_test_reduced, y_train, y_test = divide_data_in_train_test(X_reduced, y)

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
        grid_search = GridSearchCV(dt_pipeline, dt_param_grid, cv=skf, scoring='f1_macro', verbose=1, n_jobs=-1)
        grid_search.fit(X_train_reduced, y_train)

        # mejores parámetros
        best_params_reduced = {k.replace('classifier__', ''): v for k, v in grid_search.best_params_.items()}
        print("Best hyperparameters:", best_params_reduced)
        # mejor modelo reducido
        dt_best_model_reduced = DecisionTreeClassifier(**best_params_reduced, random_state=42)
        dt_best_model_reduced.fit(X_train_reduced, y_train)

        # predicciones en el conjunto de prueba
        y_pred_reduced = dt_best_model_reduced.predict(X_test_reduced)

        # calculamos las métricas de evaluación y las añadimos a las listas
        self.__calculate_and_add_metrics(position, dt_best_model_reduced, X_train_reduced, X_test_reduced, y_train, y_test, y_pred_reduced, best_params_reduced)

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
    

    def __logistic_regression_train_and_evaluate(self, position):
        X, y, encoder = self.__preprocessing()
        X_train, X_test, y_train, y_test = divide_data_in_train_test(X, y)

        # definimos un pipeline para el modelo LogisticRegression con StandardScaler
        lr_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(random_state=42, max_iter=1000))
        ])

        # definimos el espacio de búsqueda de hiperparámetros
        lr_param_grid = [
            {'classifier__penalty': ['l1'], 'classifier__solver': ['saga'], 'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100]},
            {'classifier__penalty': ['l2'], 'classifier__solver': ['lbfgs', 'saga', 'newton-cg'], 'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100]},
            {'classifier__penalty': ['elasticnet'], 'classifier__solver': ['saga'], 'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100], 'classifier__l1_ratio': [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]},
            {'classifier__penalty': [None], 'classifier__solver': ['lbfgs', 'saga', 'newton-cg']}
        ]

        # realizamos la búsqueda de hiperparámetros
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        grid_search = GridSearchCV(lr_pipeline, lr_param_grid, cv=skf, scoring='f1_macro', verbose=1, n_jobs=-1)
        grid_search.fit(X_train, y_train)

        # mejores parámetros
        best_params = {k.replace('classifier__', ''): v for k, v in grid_search.best_params_.items()}
        print("Best hyperparameters:", best_params)
        # mejor modelo
        X_train, X_test = scale_data_train_test(X_train, X_test, "standard")
        lr_best_model = LogisticRegression(**best_params, random_state=42, max_iter=3000)
        lr_best_model.fit(X_train, y_train)

        # predicciones en el conjunto de prueba
        y_pred = lr_best_model.predict(X_test)

        # calculamos las métricas de evaluación y las añadimos a las listas
        self.__calculate_and_add_metrics(position, lr_best_model, X_train, X_test, y_train, y_test, y_pred, best_params)

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
        
        return lr_best_model
    

    def __logistic_regression_selected_features_train_and_evaluate(self, position, lr_best_model):
        X, y, encoder = self.__preprocessing()

        selector = SelectFromModel(estimator=LogisticRegression(penalty='elasticnet', solver='saga', random_state=42, max_iter=1000, l1_ratio=0.2, C=0.1), threshold='1.6*mean')
        X_reduced = selector.fit_transform(X, y)

        # dividimos los datos reducidos en entrenamiento y prueba
        X_train_reduced, X_test_reduced, y_train, y_test = divide_data_in_train_test(X_reduced, y)

        # definimos un pipeline para el modelo LogisticRegression con StandardScaler
        lr_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(random_state=42, max_iter=25))
        ])

        # definimos el espacio de búsqueda de hiperparámetros
        lr_param_grid = [
            {'classifier__penalty': ['l1'], 'classifier__solver': ['saga'], 'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100]},
            {'classifier__penalty': ['l2'], 'classifier__solver': ['lbfgs', 'saga', 'newton-cg'], 'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100]},
            {'classifier__penalty': ['elasticnet'], 'classifier__solver': ['saga'], 'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100], 'classifier__l1_ratio': [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]},
            {'classifier__penalty': [None], 'classifier__solver': ['lbfgs', 'saga', 'newton-cg']}
        ]

        # realizamos la búsqueda de hiperparámetros
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        grid_search = GridSearchCV(lr_pipeline, lr_param_grid, cv=skf, scoring='f1_macro', verbose=1, n_jobs=-1)
        grid_search.fit(X_train_reduced, y_train)

        # mejores parámetros
        best_params_reduced = {k.replace('classifier__', ''): v for k, v in grid_search.best_params_.items()}
        print("Best hyperparameters:", best_params_reduced)
        # mejor modelo reducido
        X_train_reduced, X_test_reduced = scale_data_train_test(X_train_reduced, X_test_reduced, "standard")
        lr_best_model_reduced = LogisticRegression(**best_params_reduced, random_state=42, max_iter=1000)
        lr_best_model_reduced.fit(X_train_reduced, y_train)

        # predicciones en el conjunto de prueba reducido
        y_pred_reduced = lr_best_model_reduced.predict(X_test_reduced)

        # calculamos las métricas de evaluación y las añadimos a las listas
        self.__calculate_and_add_metrics(position, lr_best_model_reduced, X_train_reduced, X_test_reduced, y_train, y_test, y_pred_reduced, best_params_reduced)

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
    

    def __knn_train_and_evaluate(self, position):
        X, y, encoder = self.__preprocessing()
        X_train, X_test, y_train, y_test = divide_data_in_train_test(X, y)

        # definimos un pipeline para el modelo KNeighborsClassifier con StandardScaler
        knn_pipeline = Pipeline([
            ('scaler', MinMaxScaler()),
            ('classifier', KNeighborsClassifier())
        ])

        # definimos el espacio de búsqueda de hiperparámetros
        knn_param_grid = {
            'classifier__n_neighbors': [5, 7, 9, 11, 13, 15, 17, 20, 23],
            'classifier__weights': ['uniform', 'distance'],
            'classifier__metric': ['euclidean', 'manhattan', 'minkowski', 'chebyshev']
        }

        # realizamos la búsqueda de hiperparámetros
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        grid_search = GridSearchCV(knn_pipeline, knn_param_grid, cv=skf, scoring='f1_macro', verbose=1, n_jobs=-1)
        grid_search.fit(X_train, y_train)

        # mejores parámetros
        best_params = {k.replace('classifier__', ''): v for k, v in grid_search.best_params_.items()}
        print("Best hyperparameters:", best_params)
        # mejor modelo
        X_train, X_test = scale_data_train_test(X_train, X_test, "minmax")
        knn_best_model = KNeighborsClassifier(**best_params)
        knn_best_model.fit(X_train, y_train)

        # predicciones en el conjunto de prueba
        y_pred = knn_best_model.predict(X_test)

        # calculamos las métricas de evaluación y las añadimos a las listas
        self.__calculate_and_add_metrics(position, knn_best_model, X_train, X_test, y_train, y_test, y_pred, best_params)

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
        
        return knn_best_model
    

    def __knn_selected_features_train_and_evaluate(self, position, knn_best_model):
        X, y, encoder = self.__preprocessing()

        # calculamos la información mutua para variables continuas con random_state
        mi_continuous = mutual_info_regression(X, y, random_state=42)

        # creamos un DataFrame para mostrar los resultados junto con los nombres de las columnas
        mi_results_mutual_information = pd.DataFrame({
            'Feature': X.columns,
            'Mutual Information': mi_continuous
        }).sort_values(by='Mutual Information', ascending=False)

        # filtramos las características con coeficientes mayores a un umbral
        important_features = mi_results_mutual_information[mi_results_mutual_information['Mutual Information'] > 0.03]['Feature']
        X_reduced = X[important_features]

        # dividimos los datos reducidos en entrenamiento y prueba
        X_train_reduced, X_test_reduced, y_train, y_test = divide_data_in_train_test(X_reduced, y)

         # definimos un pipeline para el modelo KNeighborsClassifier con StandardScaler
        knn_pipeline = Pipeline([
            ('scaler', MinMaxScaler()),
            ('classifier', KNeighborsClassifier())
        ])

        # definimos el espacio de búsqueda de hiperparámetros
        knn_param_grid = {
            'classifier__n_neighbors': [5, 7, 9, 11, 13, 15, 17, 20, 23],
            'classifier__weights': ['uniform', 'distance'],
            'classifier__metric': ['euclidean', 'manhattan', 'minkowski', 'chebyshev']
        }

        # realizamos la búsqueda de hiperparámetros
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        grid_search = GridSearchCV(knn_pipeline, knn_param_grid, cv=skf, scoring='f1_macro', verbose=1, n_jobs=-1)
        grid_search.fit(X_train_reduced, y_train)

        # mejores parámetros
        best_params_reduced = {k.replace('classifier__', ''): v for k, v in grid_search.best_params_.items()}
        print("Best hyperparameters:", best_params_reduced)
        # mejor modelo reducido
        X_train_reduced, X_test_reduced = scale_data_train_test(X_train_reduced, X_test_reduced, "minmax")
        knn_best_model_reduced = KNeighborsClassifier(**best_params_reduced)
        knn_best_model_reduced.fit(X_train_reduced, y_train)

        # predicciones en el conjunto de prueba reducido
        y_pred_reduced = knn_best_model_reduced.predict(X_test_reduced)

        # calculamos las métricas de evaluación y las añadimos a las listas
        self.__calculate_and_add_metrics(position, knn_best_model_reduced, X_train_reduced, X_test_reduced, y_train, y_test, y_pred_reduced, best_params_reduced)

        # matriz de confusión para modelo reducido
        conf_matrix = confusion_matrix(y_test, y_pred_reduced)
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=encoder.classes_, yticklabels=encoder.classes_)
        plt.xlabel('Predicted label')
        plt.ylabel('True label')
        plt.title('Confusion Matrix - KNN Reduced')
        plt.show()   

        # reporte de clasificación
        print(classification_report(y_test, y_pred_reduced))
    

    def __preprocessing(self):
        matches_df = self.matches_df.copy()
        X = matches_df.drop(columns=["winner_team"])
        y = matches_df["winner_team"]
        y, encoder = code_categorical_data_multiclass(y)
        return X, y, encoder  
    

    def __calculate_and_add_metrics(self, position, model, X_train, X_test, y_train, y_test, y_pred, hyperparameters):
        self.train_accuracy[position] = model.score(X_train, y_train)
        self.test_accuracy[position] = model.score(X_test, y_test)
        self.precision_macro[position] = precision_score(y_test, y_pred, average='macro')
        self.precision_weighted[position] = precision_score(y_test, y_pred, average='weighted')
        self.recall_macro[position] = recall_score(y_test, y_pred, average='macro')
        self.recall_weighted[position] = recall_score(y_test, y_pred, average='weighted')
        self.f1_macro[position] = f1_score(y_test, y_pred, average='macro')
        self.f1_weighted[position] = f1_score(y_test, y_pred, average='weighted')
        self.hyperparameters[position] = hyperparameters
    

    def __show_results(self):
        metrics = {
            'Train Accuracy': self.train_accuracy,
            'Test Accuracy': self.test_accuracy,
            'Precision Macro': self.precision_macro,
            'Precision Weighted': self.precision_weighted,
            'Recall Macro': self.recall_macro,
            'Recall Weighted': self.recall_weighted,
            'F1 Macro': self.f1_macro,
            'F1 Weighted': self.f1_weighted,
            'Hyperparameters chosen': self.hyperparameters
        }
        models = ['Random Forest', 'Random Forest Reduced', 'Decision Tree', 'Decision Tree Reduced', 'Logistic Regression', 'Logistic Regression Reduced', 'KNN', 'KNN Reduced']
        results_df = pd.DataFrame(metrics, index=models)
        return results_df
    
