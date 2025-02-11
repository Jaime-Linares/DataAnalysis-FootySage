from src.data_preparation import code_categorical_data_multiclass, scale_data_train_test, divide_data_in_train_test
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, RandomizedSearchCV
import scipy.stats as stats
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.feature_selection import mutual_info_classif, SelectKBest
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE, BorderlineSMOTE
from sklearn.utils.class_weight import compute_sample_weight
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



class ExperimentLauncher:

    def __init__(self, matches_df):
        self.matches_df = matches_df
        self.train_accuracy = [None] * 14
        self.test_accuracy = [None] * 14
        self.precision_macro = [None] * 14
        self.precision_weighted = [None] * 14
        self.recall_macro = [None] * 14
        self.recall_weighted = [None] * 14
        self.f1_macro = [None] * 14
        self.f1_weighted = [None] * 14
        self.hyperparameters = [None] * 14


    def run(self):
        print("Starting experiment...")
        print("Random Forest")
        self.__random_forest_train_and_evaluate(0)
        print("Random Forest Oversampling")
        self.__random_forest_oversampling_train_and_evaluate(1)
        print("Random Forest Selected Features")
        self.__random_forest_selected_features_train_and_evaluate(2)
        print("Random Forest MI")
        self.__random_forest_MI_train_and_evaluate(3)
        print("Decision Tree")
        self.__decision_tree_train_and_evaluate(4)
        print("Decision Tree Oversampling")
        self.__decision_tree_oversampling_train_and_evaluate(5)
        print("Decision Tree MI")
        self.__decision_tree_MI_train_and_evaluate(6)
        print("Logistic Regression")
        self.__logistic_regression_train_and_evaluate(7)
        print("Logistic Regression Oversampling")
        self.__logistic_regression_oversampling_train_and_evaluate(8)
        print("Logistic Regression MI")
        self.__logistic_regression_MI_train_and_evaluate(9)
        print("Logistic Regression PCA")
        self.__logistic_regression_PCA_train_and_evaluate(10)
        print("KNN")
        self.__knn_train_and_evaluate(11)
        print("KNN Selected Features")
        self.__knn_MI_train_and_evaluate(12)
        print("KNN PCA")
        self.__knn_PCA_train_and_evaluate(13)
        print("Experiment finished.")
        return self.__show_results()


    def __random_forest_train_and_evaluate(self, position):
        X, y, encoder = self.__preprocessing()
        X_train, X_test, y_train, y_test = divide_data_in_train_test(X, y)

        # definimos un pipeline para el modelo RandomForestClassifier
        pipeline = Pipeline([
            ('classifier', RandomForestClassifier(class_weight='balanced', random_state=42))
        ])

        # definimos el espacio de búsqueda de hiperparámetros con distribuciones aleatorias
        param_dist = {
            'classifier__n_estimators': stats.randint(10, 75),  
            'classifier__max_depth': stats.randint(2, 6),       
            'classifier__criterion': ['gini', 'entropy'],        
            'classifier__max_features': ['sqrt', 'log2', None] 
        }

        # realizamos la búsqueda aleatoria de hiperparámetros
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        random_search = RandomizedSearchCV(pipeline, param_dist, n_iter=150, cv=skf, scoring='f1_macro', verbose=1, n_jobs=-1, random_state=42)
        random_search.fit(X_train, y_train)

        # mejores hiperparámetros
        best_params = {k.replace('classifier__', ''): v for k, v in random_search.best_params_.items()}
        print("Best hyperparameters:", best_params)
        # mejor modelo
        best_model = RandomForestClassifier(**best_params, class_weight='balanced', random_state=42)
        best_model.fit(X_train, y_train)

        # predicciones en el conjunto de prueba
        y_pred = best_model.predict(X_test)

        # calculamos las métricas de evaluación y las añadimos a las listas
        self.__calculate_and_add_metrics(position, best_model, X_train, X_test, y_train, y_test, y_pred, best_params)

        # matriz de confusión y reporte de clasificación
        self.__confusion_matrix_and_report('Random Forest', y_test, y_pred, encoder)
    

    def __random_forest_oversampling_train_and_evaluate(self, position):
        X, y, encoder = self.__preprocessing()
        X_train, X_test, y_train, y_test = divide_data_in_train_test(X, y)

        # aumentamos los datos (oversampling) aplicando SMOTE solo al conjunto de entrenamiento
        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

        # calculamos los pesos de las instancias basados en las frecuencias de clase
        sample_weights = compute_sample_weight(class_weight='balanced', y=y_train_resampled)

        # definimos un pipeline para el modelo RandomForestClassifier
        pipeline = Pipeline([
            ('classifier', RandomForestClassifier(random_state=42))
        ])

        # definimos el espacio de búsqueda de hiperparámetros con distribuciones aleatorias
        param_dist = {
            'classifier__n_estimators': stats.randint(10, 66),  
            'classifier__max_depth': stats.randint(2, 7),       
            'classifier__criterion': ['gini', 'entropy'],
            'classifier__max_features': ['sqrt', 'log2', None]
        }

        # realizamos la búsqueda aleatoria de hiperparámetros
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        random_search = RandomizedSearchCV(pipeline, param_dist, n_iter=150, cv=skf, scoring='f1_macro', verbose=1, n_jobs=-1, random_state=42)
        random_search.fit(X_train_resampled, y_train_resampled, classifier__sample_weight=sample_weights)

        # mejores hiperparámetros
        best_params = {k.replace('classifier__', ''): v for k, v in random_search.best_params_.items()}
        print("Best hyperparameters:", best_params)
        # mejor modelo
        best_model = RandomForestClassifier(**best_params, random_state=42)
        best_model.fit(X_train_resampled, y_train_resampled, sample_weight=sample_weights)

        # predicciones en el conjunto de prueba
        y_pred = best_model.predict(X_test)

        # calculamos las métricas de evaluación y las añadimos a las listas
        self.__calculate_and_add_metrics(position, best_model, X_train_resampled, X_test, y_train_resampled, y_test, y_pred, best_params)

        # matriz de confusión y reporte de clasificación
        self.__confusion_matrix_and_report('Random Forest Oversampling', y_test, y_pred, encoder)  
    

    def __random_forest_selected_features_train_and_evaluate(self, position):
        X, y, encoder = self.__preprocessing()
        X_train, X_test, y_train, y_test = divide_data_in_train_test(X, y)

        # modelo base para calcular la importancia de características
        model = RandomForestClassifier(class_weight='balanced', random_state=42, criterion='gini', max_depth=5, n_estimators=41, max_features='sqrt')
        model.fit(X_train, y_train)

        # importancia de características
        feature_importances = pd.DataFrame({
            'Feature': X.columns,
            'Importance': model.feature_importances_
        }).sort_values(by='Importance', ascending=False)

        # filtramos las características con importancia mayor a un umbral
        important_features = feature_importances[feature_importances['Importance'] > 0.0]['Feature']
        X_train_reduced = X_train[important_features]
        X_test_reduced = X_test[important_features]

        # definimos un pipeline para el modelo RandomForestClassifier
        pipeline = Pipeline([
            ('classifier', RandomForestClassifier(class_weight='balanced', random_state=42))
        ])

        # definimos el espacio de búsqueda de hiperparámetros con distribuciones aleatorias
        param_dist = {
            'classifier__n_estimators': stats.randint(10, 61),  
            'classifier__max_depth': stats.randint(2, 7),       
            'classifier__criterion': ['gini', 'entropy'],
            'classifier__max_features': ['sqrt', 'log2']
        }

        # realizamos la búsqueda aleatoria de hiperparámetros
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        random_search = RandomizedSearchCV(pipeline, param_dist, n_iter=300, cv=skf, scoring='f1_macro', verbose=1, n_jobs=-1, random_state=42)
        random_search.fit(X_train_reduced, y_train)

        # mejores hiperparámetros
        best_params_reduced = {k.replace('classifier__', ''): v for k, v in random_search.best_params_.items()}
        print("Best hyperparameters:", best_params_reduced)
        # mejor modelo reducido
        best_model_reduced = RandomForestClassifier(**best_params_reduced, class_weight='balanced', random_state=42)
        best_model_reduced.fit(X_train_reduced, y_train)

        # predicciones en el conjunto de prueba
        y_pred_reduced = best_model_reduced.predict(X_test_reduced)

        # calculamos las métricas de evaluación y las añadimos a las listas
        self.__calculate_and_add_metrics(position, best_model_reduced, X_train_reduced, X_test_reduced, y_train, y_test, y_pred_reduced, best_params_reduced)

        # matriz de confusión y reporte de clasificación
        self.__confusion_matrix_and_report('Random Forest Reduced', y_test, y_pred_reduced, encoder)
    

    def __random_forest_MI_train_and_evaluate(self, position):
        X, y, encoder = self.__preprocessing()
        X_train, X_test, y_train, y_test = divide_data_in_train_test(X, y)

        # selección de características usando Mutual Information
        mi_scores = mutual_info_classif(X_train, y_train, random_state=42) 
        selector = SelectKBest(score_func=lambda X, y: (mi_scores, []), k=50)  
        X_train_reduced = selector.fit_transform(X_train, y_train)
        X_test_reduced = selector.transform(X_test)

        # calculamos los pesos de las instancias basados en las frecuencias de clase
        sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)

        # definimos un pipeline para RandomForest
        pipeline = Pipeline([
            ('classifier', RandomForestClassifier(random_state=42))
        ])

        # espacio de búsqueda de hiperparámetros aleatorios
        param_dist = {
            'classifier__n_estimators': stats.randint(25, 100),
            'classifier__max_depth': stats.randint(2, 7), 
            'classifier__criterion': ['gini', 'entropy'],
            'classifier__max_features': ['sqrt', 'log2']
        }

        # realizamos la búsqueda aleatoria de hiperparámetros
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        random_search = RandomizedSearchCV(pipeline, param_dist, n_iter=100, cv=skf, scoring='f1_macro', verbose=1, n_jobs=-1, random_state=42)
        random_search.fit(X_train_reduced, y_train, classifier__sample_weight=sample_weights)

        # mejores hiperparámetros
        best_params = {k.replace('classifier__', ''): v for k, v in random_search.best_params_.items()}
        print("Best hyperparameters:", best_params)
        # entrenamos el modelo final con las mejores características seleccionadas
        best_model_mi = RandomForestClassifier(**best_params, random_state=42)
        best_model_mi.fit(X_train_reduced, y_train, sample_weight=sample_weights)

        # predicciones en el conjunto de prueba
        y_pred = best_model_mi.predict(X_test_reduced)

        # calculamos las métricas de evaluación y mostramos resultados
        self.__calculate_and_add_metrics(position, best_model_mi, X_train_reduced, X_test_reduced, y_train, y_test, y_pred, best_params)

        # matriz de confusión y reporte de clasificación
        self.__confusion_matrix_and_report('Random Forest MI', y_test, y_pred, encoder)


    def __decision_tree_train_and_evaluate(self, position):
        X, y, encoder = self.__preprocessing()
        X_train, X_test, y_train, y_test = divide_data_in_train_test(X, y)

        # definimos un pipeline para el modelo DecisionTreeClassifier
        pipeline = Pipeline([
            ('classifier', DecisionTreeClassifier(random_state=42))
        ])

        # definimos el espacio de búsqueda de hiperparámetros aleatorios
        param_dist = {
            'classifier__max_depth': stats.randint(2, 7),
            'classifier__criterion': ['gini', 'entropy'],
            'classifier__max_features': ['sqrt', 'log2', None],
            'classifier__min_samples_split': stats.randint(2, 20),
            'classifier__min_samples_leaf': stats.randint(1, 10)
        }

        # realizamos la búsqueda aleatoria de hiperparámetros
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        random_search = RandomizedSearchCV(pipeline, param_dist, n_iter=100, cv=skf, scoring='f1_macro', verbose=1, n_jobs=-1, random_state=42)
        random_search.fit(X_train, y_train)

        # mejores hiperparámetros
        best_params = {k.replace('classifier__', ''): v for k, v in random_search.best_params_.items()}
        print("Best hyperparameters:", best_params)
        # mejor modelo con los hiperparámetros encontrados
        best_model = DecisionTreeClassifier(**best_params, random_state=42)
        best_model.fit(X_train, y_train)

        # predicciones en el conjunto de prueba
        y_pred = best_model.predict(X_test)

        # calculamos las métricas de evaluación y las añadimos a las listas
        self.__calculate_and_add_metrics(position, best_model, X_train, X_test, y_train, y_test, y_pred, best_params)

        # matriz de confusión y reporte de clasificación
        self.__confusion_matrix_and_report('Decision Tree', y_test, y_pred, encoder)


    def __decision_tree_oversampling_train_and_evaluate(self, position):
        X, y, encoder = self.__preprocessing()
        X_train, X_test, y_train, y_test = divide_data_in_train_test(X, y)  

        # aumentamos los datos (oversampling) aplicando SMOTE solo al conjunto de entrenamiento
        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

        # definimos un pipeline para el modelo DecisionTreeClassifier
        pipeline = Pipeline([
            ('classifier', DecisionTreeClassifier(random_state=42))
        ])

        # definimos el espacio de búsqueda de hiperparámetros aleatorios
        param_dist = {
            'classifier__max_depth': stats.randint(2, 7),
            'classifier__criterion': ['gini', 'entropy'],
            'classifier__max_features': ['sqrt', 'log2', None],
            'classifier__min_samples_split': stats.randint(2, 16),
            'classifier__min_samples_leaf': stats.randint(1, 8)
        }

        # realizamos la búsqueda aleatoria de hiperparámetros
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        random_search = RandomizedSearchCV(pipeline, param_dist, n_iter=500, cv=skf, scoring='f1_macro', verbose=1, n_jobs=-1, random_state=42)
        random_search.fit(X_train_resampled, y_train_resampled)

        # mejores hiperparámetros
        best_params = {k.replace('classifier__', ''): v for k, v in random_search.best_params_.items()}
        print("Best hyperparameters:", best_params)
        # mejor modelo con los hiperparámetros encontrados
        best_model = DecisionTreeClassifier(**best_params, random_state=42)
        best_model.fit(X_train_resampled, y_train_resampled)

        # predicciones en el conjunto de prueba
        y_pred = best_model.predict(X_test)

        # calculamos las métricas de evaluación y las añadimos a las listas
        self.__calculate_and_add_metrics(position, best_model, X_train_resampled, X_test, y_train_resampled, y_test, y_pred, best_params)

        # matriz de confusión y reporte de clasificación
        self.__confusion_matrix_and_report('Decision Tree Oversampling', y_test, y_pred, encoder)
    

    def __decision_tree_MI_train_and_evaluate(self, position):
        X, y, encoder = self.__preprocessing()
        X_train, X_test, y_train, y_test = divide_data_in_train_test(X, y) 

        # calculamos MI
        mi_classification = mutual_info_classif(X_train, y_train, random_state=42)
        important_features = X_train.columns[mi_classification > 0.01]
        X_train_reduced = X_train[important_features]
        X_test_reduced = X_test[important_features]

        # definimos un pipeline para el modelo DecisionTreeClassifier
        pipeline = Pipeline([
            ('classifier', DecisionTreeClassifier(random_state=42))
        ])

        # definimos el espacio de búsqueda aleatoria de hiperparámetros
        param_dist = {
            'classifier__max_depth': stats.randint(2, 10),              
            'classifier__criterion': ['gini', 'entropy'],
            'classifier__max_features': ['sqrt', 'log2', None]   
        }

        # realizamos la búsqueda aleatoria de hiperparámetros
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        random_search = RandomizedSearchCV(pipeline, param_dist, n_iter=100, cv=skf, scoring='f1_macro', verbose=1, n_jobs=-1, random_state=42)
        random_search.fit(X_train_reduced, y_train)

        # mejores hiperparámetros
        best_params_reduced = {k.replace('classifier__', ''): v for k, v in random_search.best_params_.items()}
        print("Best hyperparameters:", best_params_reduced)
        # mejor modelo reducido
        best_model_reduced = DecisionTreeClassifier(**best_params_reduced, random_state=42)
        best_model_reduced.fit(X_train_reduced, y_train)

        # predicciones en el conjunto de prueba
        y_pred_reduced = best_model_reduced.predict(X_test_reduced)

        # calculamos las métricas de evaluación y las añadimos a las listas
        self.__calculate_and_add_metrics(position, best_model_reduced, X_train_reduced, X_test_reduced, y_train, y_test, y_pred_reduced, best_params_reduced)

        # matriz de confusión y reporte de clasificación
        self.__confusion_matrix_and_report('Decision Tree MI', y_test, y_pred_reduced, encoder)
    

    def __logistic_regression_train_and_evaluate(self, position):
        X, y, encoder = self.__preprocessing()
        X_train, X_test, y_train, y_test = divide_data_in_train_test(X, y)

        # definimos un pipeline para el modelo LogisticRegression con StandardScaler
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced'))
        ])

        # definimos el espacio de búsqueda aleatoria de hiperparámetros
        param_dist = [
            {'classifier__penalty': ['l1'], 'classifier__solver': ['saga'], 'classifier__C': stats.uniform(0.5, 1.51)},
            {'classifier__penalty': ['l2'], 'classifier__solver': ['lbfgs', 'saga', 'newton-cg'], 'classifier__C': stats.uniform(0.25, 1.51)},
            {'classifier__penalty': ['elasticnet'], 'classifier__solver': ['saga'], 'classifier__C': stats.uniform(0.3, 1.51), 'classifier__l1_ratio': stats.uniform(0.2, 0.61)},
            {'classifier__penalty': [None], 'classifier__solver': ['lbfgs', 'saga', 'newton-cg']}
        ]

        # realizamos la búsqueda aleatoria de hiperparámetros
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        random_search = RandomizedSearchCV(pipeline, param_distributions=param_dist, n_iter=100, cv=skf, scoring='f1_macro', verbose=1, n_jobs=-1, random_state=42)
        random_search.fit(X_train, y_train)

        # mejores hiperparámetros
        best_params = {k.replace('classifier__', ''): v for k, v in random_search.best_params_.items()}
        print("Best hyperparameters:", best_params)
        # mejor modelo
        X_train, X_test = scale_data_train_test(X_train, X_test, "standard")
        best_model = LogisticRegression(**best_params, random_state=42, max_iter=3000)
        best_model.fit(X_train, y_train)

        # predicciones en el conjunto de prueba
        y_pred = best_model.predict(X_test)

        # calculamos las métricas de evaluación y las añadimos a las listas
        self.__calculate_and_add_metrics(position, best_model, X_train, X_test, y_train, y_test, y_pred, best_params)

        # matriz de confusión y reporte de clasificación
        self.__confusion_matrix_and_report('Logistic Regression', y_test, y_pred, encoder)


    def __logistic_regression_oversampling_train_and_evaluate(self, position):
        X, y, encoder = self.__preprocessing()
        X_train, X_test, y_train, y_test = divide_data_in_train_test(X, y)

        # aumentamos los datos (oversampling) aplicando SMOTE solo al conjunto de entrenamiento
        smote = BorderlineSMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

        # calcular los pesos de las instancias basados en las frecuencias de clase
        sample_weights = compute_sample_weight(class_weight='balanced', y=y_train_resampled)

        # definimos un pipeline para el modelo LogisticRegression con StandardScaler
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(random_state=42, max_iter=3000))
        ])

        # definimos el espacio de búsqueda aleatoria de hiperparámetros
        param_dist = [
            {'classifier__penalty': ['l1'], 'classifier__solver': ['saga'], 'classifier__C': stats.uniform(0.25, 1.51)},
            {'classifier__penalty': ['l2'], 'classifier__solver': ['lbfgs', 'saga', 'newton-cg'], 'classifier__C': stats.uniform(0.25, 1.25)},
            {'classifier__penalty': ['elasticnet'], 'classifier__solver': ['saga'], 'classifier__C': stats.uniform(0.3, 1.25), 'classifier__l1_ratio': stats.uniform(0.2, 0.61)},
            {'classifier__penalty': [None], 'classifier__solver': ['lbfgs', 'saga', 'newton-cg']}
        ]

        # realizamos la búsqueda aleatoria de hiperparámetros
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        random_search = RandomizedSearchCV(pipeline, param_distributions=param_dist, n_iter=100, cv=skf, scoring='f1_macro', verbose=1, n_jobs=-1, random_state=42)
        random_search.fit(X_train_resampled, y_train_resampled, classifier__sample_weight=sample_weights)

        # mejores hiperparámetros
        best_params = {k.replace('classifier__', ''): v for k, v in random_search.best_params_.items()}
        print("Best hyperparameters:", best_params)
        # mejor modelo
        X_train_resampled, X_test = scale_data_train_test(X_train_resampled, X_test, "standard")
        best_model = LogisticRegression(**best_params, random_state=42, max_iter=3000)
        best_model.fit(X_train_resampled, y_train_resampled, sample_weight=sample_weights)

        # predicciones en el conjunto de prueba
        y_pred = best_model.predict(X_test)

        # calculamos las métricas de evaluación y las añadimos a las listas
        self.__calculate_and_add_metrics(position, best_model, X_train_resampled, X_test, y_train_resampled, y_test, y_pred, best_params)

        # matriz de confusión y reporte de clasificación
        self.__confusion_matrix_and_report('Logistic Regression Oversampling', y_test, y_pred, encoder)
    

    def __logistic_regression_MI_train_and_evaluate(self, position):
        X, y, encoder = self.__preprocessing()
        X_train, X_test, y_train, y_test = divide_data_in_train_test(X, y)

        selector = SelectKBest(lambda X, y: mutual_info_classif(X, y, random_state=666), k=50)
        #selector = SelectFromModel(estimator=LogisticRegression(penalty='elasticnet', solver='saga', random_state=42, max_iter=5000, l1_ratio=0.4, C=0.25), threshold='1.6*mean')
        X_train_reduced = selector.fit_transform(X_train, y_train)
        X_test_reduced = selector.transform(X_test)

        # definimos un pipeline para el modelo LogisticRegression con StandardScaler
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(random_state=42, max_iter=1000))
        ])

        # definimos el espacio de búsqueda aleatoria de hiperparámetros
        param_dist = [
            {'classifier__penalty': ['l1'], 'classifier__solver': ['saga'], 'classifier__C': stats.uniform(0.29, 0.7)},
            {'classifier__penalty': ['l2'], 'classifier__solver': ['lbfgs', 'saga', 'newton-cg'], 'classifier__C': stats.uniform(0.25, 0.75)},
            {'classifier__penalty': ['elasticnet'], 'classifier__solver': ['saga'], 'classifier__C': stats.uniform(0.25, 0.75), 'classifier__l1_ratio': stats.uniform(0.2, 0.4)},
            {'classifier__penalty': [None], 'classifier__solver': ['lbfgs', 'saga', 'newton-cg']}
        ]

        # realizamos la búsqueda aleatoria de hiperparámetros
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        random_search = RandomizedSearchCV(pipeline, param_distributions=param_dist, n_iter=200, cv=skf, scoring='f1_macro', verbose=1, n_jobs=-1, random_state=42)
        random_search.fit(X_train_reduced, y_train)

        # mejores hiperparámetros
        best_params_reduced = {k.replace('classifier__', ''): v for k, v in random_search.best_params_.items()}
        print("Best hyperparameters:", best_params_reduced)
        # mejor modelo
        X_train_reduced, X_test_reduced = scale_data_train_test(X_train_reduced, X_test_reduced, "standard")
        best_model_reduced = LogisticRegression(**best_params_reduced, random_state=42, max_iter=1000)
        best_model_reduced.fit(X_train_reduced, y_train)

        # predicciones en el conjunto de prueba reducido
        y_pred_reduced = best_model_reduced.predict(X_test_reduced)

        # calculamos las métricas de evaluación y las añadimos a las listas
        self.__calculate_and_add_metrics(position, best_model_reduced, X_train_reduced, X_test_reduced, y_train, y_test, y_pred_reduced, best_params_reduced)

        # matriz de confusión y reporte de clasificación
        self.__confusion_matrix_and_report('Logistic Regression MI', y_test, y_pred_reduced, encoder)


    def __logistic_regression_PCA_train_and_evaluate(self, position):
        X, y, encoder = self.__preprocessing()
        X_train, X_test, y_train, y_test = divide_data_in_train_test(X, y)

        # escalamos los datos antes de aplicar PCA
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # aplicamos PCA para reducción de dimensionalidad
        pca = PCA(n_components=0.9, random_state=42)
        X_train_pca = pca.fit_transform(X_train_scaled)
        X_test_pca = pca.transform(X_test_scaled)

        # definimos un pipeline para el modelo LogisticRegression
        pipeline = Pipeline([
            ('classifier', LogisticRegression(random_state=42, max_iter=1000))
        ])

        # definimos el espacio de búsqueda de hiperparámetros
        param_distributions = [
            {'classifier__penalty': ['l1'], 'classifier__solver': ['saga'], 'classifier__C': stats.uniform(0.2, 1.3)},
            {'classifier__penalty': ['l2'], 'classifier__solver': ['lbfgs', 'saga', 'newton-cg'], 'classifier__C': stats.uniform(0.2, 1.3)},
            {'classifier__penalty': ['elasticnet'], 'classifier__solver': ['saga'], 'classifier__C': stats.uniform(0.2, 1.3), 'classifier__l1_ratio': stats.uniform(0.2, 0.4)},
            {'classifier__penalty': [None], 'classifier__solver': ['lbfgs', 'saga', 'newton-cg']}
        ]

        # realizamos la búsqueda de hiperparámetros
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        random_search = RandomizedSearchCV(pipeline, param_distributions, n_iter=100, cv=skf, scoring='f1_macro', verbose=1, n_jobs=-1, random_state=42)
        random_search.fit(X_train_pca, y_train)

        # mejores hiperparámetros
        best_params_pca = {k.replace('classifier__', ''): v for k, v in random_search.best_params_.items()}
        print("Best hyperparameters:", best_params_pca)
        # mejor modelo
        best_model_reduced = LogisticRegression(**best_params_pca, random_state=42, max_iter=1000)
        best_model_reduced.fit(X_train_pca, y_train)

        # predicciones en el conjunto de prueba
        y_pred_pca = best_model_reduced.predict(X_test_pca)

        # calculamos las métricas de evaluación y las añadimos a las listas
        self.__calculate_and_add_metrics(position, best_model_reduced, X_train_pca, X_test_pca, y_train, y_test, y_pred_pca, best_params_pca)

        # matriz de confusión y reporte de clasificación
        self.__confusion_matrix_and_report('Logistic Regression PCA', y_test, y_pred_pca, encoder)
    

    def __knn_train_and_evaluate(self, position):
        X, y, encoder = self.__preprocessing()
        X_train, X_test, y_train, y_test = divide_data_in_train_test(X, y)

        # definimos un pipeline para el modelo KNeighborsClassifier con MinMaxScaler
        pipeline = Pipeline([
            ('scaler', MinMaxScaler()),
            ('classifier', KNeighborsClassifier())
        ])

        # definimos el espacio de búsqueda de hiperparámetros
        param_distributions = {
            'classifier__n_neighbors': stats.randint(5, 26),
            'classifier__weights': ['distance'],
            'classifier__metric': ['euclidean', 'manhattan', 'minkowski', 'chebyshev']
        }

        # realizamos la búsqueda de hiperparámetros
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        random_search = RandomizedSearchCV(pipeline, param_distributions, n_iter=30, cv=skf, scoring='f1_macro', verbose=1, n_jobs=-1, random_state=42)
        random_search.fit(X_train, y_train)

        # mejores hiperparámetros
        best_params = {k.replace('classifier__', ''): v for k, v in random_search.best_params_.items()}
        print("Best hyperparameters:", best_params)
        # mejor modelo
        X_train, X_test = scale_data_train_test(X_train, X_test, "minmax")
        best_model = KNeighborsClassifier(**best_params)
        best_model.fit(X_train, y_train)

        # predicciones en el conjunto de prueba
        y_pred = best_model.predict(X_test)

        # calculamos las métricas de evaluación y las añadimos a las listas
        self.__calculate_and_add_metrics(position, best_model, X_train, X_test, y_train, y_test, y_pred, best_params)

        # matriz de confusión y reporte de clasificación
        self.__confusion_matrix_and_report('KNN', y_test, y_pred, encoder)
    

    def __knn_MI_train_and_evaluate(self, position):
        X, y, encoder = self.__preprocessing()      
        X_train, X_test, y_train, y_test = divide_data_in_train_test(X, y) 

        # calculamos la información mutua
        mi_classification = mutual_info_classif(X_train, y_train, random_state=42)
        important_features = X_train.columns[mi_classification > 0.04]
        X_train_reduced = X_train[important_features]
        X_test_reduced = X_test[important_features]
        
        # definimos un pipeline para el modelo KNeighborsClassifier con MinMaxScaler
        pipeline = Pipeline([
            ('scaler', MinMaxScaler()),
            ('classifier', KNeighborsClassifier())
        ])

        # definimos el espacio de búsqueda aleatoria de hiperparámetros
        param_distributions = {
            'classifier__n_neighbors': stats.randint(5, 24),
            'classifier__weights': ['distance'],
            'classifier__metric': ['euclidean', 'manhattan', 'chebyshev']
        }

        # realizamos la búsqueda de hiperparámetros
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        random_search = RandomizedSearchCV(pipeline, param_distributions, n_iter=30, cv=skf, scoring='f1_macro', verbose=1, n_jobs=-1, random_state=42)
        random_search.fit(X_train_reduced, y_train)

        # mejores hiperparámetros
        best_params_reduced = {k.replace('classifier__', ''): v for k, v in random_search.best_params_.items()}
        print("Best hyperparameters:", best_params_reduced)
        # mejor modelo
        X_train_reduced, X_test_reduced = scale_data_train_test(X_train_reduced, X_test_reduced, "minmax")
        best_model_reduced = KNeighborsClassifier(**best_params_reduced)
        best_model_reduced.fit(X_train_reduced, y_train)

        # predicciones en el conjunto de prueba reducido
        y_pred_reduced = best_model_reduced.predict(X_test_reduced)

        # calculamos las métricas de evaluación y las añadimos a las listas
        self.__calculate_and_add_metrics(position, best_model_reduced, X_train_reduced, X_test_reduced, y_train, y_test, y_pred_reduced, best_params_reduced)

        # matriz de confusión y reporte de clasificación
        self.__confusion_matrix_and_report('KNN MI', y_test, y_pred_reduced, encoder)


    def __knn_PCA_train_and_evaluate(self, position):
        X, y, encoder = self.__preprocessing()      
        X_train, X_test, y_train, y_test = divide_data_in_train_test(X, y) 

        # escalado de datos antes del PCA
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # aplicación de PCA para reducir dimensionalidad
        pca = PCA(n_components=45, random_state=42)  
        X_train_pca = pca.fit_transform(X_train_scaled)
        X_test_pca = pca.transform(X_test_scaled)
        
        # definimos un pipeline para el modelo KNeighborsClassifier
        pipeline = Pipeline([
            ('classifier', KNeighborsClassifier())
        ])

        # definimos el espacio de búsqueda aleatoria de hiperparámetros
        param_distributions = {
            'classifier__n_neighbors': stats.randint(6, 29),
            'classifier__weights': ['distance'],
            'classifier__metric': ['euclidean', 'manhattan', 'chebyshev'],
        }

        # realizamos la búsqueda de hiperparámetros con RandomizedSearchCV
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        random_search = RandomizedSearchCV(pipeline, param_distributions, n_iter=30, cv=skf, scoring='f1_macro', verbose=1, n_jobs=-1, random_state=42)
        random_search.fit(X_train_pca, y_train)

        # mejores hiperparámetros
        best_params_reduced = {k.replace('classifier__', ''): v for k, v in random_search.best_params_.items()}
        print("Best hyperparameters:", best_params_reduced)
        # mejor modelo reducido
        best_model_reduced = KNeighborsClassifier(**best_params_reduced)
        best_model_reduced.fit(X_train_pca, y_train)

        # predicciones en el conjunto de prueba reducido
        y_pred_reduced = best_model_reduced.predict(X_test_pca)

        # calculamos las métricas de evaluación y las añadimos a las listas
        self.__calculate_and_add_metrics(position, best_model_reduced, X_train_pca, X_test_pca, y_train, y_test, y_pred_reduced, best_params_reduced)

        # matriz de confusión y reporte de clasificación
        self.__confusion_matrix_and_report('KNN PCA', y_test, y_pred_reduced, encoder)
    

    def __preprocessing(self):
        matches_df = self.matches_df.copy()
        X = matches_df.drop(columns=["winner_team"])
        y = matches_df["winner_team"]
        y, encoder = code_categorical_data_multiclass(y)
        return X, y, encoder  
    

    def __confusion_matrix_and_report(self, model_name, y_test, y_pred, encoder):
        # matriz de confusión para modelo reducido
        conf_matrix = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=encoder.classes_, yticklabels=encoder.classes_)
        plt.xlabel('Predicted label')
        plt.ylabel('True label')
        plt.title(f'Confusion Matrix - {model_name}')
        plt.show()   

        # reporte de clasificación
        print(classification_report(y_test, y_pred))
    

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
        models = ['Random Forest', 'Random Forest Oversampling', 'Random Forest Reduced', 'Random Forest MI', 'Decision Tree', 'Decision Tree Oversampling', 
                  'Decision Tree MI', 'Logistic Regression', 'Logistic Regression Oversampling', 'Logistic Regression MI', 'Logistic Regression PCA',
                  'KNN', 'KNN MI', 'KNN PCA']
        results_df = pd.DataFrame(metrics, index=models)
        return results_df
    
