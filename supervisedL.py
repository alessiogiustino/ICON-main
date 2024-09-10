import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedKFold, learning_curve, train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier




#Funzione che mostra la curva di apprendimento per ogni modello
def plot_learning_curves(model, X, y, differentialColumn, model_name):
    train_sizes, train_scores, test_scores = learning_curve(model, X, y, cv=3, scoring='accuracy')

    # Calcola gli errori su addestramento e test
    train_errors = 1 - train_scores
    test_errors = 1 - test_scores

    # Calcola la deviazione standard e la varianza degli errori su addestramento e test
    train_errors_std = np.std(train_errors, axis=1)
    test_errors_std = np.std(test_errors, axis=1)
    train_errors_var = np.var(train_errors, axis=1)
    test_errors_var = np.var(test_errors, axis=1)

    # Stampa i valori numerici della deviazione standard e della varianza
    print(
        f"\033[95m{model_name} - Train Error Std: {train_errors_std[-1]}, Test Error Std: {test_errors_std[-1]}, Train Error Var: {train_errors_var[-1]}, Test Error Var: {test_errors_var[-1]}\033[0m")

    # Calcola gli errori medi su addestramento e test
    mean_train_errors = 1 - np.mean(train_scores, axis=1)
    mean_test_errors = 1 - np.mean(test_scores, axis=1)

    #Visualizza la curva di apprendimento
    plt.figure(figsize=(16, 10))
    plt.plot(train_sizes, mean_train_errors, label='Errore di training', color='green')
    plt.plot(train_sizes, mean_test_errors, label='Errore di testing', color='red')
    plt.title(f'Curva di apprendimento per {model_name}')
    plt.xlabel('Dimensione del training set')
    plt.ylabel('Errore')
    plt.legend()
    plt.show()




def returnBestModel(dataset, differentialColumn):
    X = dataset.drop(differentialColumn, axis=1).to_numpy()
    y = dataset[differentialColumn].to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    dtc = DecisionTreeClassifier()
    rfc = RandomForestClassifier()
    reg = LogisticRegression()
    DecisionTreeHyperparameters = {
        'criterion': ['gini', 'entropy','log_loss'],
        'max_depth': [None, 5, 10],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 5, 10],
        'splitter': ['best']}
    RandomForestHyperparameters = {
        'criterion': ['gini', 'entropy','log_loss'],
        'n_estimators': [2, 4,6],
        'max_depth': [5, 10],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 5]}
    LogisticRegressionHyperparameters = {
        'C': [0.01, 0.1, 1],
        'solver': ['liblinear', 'lbfgs'],
        'max_iter': [100,200]}
    gridSearchCV_dtc = GridSearchCV(dtc, DecisionTreeHyperparameters, cv=3, n_jobs=-1, verbose=1)
    gridSearchCV_rfc = GridSearchCV(rfc, RandomForestHyperparameters, cv=3, n_jobs=-1, verbose=1)
    gridSearchCV_reg = GridSearchCV(reg, LogisticRegressionHyperparameters, cv=3, n_jobs=-1, verbose=1)

    gridSearchCV_dtc.fit(X_train, y_train)


    gridSearchCV_rfc.fit(X_train, y_train)

    gridSearchCV_reg.fit(X_train, y_train)

    print("\033[94mDecisionTree best parameters: \033[0m", gridSearchCV_dtc.best_params_)
    print("\033[94mRandomForest best parameters: \033[0m", gridSearchCV_rfc.best_params_)
    print("\033[94mLogisticRegression best parameters: \033[0m", gridSearchCV_reg.best_params_)
    return gridSearchCV_dtc.best_estimator_, gridSearchCV_rfc.best_estimator_, gridSearchCV_reg.best_estimator_



#Funzione che esegue il training del modello mediante cross validation
def trainModelKFold(dataSet, differentialColumn):
    model={
        'DecisionTree':{
            'accuracy_list':[],
            'precision_list':[],
            'recall_list':[],
            'f1':[]
        },
        'RandomForest':{
            'accuracy_list':[],
            'precision_list':[],
            'recall_list':[],
            'f1':[]
        },
        'LogisticRegression':{
            'accuracy_list':[],
            'precision_list':[],
            'recall_list':[],
            'f1':[]
        }
    }
    dtc,rfc,reg = returnBestModel(dataSet, differentialColumn)
    X = dataSet.drop(differentialColumn, axis=1).to_numpy()
    y = dataSet[differentialColumn].to_numpy()
    cv = RepeatedKFold(n_splits=3, n_repeats=3)
    scoring_metrics = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
    results_dtc = {}
    results_rfc = {}
    results_reg = {}
    for metric in scoring_metrics:
        scores_dtc = cross_val_score(dtc, X, y, scoring=metric, cv=cv)
        scores_rfc = cross_val_score(rfc, X, y, scoring=metric, cv=cv)
        scores_reg = cross_val_score(reg, X, y, scoring=metric, cv=cv)
        results_dtc[metric] = scores_dtc
        results_rfc[metric] = scores_rfc
        results_reg[metric] = scores_reg
    model['LogisticRegression']['accuracy_list'] = (results_reg['accuracy'])
    model['LogisticRegression']['precision_list'] = (results_reg['precision_macro'])
    model['LogisticRegression']['recall_list'] = (results_reg['recall_macro'])
    model['LogisticRegression']['f1'] = (results_reg['f1_macro'])
    model['DecisionTree']['accuracy_list'] = (results_dtc['accuracy'])
    model['DecisionTree']['precision_list'] = (results_dtc['precision_macro'])
    model['DecisionTree']['recall_list'] = (results_dtc['recall_macro'])
    model['DecisionTree']['f1'] = (results_dtc['f1_macro'])
    model['RandomForest']['accuracy_list'] = (results_rfc['accuracy'])
    model['RandomForest']['precision_list'] = (results_rfc['precision_macro'])
    model['RandomForest']['recall_list'] = (results_rfc['recall_macro'])
    model['RandomForest']['f1'] = (results_rfc['f1_macro'])
    plot_learning_curves(dtc, X, y, differentialColumn, 'DecisionTree')
    plot_learning_curves(rfc, X, y, differentialColumn, 'RandomForest')
    plot_learning_curves(reg, X, y, differentialColumn, 'LogisticRegression')
    visualizeMetricsGraphs(model)
    return model

#Funzione che visualizza i grafici delle metriche per ogni modello
def visualizeMetricsGraphs(model):
    models = list(model.keys())

    # Creazione di un array numpy per ogni metrica
    accuracy = np.array([model[clf]['accuracy_list'] for clf in models])
    precision = np.array([model[clf]['precision_list'] for clf in models])
    recall = np.array([model[clf]['recall_list'] for clf in models])
    f1 = np.array([model[clf]['f1'] for clf in models])

    # Calcolo delle medie per ogni modello e metrica
    mean_accuracy = np.mean(accuracy, axis=1)
    mean_precision = np.mean(precision, axis=1)
    mean_recall = np.mean(recall, axis=1)
    mean_f1 = np.mean(f1, axis=1)

    # Creazione del grafico a barre
    bar_width = 0.2
    index = np.arange(len(models))
    plt.bar(index, mean_accuracy, bar_width, label='Accuracy')
    plt.bar(index + bar_width, mean_precision, bar_width, label='Precision')
    plt.bar(index + 2 * bar_width, mean_recall, bar_width, label='Recall')
    plt.bar(index + 3 * bar_width, mean_f1, bar_width, label='F1')
    # Aggiunta di etichette e legenda
    plt.xlabel('Modelli')
    plt.ylabel('Punteggi medi')
    plt.title('Punteggio medio per ogni modello')
    plt.xticks(index + 1.5 * bar_width, models)
    plt.legend()
    # Visualizzazione del grafico
    plt.show()