from Preprocessing import sampling_SmoteTomek, imputer, scaler
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.pipeline import Pipeline
from joblib import dump
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

'''
4 methods for generating models, using Pipeline to integrate Filling, normalisation, resampling and the classifier
'''


def neural_network():
    mlp = MLPClassifier(hidden_layer_sizes=[100, 100], alpha=0.01, max_iter=1000, verbose=False,
                        early_stopping=False,
                        learning_rate_init=1e-3, activation='relu', random_state=42)
    model = Pipeline([
        ('num_imputer', imputer()),
        ('std_scaler', scaler()),
        # ('sampling', sampling_SmoteTomek()),
        ('mlp', mlp)
    ])
    return model


# Even after many attempts, LinearSVC fails to converge
def svm():
    # using balanced class weight
    svc = SVC(C=1, random_state=42, max_iter=4000, gamma='auto', class_weight='balanced')
    model = Pipeline([

        ('num_imputer', imputer()),
        ('std_scaler', scaler()),
        # ('sampling', sampling_SmoteTomek()),
        ('svm', svc)
    ])
    return model


def randomforest():
    # using balanced class weight
    rf = BalancedRandomForestClassifier(n_estimators=200, verbose=False, n_jobs=4,
                                        random_state=42)
    model = Pipeline([

        ('num_imputer', imputer()),
        ('std_scaler', scaler()),
        # ('sampling', sampling_SmoteTomek()),
        ('rf', rf)
    ])
    return model


def logisticregression():
    # using balanced class weight
    lr = LogisticRegression(C=1, class_weight='balanced', random_state=0, max_iter=1000)
    model = Pipeline([

        ('num_imputer', imputer()),
        ('std_scaler', scaler()),
        # ('sampling', sampling_SmoteTomek()),
        ('lr', lr)
    ])
    return model


'''
The features that will be used to select the model have an importance between 0.001 and 0.006, 
so features below 0.003 are filtered out
'''


def rf_selection(X, y, type):
    rf = BalancedRandomForestClassifier(oob_score=True, n_jobs=4, n_estimators=400)
    model = Pipeline([
        ('num_imputer', imputer()),
        ('std_scaler', scaler()),
        # ('sampling', sampling_SmoteTomek()),
        ('rf', rf)
    ])
    model.fit(X, y)
    importance = rf.feature_importances_
    count = 0
    index = []
    # save the index to use it later
    for value in importance:
        if value > 0.003:
            index.append(count)
        count += 1
    np.save('feature_index_' + type + '.npy', index)
    return index


# fit the dataset and save the model including the preprocessing process.
def train(X, y, model, filename):
    model.fit(X, y)
    dump(model, filename)


'''
The performance of each model is tested using KFold cross-validation, repeated twice,
 i.e. the results are averaged over a total of 10 times, Scoring = Balanced Accuracy
 output a boxplot
'''


def evaluator(X, y, model):
    validation = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=0)
    scores = cross_val_score(model, X, y, scoring='balanced_accuracy', cv=validation)
    return scores


def model_evaluation(X, y):
    untrained_models = {
        'svm': svm(),
        'nn': neural_network(),
        'rf': randomforest(),
        'lr': logisticregression()
    }
    all_scores = list()
    for name, model in untrained_models.items():
        scores = evaluator(X, y, model)
        all_scores.append(scores)
        print('%s :%.3f' % (name, np.mean(scores)))
    plt.boxplot(all_scores, labels=untrained_models.keys(), showmeans=True)
    plt.savefig('boxplot.png')
    plt.show()


'''
Random forest with few variable parameters, only the number of trees
SVM can test the penalty factor C and gamma,
gamma represents the range of influence of a data point and 
can be simply understood as the volume of a point (personal understanding),
2 parameters together control the level of fitting of the SVM
nothing special for neural network

'''


def params_tunning(X, y, type):
    if type == 'rf':
        model = BalancedRandomForestClassifier(class_weight='balanced', verbose=False, n_jobs=4,
                                               oob_score=True)
        grid_param = {
            'n_estimators': [100, 200, 300, 400, 500]
        }
    elif type == 'svm':
        grid_param = {
            'C': [1, 5, 10, 20],
            'gamma': ['auto', 0.01, 0.1, 1]
        }
        model = SVC(random_state=0, max_iter=2000, class_weight='balanced')

    else:
        model = MLPClassifier(max_iter=500, random_state=42, hidden_layer_sizes=[100, 100], alpha=0.01)
        grid_param = {
            'hidden_layer_sizes': [[100, 100], [100, 100, 100]],
            'learning_rate_init': [1e-3, 1e-2, 5e-3],
            'alpha': [0.001, 0.01]
        }
    grid_search = GridSearchCV(model, param_grid=grid_param, n_jobs=4, scoring='balanced_accuracy',
                               cv=RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=0))
    model = Pipeline([
        ('num_imputer', imputer()),
        ('std_scaler', scaler()),
        # Once resampled the data will leak, but the model evaluation does not, it is not clear why
        # ('sampling', sampling_SmoteTomek()),
        ('grid_search', grid_search)
    ])
    model.fit(X, y)
    result = pd.DataFrame.from_dict(grid_search.cv_results_)
    # result = pd.DataFrame.from_dict(model.steps[3][1].cv_results_)
    result.to_csv(type + '.csv')
    print(result)
