from Model import model_evaluation, rf_selection, params_tunning, neural_network, randomforest, svm
from Preprocessing import training_data_color, test_data
from Test import test, predict
import numpy as np
from matplotlib import pyplot as plt
from joblib import load
from plot_learning_curve import plot_learning_curve

# color classification
# load data
X, y, encoder = training_data_color(0)

# If the model is updated, this method needs to be run again, otherwise it does not
# rf_selection(X,y,'color')

index = np.load('feature_index_color.npy').tolist()
X = X[:, index]
# Hyperparameter tuning and model evaluation,
# changing parameters to run different models with hyperparameter tuning
# params_tunning(X, y, 'rf')
# model_evaluation(X, y)
# Drawing learning curves,Change the input model and title as well as the file to try a different model
# plt = plot_learning_curve(neural_network(),'nn with 5 L2 penalty',X,y)
# plt.savefig('nn_alpha5.png')
# plt.show()
# load test set
# X_test = test_data(0)
# index = np.load('feature_index_color.npy').tolist()
# X_test = X_test[:,index]

# Validate the different models and save them, please run them once before predicting
test(X, y, randomforest())
test(X, y, neural_network())
test(X, y, svm())
# Classifying test sets,change to file name to load another model
# model = load('svm.pkl')
# predict(X_test, model, encoder, 'color')


# texture classification
# load data
# X, y, encoder = training_data_color(0)

# If the model is updated, this method needs to be run again, otherwise it does not
# rf_selection(X,y,'texture')

# index = np.load('feature_index_texture.npy').tolist()
# X = X[:,index]
# Hyperparameter tuning and model evaluation,
# changing parameters to run different models with hyperparameter tuning
# params_tunning(X, y, 'rf')
# model_evaluation(X, y)
# Drawing learning curves,Change the input model and title as well as the file to try a different model
# plt = plot_learning_curve(neural_network(),'nn with 5 L2 penalty',X,y)
# plt.savefig('nn_alpha5.png')
# plt.show()
# load test set
# X_test = test_data(0)
# index = np.load('feature_index_texture.npy').tolist()
# X_test = X_test[:,index]

# Validate the different models and save them, please run them once before predicting
# test(X, y, randomforest())
# test(X, y, neural_network())
# test(X, y, svm())
# Classifying test sets

# predict(X_test, model, encoder, 'texture')
