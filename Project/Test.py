from Model import train
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pandas as pd
from sklearn.metrics import balanced_accuracy_score


def test(X, y, model):
    '''
    Although it is called a test, it is actually a validation
    Validate and persist models
    '''
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, train_size=0.9)
    train(X_train, y_train, model, 'svm.pkl')
    y_predict = model.predict(X_test)
    print(classification_report(y_test, y_predict))
    print(balanced_accuracy_score(y_test, y_predict))


def predict(X, model, encoder, category):
    '''
    Use the test set to generate classification results for uploading to leaderboard
    '''
    y_predict = model.predict(X)
    y_predict = encoder.inverse_transform(y_predict)
    df = pd.DataFrame(y_predict)
    df.to_csv(category + '_test.csv', index=False, header=False)
    print(y_predict)
