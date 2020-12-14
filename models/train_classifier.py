"""
This script is an primarily a ML pipeline for the processing training of a model to classify 
msgs in multi-categories, and saves it in a pickle file

To run ML pipeline that loads data prepared by the ETL (see process_data.py)
python [path]train_classifier.py DB PKL,
for example:

python models/train_classifier.py data/DisasterResponse.db models/best_model.pkl

It assumes the following:
* DB has been prepared by the ETL
* table in the database will be called 'classified_msgs'
* 'classified_msgs' has messages (in a field called 'messages') and classifications (starting at iloc 4)


"""

import sys

import pandas as pd
from sqlalchemy import create_engine

import pickle

from sklearn.pipeline import Pipeline

from sklearn.metrics import make_scorer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import AdaBoostClassifier
from sklearn.multioutput import MultiOutputClassifier

from sklearn.model_selection import GridSearchCV

'''
1. loads custom (performance metric) functions:
average_accuracy
average_precision
average_recall
average_f1
naive_predictions
print_classification_report

2. loads custom tokenizer:
tokenize

The split of these functions into a separate file has two goals:
a. modulerize the code
b. allow the app to run properly without the need to repeat the hard coding of the 
   tokenize function there. Moreover, I noticed that when the tokenize function was
   in the same script as the train_classifier, after dumping the best model to a 
   pickle file, the load in the (live web)app did not work because the load was expecting a 
   tokenize function in the __main__ environment. 

   By importing the tokenize function in the exact same way in both thr train_classiier.py 
   and run.py I avoid any conflicts and the (live web)app runs as expected.
'''
sys.path.append('.')
from models.supporting_functions import *


def load_data(database_filepath):
    """
    loads messages and categories from database

    :param database_filepath: full path of a db file with messages and categories

    :return: 
    pandas DataFrames:
    X: messages
    Y: targets

    """

    # print('sqlite:///{}'.format(database_filepath))
    engine = create_engine('sqlite:///{}'.format(database_filepath))

    df = pd.read_sql('SELECT * FROM classified_msgs', engine)

    X = df['message']
    Y = df.iloc[:, 4:]

    print("X is:", X.shape)
    print("Y is:", Y.shape)

    return X, Y


def build_model():
    """
    Creates a ML pipeline with GridSearchCV

    input: nothing

    returns:
    GridSearchCV object


    """

    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(AdaBoostClassifier(random_state=42)))
    ], verbose=True)

    # parameters to tune
    parameters = {
        'tfidf__use_idf': [True, False],
        'clf__estimator__n_estimators': [10, 50, 100, 150, 200, 500],
        'clf__estimator__learning_rate': [1.0, 1.5, 2.0]
    }

    # simplified grid for testing purposes.
    # comment out for real run
    # uncomment for simple tests

    '''
    parameters = {
        'clf__estimator__learning_rate': [1.0, 1.5]
    }
    '''

    # scorer using the average_f1
    scorer = make_scorer(average_f1)

    # Grid Search object
    cv = GridSearchCV(pipeline, param_grid=parameters, scoring=scorer, verbose=2, n_jobs=-1)

    return cv


def evaluate_model(model, X_test, Y_test):
    """
    shows performance metrics of the (best) model, compared to a naive predictor

    inputs:
    model: already fitted
    X_test: test features
    Y_test: test targets

    returns:
    none

    """

    # create naive predictions
    naive_pred = naive_predictions(Y_test)

    # create model predictions
    Y_pred = model.predict(X_test)

    print("Naive accuracy: ", average_accuracy(Y_test, naive_pred))
    print("Optimized model accuracy: ", average_accuracy(Y_test, Y_pred))

    print("Naive precision: ", average_precision(Y_test, naive_pred.to_numpy()))
    print("Optimized model precision: ", average_precision(Y_test, Y_pred))

    print("Naive recall: ", average_recall(Y_test, naive_pred.to_numpy()))
    print("Optimized model recall: ", average_recall(Y_test, Y_pred))

    print("Naive f1-score: ", average_f1(Y_test, naive_pred.to_numpy()))
    print("Optimized model f1-score: ", average_f1(Y_test, Y_pred))

    print("Detailed report:")
    print_classification_report(Y_test, Y_pred)


def save_model(model, model_filepath):
    """
    Saves model to a pickle file

    inputs:
    model: already fitted
    model_filepath: full path for the pickle file

    returns:
    none
    """

    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('')
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('')
        print('Building model...')
        model = build_model()

        print('')
        print('Training model...')
        print('Hold on tight: this might take a while...')
        print('')
        model.fit(X_train, Y_train)

        print('')
        print("Best model:")
        print(model.best_estimator_)

        print('')
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test)

        print('')
        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model.best_estimator_, model_filepath)

        print('')
        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '
              'as the first argument and the filepath of the pickle file to '
              'save the model to as the second argument. \n\nExample: python '
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
