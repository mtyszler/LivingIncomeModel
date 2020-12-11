import json
import plotly
import pandas as pd
import random

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
import joblib
from sqlalchemy import create_engine

import sys

'''
1. loads custom tokenizer:
tokenize

The tokenize function is split from the train_classifier.py for two reasons:
a. modulerize the code
b. allow the app to run properly without the need to repeat the hard coding of the 
   tokenize function there. Moreover, I noticed that when the tokenize function was
   in the same script as the train_classifier, after dumping the best model to a 
   pickle file, the load in the (live web)app did not work because the load was expecting a 
   tokenize function in the __main__ environment. 

   By importing the tokenize function in the exact same way in both thr train_classiier.py 
   and run.py I avoid any conflicts and the (live web)app runs as expected.
'''
sys.path.append('../')
sys.path.append('.')
from models.supporting_functions import tokenize

app = Flask(__name__)


# load model
try:
    model = joblib.load("../models/classifier.pkl")
except:
    model = joblib.load("models/classifier.pkl")


# load data
try:
    engine = create_engine('sqlite:///../data/DisasterResponse.db')
    df = pd.read_sql_table('classified_msgs', engine)
except:
    engine = create_engine('sqlite:///data/DisasterResponse.db')
    df = pd.read_sql_table('classified_msgs', engine)


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    # render base page
    return render_template('classifier.html')


@app.route('/training')
def training():
    # get shape info:
    X_len = len(df.index) 
    Y_len = len(df.iloc[:, 4:].columns)


    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    categs_sorted = df.iloc[:, 4:].mean().sort_values(ascending=False)
    categ_share = categs_sorted
    categ_names = list(categ.replace('_', ' ') for categ in categs_sorted.index)

    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        # graph 1
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },

        # graph 2
        {
            'data': [
                Bar(
                    x=categ_names,
                    y=categ_share * 100
                )
            ],

            'layout': {
                'title': 'Occurrences of Categories',
                'orientation': 'v',
                'yaxis': {
                    'title': "Proportion (%)",
                    'range': [1, 100],
                    'hoverformat': '.2f'
                },
                'xaxis': {
                    'title': "Category"
                }
            }
        }
    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('training.html', ids=ids, graphJSON=graphJSON, n_rows = X_len, n_cols = Y_len)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '')

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


@app.route('/performance')
def performance():
    
    # gets the number of messages & categs
    n_msgs = len(df.index) 
    n_categs = len(df.iloc[:, 4:].columns)

    # selects a random row:
    row_number = random.randrange(0,n_msgs)

    # picks the message and targets:
    msg = df['message'][row_number]
    targets = df.iloc[row_number, 4:]

    # predict:
    predictions = model.predict([msg])[0]

    # number of correct classifications:
    correct_classifications = (targets == predictions).sum()
    incorrect_classifications = n_categs - correct_classifications

    # combine it all
    classification_results = zip(df.columns[4:], targets.astype(bool), predictions.astype(bool))

    # render perfomance page
    return render_template(
        'performance.html',
        msg = msg,
        classification_result=classification_results,
        correct = correct_classifications,
        incorrect = incorrect_classifications
        )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
