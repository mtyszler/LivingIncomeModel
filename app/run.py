import json
import plotly
import numpy as np
import pandas as pd
import random

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
import joblib
from sqlalchemy import create_engine

import sys

sys.path.append('../')
sys.path.append('.')

app = Flask(__name__)


# load model
try:
    model = joblib.load("../models/LI_simplified_model.pkl")
except:
    model = joblib.load("models/LI_simplified_model.pkl")


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
    # read user input
    input1 = request.args.get('input1', type=float)/100 # converts percentage to number [0,1]
    input2 = request.args.get('input2', type=float)
    input3 = request.args.get('input3', type=float)
    input4 = request.args.get('input4', type=float)
    input5 = request.args.get('input5', type=float)
    input6 = request.args.get('input6', type=float)
    input7 = request.args.get('input7', type=float)/100 # converts percentage to number [0,1]
    input8 = request.args.get('input8', type=float)
    input9 = request.args.get('input9', type=float)/100 # converts percentage to number [0,1]
    input10 = request.args.get('input10', type=float)

    # collect in an array
    X = np.array([
        input1, 
        input2, 
        input3, 
        input4,
        input5, 
        input6, 
        input7, 
        input8, 
        input9, 
        input10])

    # ensure shape is correct
    X  = X.reshape(1, -1)
    
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
