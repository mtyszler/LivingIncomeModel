import json
import plotly
import numpy as np
import pandas as pd
import random

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
import joblib

from sklearn.model_selection import train_test_split

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
    data = joblib.load("../data/data_for_restricted_ML.pkl")
except:
    data = joblib.load("data/data_for_restricted_ML.pkl")

# split data into the target and features 
target = data['Living Income Achieved']
data.drop(['Living Income Achieved'], axis = 1, inplace = True)

# split the data into training and testing sets used for model tuning:
X_train, X_test, y_train, y_test = train_test_split(data, target,
                                                    train_size=0.75, test_size=0.25,
                                                   random_state = 32)

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    # render base page
    return render_template('classifier.html')

def LI_histogram(colname, features, target):
    '''
    Create a histogram overlaying those who achieved vs 
    those who did not achieved a Living Income
    
    input: column name, existing in data
    output: dictionary to be appended to graph object

    '''
    LI = {
            'data': [
                {
                    'type':'histogram',
                    'histnorm': 'percent',
                    'x':features[target==False][colname],
                    'name': 'Did not achieve',
                    'opacity': '0.5'
                },
                {
                    'type':'histogram',
                    'histnorm': 'percent',
                    'x':features[target==True][colname],
                    'name': 'Achieved',
                    'opacity': '0.5'
                }

            ],

            'layout': {
                'title': 'Distribution of [' + colname +']',
                'yaxis': {
                    'title': "Proportion (%)",
                    #'range': [1, 100],
                    'hoverformat': '.2f'
                },
                'barmode': "overlay"#,
                #'xaxis': {
                #    'title': "Category"
                #}
            }
        }

    return LI

@app.route('/training')
def training():
    # get shape info:
    n_obs = len(X_train.index) 
    n_features = len(X_train.columns)

    
    # create visuals
    graphs = [
        # graph 1: bar chart of number of household achieving the LI
        {
            'data': [
                Bar(
                    y=y_train.value_counts()/n_obs*100,
                    x=['Did not achieve', 'Achieved']
                )
            ],

            'layout': {
                'title': 'How many achieved a Living Income?',
                'yaxis': {
                    'title': "Proportion (%)",
                    'range': [1, 100],
                    'hoverformat': '.2f'
                }
            }
        }
    ]
    ids = ['Living Income Achieved']

    # create histograms for each column
    for col in X_train.columns:
        graphs.append(LI_histogram(col, X_train, y_train))
        ids.append(col)        

    # encode plotly graphs in JSON
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('training.html', ids=ids, graphJSON=graphJSON, n_obs = n_obs, n_features = n_features)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # read user input (see classifier.html)
    input1 = request.args.get('input1', type=float)/100 # converts percentage to number [0,1]
    input3 = request.args.get('input3', type=float)
    input4 = request.args.get('input4', type=float)
    input2 = input3/input4 
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
    classification_label = model.predict(X)[0]
    classification_prob = np.int(np.round(model.predict_proba(X)[0][1]*100,0))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        classification_label=classification_label,
        classification_prob=classification_prob
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
