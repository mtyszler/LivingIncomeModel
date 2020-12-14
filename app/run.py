import json
import plotly
import numpy as np
import pandas as pd
import random

from flask import Flask
from flask import render_template, request, jsonify
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

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

def LI_histogram(colname, features, target, set_name):
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
                'title': set_name +': Distribution of [' + colname +']',
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

@app.route('/dataset')
def dataset():
    # get shape info:
    n_obs_train = len(X_train.index) 
    n_features = len(X_train.columns)
    n_obs_test = len(X_test.index) 

    train_share = np.int(np.round(n_obs_train/(n_obs_train+n_obs_test)*100,0))
    test_share = np.int(100-train_share)

    # create visuals
    graphs = [
        # graph 1: bar chart of number of household achieving the LI, training
        {
            'data': [
                {
                    'type':'bar',
                    'y':y_train.value_counts()/n_obs_train*100,
                    'x':['Did not achieve', 'Achieved'],
                    'name':"Training set"
                },
                {
                    'type':'bar',
                    'y':y_test.value_counts()/n_obs_test*100,
                    'x':['Did not achieve', 'Achieved'],
                    'name':"Testing set"
                }
            ],

            'layout': {
                'title': 'How many achieved a Living Income?',
                'yaxis': {
                    'title': "Proportion (%)",
                    'range': [1, 100],
                    'hoverformat': '.2f'
                },
                'barmode':'group'
            }
        }
    ]
    ids = ['Living Income Achieved']

    # create histograms for each column, each set
    graphs_training = []
    ids_training = []

    graphs_testing = []
    ids_testing = []

    for col in X_train.columns:
        graphs_training.append(LI_histogram(col, X_train, y_train, "Training set"))
        ids_training.append(col)

        graphs_testing.append(LI_histogram(col, X_test, y_test, "Testing set"))
        ids_testing.append(col)        

    # encode plotly graphs in JSON
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    graphJSON_training = json.dumps(graphs_training, cls=plotly.utils.PlotlyJSONEncoder)
    graphJSON_testing = json.dumps(graphs_testing, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('dataset.html', 
        ids=ids, graphJSON=graphJSON, 
        ids_training = ids_training, graphJSON_training = graphJSON_training,
        ids_testing = ids_testing, graphJSON_testing = graphJSON_testing,
        n_obs_train = n_obs_train, n_obs_test = n_obs_test, 
        train_share = train_share, test_share = test_share,
        n_features = n_features)


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
        classification_prob=classification_prob,
        input_type="user"
    )

@app.route('/go_mean')
def go_mean():
    # Calculate means
    X = np.array(data.mean())
    X  = X.reshape(1, -1)

    # round to simulate user input
    input1 = np.int(np.round(X[0,0]*100,0)) # converts percentage to number [0,100]
    input3 = np.int(np.round(X[0,2],0))
    input4 = np.int(np.round(X[0,3],0))
    input2 = input3/input4 
    input5 = np.int(np.round(X[0,4],0)) 
    input6 = np.int(np.round(X[0,5],0)) 
    input7 = np.int(np.round(X[0,6]*100,0)) # converts percentage to number [0,100]
    input8 = np.int(np.round(X[0,7],0)) 
    input9 = np.int(np.round(X[0,8]*100,0)) # converts percentage to number [0,100]
    input10 = np.int(np.round(X[0,9],0))

    # collect in an array
    X = np.array([
        input1/100, 
        input2, 
        input3, 
        input4,
        input5, 
        input6, 
        input7/100, 
        input8, 
        input9/100, 
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
        classification_prob=classification_prob,
        input_type = "server",
        server_1 =  input1,
        server_2 =  input2,
        server_3 =  input3,
        server_4 =  input4,
        server_5 =  input5,
        server_6 =  input6,
        server_7 =  input7,
        server_8 =  input8,
        server_9 =  input9,
        server_10 = input10
    )


@app.route('/performance')
def performance():

    # test set predictions:
    test_predictions = model.predict(X_test)
    
    # performance metrics:
    accuracy = np.int(np.round(accuracy_score(y_test, test_predictions)*100,0))
    prfs = precision_recall_fscore_support(y_test, test_predictions, beta = 0.5, average = 'binary')
    precision = np.int(np.round(prfs[0]*100,0))
    recall = np.int(np.round(prfs[1]*100,0))
    fscore = np.int(np.round(prfs[2]*100,0))

    cm = confusion_matrix(y_test, test_predictions)
    ff = cm[0][0] # false, predicted false
    ft = cm[0][1] # false, predicted true
    tf = cm[1][0] # true, predicted false
    tt = cm[1][1] # true, predicted true

    real_t = tf+tt
    pred_t = ft+tt
    # selects a random row:

    # render perfomance page
    return render_template(
        'performance.html',
        accuracy=accuracy,
        precision = precision,
        recall = recall,
        fscore = fscore,
        ff=ff, ft=ft, tf=tf,tt=tt,
        real_t=real_t, pred_t=pred_t
        )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
