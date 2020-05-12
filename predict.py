from __future__ import absolute_import, division, print_function
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from pickle import load
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # disable all warnings
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # only use CPU and not GPU
import tensorflow as tf

"""
Uses a trained neuronal network to predict components of a compliance matrix.
If you want the stiffness matrix, you have to train the model with different targets:
C11, C22, C33, C12, C23, C13, C44, C55, C66
Then you have to assemble these components as full compliance matrix and calculate the inverse...

For the sake of demonstration, the workflow for C11 is demonstrated in following modules:
regressor.py (if run, creates a subfolder in folder results, creates and trains a model inside this folder)
predict.py (can load a model if a result folder is specified and uses this model to make predictions)
"""

df = pd.read_csv(os.path.join('data', 'test_values.csv'), sep=';')
features = ['a11', 'a22', 'EM', 'VF', 'nuM']
target = 'C11'
test_features = df[features].values.tolist()

cwd = os.getcwd()
try:
    # path to the result folder
    path_regressor = os.path.join('results', 'C11', 'model')
    # add to system path to allow for import of reg.py in path_regressor
    sys.path.insert(0, path_regressor)
    # import saved network configuration from "create_model" definition in the subfolder
    import reg as r
    # change into the result folder and load the pickle objects
    os.chdir(path_regressor)
    # create model structure according to the subfolder file
    model = r.create_model()
    # get the latest checkpoint from the result folder
    latest = tf.train.latest_checkpoint('checkpoint')
    # load the previously saved weights
    model.load_weights(latest)
    # load the scaler from the model
    scaler = load(open('scaler.pkl', 'rb'))
    scaler_y = load(open('scaler_y.pkl', 'rb'))
    # transform test values according to the scaler on which the training data of the model was fitted
    x = scaler.transform(test_features)
    # transform into list of numpy arrays as input
    x = [v for v in np.array(x).T]
    # predict the stiffness values with the trained model
    y = model.predict(x)
    y = np.concatenate(y, axis=0)
finally:
    os.chdir(cwd)

# prepare the diagram data
y_predicted = scaler_y.inverse_transform(y.T)
y_real = df[target].values
x = df['a11'].values

# plot the results
plt.plot(x, y_real, 'rx', label='Digimat C11')
plt.plot(x, y_predicted, 'bx', label='Neuronal Network C11')
plt.xlabel('a11 / -')
plt.ylabel('Stiffness / MPa')
plt.legend()
plt.show()
