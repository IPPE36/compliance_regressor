from __future__ import absolute_import, division, print_function
import pandas as pd
import numpy as np
import time
import shutil
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from pickle import dump, load
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # disable all warnings
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # only use CPU and not GPU

import tensorflow as tf
import tensorflow.keras.callbacks as cb

# seed the randomness to produce reproducible results
tf.random.set_seed(1)
np.random.seed(1)


# build the model
def create_model():
    """This function builds the neuronal network according to saved parameters in the working directory.
    It is coded outside of "if __name__ == "__main"" to allow other modules the import of this function.

    Uses the pickle module to load following parameters:
    Feature names f are necessary because they determine the input layer of the neuronal network.
    Units determine the number of hidden layers and their respective amount of neurons.

    To load a trained model, use this function to create a non-trained model structure, then load the weighrs.

    :return: untrained keras model
    """

    # utility method to generate numeric feature columns
    def gen_numeric_features(columns):
        numeric_feature_columns = []
        numeric_feature_layer_inputs = {}
        for c in columns:
            numeric_feature_columns.append(tf.feature_column.numeric_column(c))
            numeric_feature_layer_inputs[c] = tf.keras.Input(shape=(1,), name=c)
        return numeric_feature_columns, numeric_feature_layer_inputs

    # utility method to generate bucketized columns
    def gen_bucket_features(columns, n_buckets, *args):
        # utility method to generate convenient threshold values for bucketization of feature columns
        def get_quantile_based_boundaries(vals, n):
            boundaries = np.arange(1.0, n) / n
            quantiles = vals.quantile(boundaries)
            return [quantiles[q] for q in quantiles.keys()]

        bucket_feature_columns = []
        for c, n in zip(columns, n_buckets):
            try:
                feature_column = tf.feature_column.numeric_column(c)
                if 'quantile_based' in args:
                    boundary_list = get_quantile_based_boundaries(df[c], n)
                else:
                    boundary_list = n
                bucket_feature_columns.append(
                    tf.feature_column.bucketized_column(feature_column, boundaries=boundary_list))
            except ValueError:
                print(str(c) + "... reduce number of buckets!")
        return bucket_feature_columns

    # load the pickle features and targets
    f = load(open('features.pkl', 'rb'))
    t = load(open('targets.pkl', 'rb'))
    units = load(open('units.pkl', 'rb'))

    # prepare tensor flow feature columns
    feature_columns, feature_layer_inputs = gen_numeric_features(f)
    bfc = gen_bucket_features(['VF'], [[0.33, 0.66, 0.85]])
    feature_columns.extend(bfc)

    # build the neural network
    def neuronal_network():
        feature_layer = tf.keras.layers.DenseFeatures(feature_columns)
        x = feature_layer(feature_layer_inputs)
        for idx, nodes in enumerate(units):
            x = tf.keras.layers.Dense(units=nodes, activation='relu', use_bias=True)(x)
            x = tf.keras.layers.Dropout(0.5)(x)
        for t_ in t:
            # for each target create a seperate! layer dependent on the last hidden layer...
            # the amount of losses to optimize is equal to the number of these output layers...
            yield tf.keras.layers.Dense(1, activation='linear', name=t_)(x)

    model = tf.keras.Model(inputs=list(feature_layer_inputs.values()), outputs=list(neuronal_network()))
    model.compile(optimizer='adam', loss=['MSE']*len(t), loss_weights=[1.0]*len(t), metrics=["MAE"])
    return model


if __name__ == '__main__':
    # load the data into a pandas df
    df = pd.read_csv(os.path.join('data', 'results.csv'), sep=';')

    # visualize data, be careful when df is huge
    # scatter_matrix(df[['a11', 'a22', 'a33', 'EM', 'VF', 'nuM', 'C11']], alpha=0.2, figsize=(7, 7), diagonal='kde')

    # Clean dataFrame
    df.drop(df[df['nuM'] > 0.4].index, inplace=True)

    features = ['a11', 'a22', 'EM', 'VF', 'nuM']
    targets = ['C11', 'C22', 'C33', 'C12', 'C23', 'C13', 'C44', 'C55', 'C66']
    n_features = len(features)
    n_targets = len(targets)
    X = df[features].values.astype(float)
    y = df[targets].values.astype(float)
    # combine features and labels for consistent splitting of datasets
    X = np.concatenate((X, y), axis=1)

    # split the data into train-, validation- and testing data
    train, test = train_test_split(X, test_size=0.2)
    train, val = train_test_split(train, test_size=0.2)

    # fit a datascaler to train set and apply to validation and test set
    scaler = StandardScaler()
    # usually scaling of targets is not necessary, in this case the target values are very small, so it helps
    scaler_y = StandardScaler()

    # apply scaling to features that are not naturally in the range of 0 to 1 like fibre orientation
    train[:, :-n_targets] = scaler.fit_transform(train[:, :-n_targets])
    train[:, -n_targets:] = scaler_y.fit_transform(train[:, -n_targets:])

    # fit only on train data, transform the other datasets to the same scale!
    val[:, :-n_targets] = scaler.transform(val[:, :-n_targets])
    val[:, -n_targets:] = scaler_y.transform(val[:, -n_targets:])
    test[:, :-n_targets] = scaler.transform(test[:, :-n_targets])
    test[:, -n_targets:] = scaler_y.transform(test[:, -n_targets:])

    # create a subdirectory for the results
    cwd = os.getcwd()
    result_dir = os.path.join('results', str(int(time.time())))
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(os.path.join(result_dir, 'model'), exist_ok=True)

    # use the pickle module to save parameters in files
    try:
        os.chdir(os.path.join(result_dir, 'model'))
        # copy this script and dump parameters in pickle files
        shutil.copy(__file__, os.getcwd() + os.sep + 'reg.py')
        dump(features, open('features.pkl', 'wb'))
        dump(targets, open('targets.pkl', 'wb'))
        dump(scaler, open('scaler.pkl', 'wb'))
        dump(scaler_y, open('scaler_y.pkl', 'wb'))
        dump([1024, 512], open('units.pkl', 'wb'))
        # create a new model instance
        model = create_model()
    finally:
        os.chdir(cwd)

    # callbacks for logging and saving
    callbacks = [
        cb.CSVLogger(
            filename=os.path.join(result_dir, 'model', 'training.log'),
            append=True),
        cb.ModelCheckpoint(
            filepath=os.path.join(result_dir, 'model', 'checkpoint', 'cp.ckpt'),
            save_weights_only=True,
            save_best_only=True,
            monitor='val_loss',
        ),
        cb.ReduceLROnPlateau(monitor='val_loss',
                             factor=0.5,
                             patience=10,
                             verbose=1,
                             mode='auto',
                             min_delta=0.0001,
                             cooldown=0,
                             min_lr=0),
        cb.EarlyStopping(monitor='val_loss',
                         min_delta=0,
                         patience=30,
                         verbose=1,
                         mode='auto',
                         baseline=None,
                         restore_best_weights=False)
    ]

    # utility method to create a tf data data set from a pd data frame
    def df_to_dataset(data, shuffle=True, batch_size=32):
        """Prepares a tensorflow dataset from a numpy array
        :param data: numpy array, the target columns are the last columns in the array
        :param shuffle: determines shuffling
        :param batch_size: the used batch size
        :return:
        """
        # take last columns of data as targets as dictionary of numpy arrays
        l = {t_: data[:, -(i+1)] for i, t_ in enumerate(targets[::-1])}
        # features must be in form of a dictionary
        f = {f_: data[:, i] for i, f_ in enumerate(features)}
        # construct tf data set (requires 1 argument of type tuple!)
        ds = tf.data.Dataset.from_tensor_slices((f, l))
        # shuffle the data set
        if shuffle:
            ds = ds.shuffle(buffer_size=len(df))
        # batch the data set
        ds = ds.batch(batch_size)
        return ds


    # Minibatch training (1 < batch size < dataset size) should be multiple of 16, 32, 64, 128,...
    batch_size = 512
    train_ds = df_to_dataset(train, batch_size=batch_size)
    val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
    test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)

    epochs = 500

    print('Fit model on training data')
    verbose = 2 if n_targets == 1 else 3  # too much information in case of multiple losses
    history = model.fit(train_ds, epochs=epochs, validation_data=val_ds, callbacks=callbacks, verbose=verbose)
    print('\nhistory dict:', history.history)

    print('\nEvaluate on test data')
    results = model.evaluate(test_ds)
    print('test loss, test acc:', results)

    # test on generated data
    df_test = pd.read_csv(os.path.join('data', 'test_values2.csv'), sep=';')
    test_features = df_test[features].values.tolist()

    # transform test values according to the scaler on which the training data of the model was fitted
    x_test = scaler.transform(test_features)
    x_test = [v for v in np.array(x_test).T]
    y_real = df_test[targets].values
    # predict the stiffness values with the trained model
    y_predicted = model.predict(x_test)
    y_predicted = np.column_stack(y_predicted) if n_targets > 1 else y_predicted
    y_predicted = scaler_y.inverse_transform(y_predicted)

    # prepare the diagram data
    x = df_test['a11'].values
    colors = ['red', 'blue', 'black'] * 3
    label = 'compliance' if n_targets < 9 else 'stiffness'

    if n_targets == 9:
        # utility method to calculate stiffness from compliance
        def compliance_to_stiffness(arr):
            C11 = arr[targets.index('C11')]
            C22 = arr[targets.index('C22')]
            C33 = arr[targets.index('C33')]
            C12 = arr[targets.index('C12')]
            C23 = arr[targets.index('C23')]
            C13 = arr[targets.index('C13')]
            C44 = arr[targets.index('C44')]
            C55 = arr[targets.index('C55')]
            C66 = arr[targets.index('C66')]

            # assemble compliance matrix
            arr = np.array([
                [C11, C12, C13, 0, 0, 0],
                [C12, C22, C23, 0, 0, 0],
                [C13, C23, C33, 0, 0, 0],
                [0, 0, 0, C44, 0, 0],
                [0, 0, 0, 0, C55, 0],
                [0, 0, 0, 0, 0, C66],
            ])

            # calculate stiffness
            arr = np.linalg.inv(arr)

            S11 = arr[0, 0]
            S22 = arr[1, 1]
            S33 = arr[2, 2]
            S12 = arr[0, 1]
            S23 = arr[2, 1]
            S13 = arr[2, 0]
            S44 = arr[3, 3]
            S55 = arr[4, 4]
            S66 = arr[5, 5]

            return np.array([S11, S22, S33, S12, S23, S13, S44, S55, S66])

        for idx, (rr, rp) in enumerate(zip(y_real, y_predicted)):
            y_real[idx, :] = compliance_to_stiffness(rr)
            y_predicted[idx, :] = compliance_to_stiffness(rp)

    for idx, (t_, yr, yp, c_) in enumerate(zip(targets, y_real.T, y_predicted.T, colors)):
        plt.plot(x, yr, '+', color=c_, label=f'Digimat {t_}')
        plt.plot(x, yp, 'x', color=c_, label=f'Neuronal Network {t_}')
        # group 3 results respectively: e.g. C11, C22, C33...
        if (idx+1) % 3 == 0:
            plt.xlabel('a11 / -')
            plt.ylabel(label)
            plt.legend()
            plt.show()
