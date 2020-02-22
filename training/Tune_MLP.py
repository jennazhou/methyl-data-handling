import pandas as pd
import numpy as np

ppmi = pd.read_csv('./trans_processed_PPMI_data.csv')
ppmi.rename(columns={'Unnamed: 0':'Sentrix_position'}, inplace=True)
ppmi.set_index('Sentrix_position', inplace=True)
ppmi = ppmi.transpose()

from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
label = encoder.fit_transform(ppmi['Category'])

tr = ppmi.drop(['Category'], axis=1)
X = tr.values
y = label
print(X.shape)
print(y.shape)

#Stratified sampling
from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
split.get_n_splits(X, y)

for train_index, test_index in split.split(X, y):
    print("TRAIN:", len(train_index), "TEST:", len(test_index))
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

### Scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
####所有的test都只能apply transform，不能用fit_transform!!!
X_test = scaler.transform(X_test)





#---------------------------------------
# MLP model starts here
from keras import backend as K
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import to_categorical

import talos
from talos.utils.gpu_utils import multi_gpu

import os

###Hyperparameters for MLP
hidden_layer_neuron = [64, 64, 128, 256] #32
batch_size = [10], #, 10, 30
epochs = [10], #, 20, 30
dropout = [0]#try every .1 value between 0 and .5

for hln in hidden_layer_neruon:
    for bs in batch_size:
        for e in epochs:
            for do in dropout:
                p = {'lr': (0.8, 1.0, 3),
                     'first_neuron':[32, 64, 128, 256, 512],
                     'kernel_initializer': ['uniform'], #,'normal'
                     'activation':['relu'], #, 'elu', 'tanh'
                     'last_activation': ['sigmoid'] #, 'softmax'
                    }

                # The GPU id to use, usually either "0" or "1";
                os.environ["CUDA_VISIBLE_DEVICES"]="2";  
                def mlp_model(x_train, y_train, x_val, y_val, params):
                    # Build the model.
                    # Anyhow give parameters first
                    mlp = Sequential([
                      Dense(params['first_neuron'], activation=params['activation'], input_shape=(747668,), kernel_initializer=params['kernel_initializer']),
                      Dropout(do, seed=42),
                      Dense(hln, activation=params['activation']),
                      Dropout(do, seed=42),
                    #   Dense(8, activation='relu'),
                    #   Dropout(Dropout(0.50, seed=42)),
                      Dense(2, activation=params['last_activation']),
                    ])

                    # split a single job to multiple GPUs
                #     mlp = multi_gpu(mlp)

                    # Compile the model
                    mlp.compile(
                      optimizer='Adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy'],
                    )

                    # Train the data
                    history = mlp.fit(
                      x_train, # training data
                      to_categorical(y_train), # training targets
                      epochs=e,
                      batch_size=bs,
                    )

                    return history, mlp

                #Tune hyperparameters using talos
                ## To start fast, limit the permutation to 1/100 of the original permutation
                scan_object = talos.Scan(x=X_train,
                                         y=y_train, 
                                         params=p,
                                         model=mlp_model,
                                         experiment_name='mlp')

                # accessing the results data frame
                print("Dataframe of params:")
                print(scan_object.data)

                # accessing epoch entropy values for each round
                print("Learning entropy")
                print(scan_object.learning_entropy)

                # access the summary details
                print("Detail summary:")
                print(scan_object.details)

                sorted_so = scan_object.data.sort_values(by=["accuracy"], ascending=False)
                df_cur = pd.DataFrame(sorted_so.iloc[:1])
                print(df_cur)


####Documentation for training: as the number of neuron increases, accuracy does not necessarily increase, however the time taken for training increases significantly
#Documentation:
