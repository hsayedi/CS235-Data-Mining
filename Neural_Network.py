#!/usr/bin/env python
# coding: utf-8

# ## Implementation - Kevin Vega

import collections
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf

from keras.layers import Dense
from keras.models import Sequential

from sklearn.preprocessing import LabelEncoder

# ## Neural Network


def process_data(file="data/new-york-city-airbnb-open-data/AB_NYC_2019.csv",
                 target='neighbourhood_group', test_size=0.2):
    df = pd.read_csv(str(file))
    
    # Convert date strings to datetime
    df['last_review'] = pd.to_datetime(df.last_review)
    
    # Remove unnecessary data
    df.drop(['id', 'name', 'host_name', 'host_id'], axis=1, inplace=True)
    
    # Convert categorical data to numerical
    le = LabelEncoder()
    df['neighbourhood_group'] = pd.Series(le.fit_transform(df['neighbourhood_group']))
    df['neighbourhood'] = pd.Series(le.fit_transform(df['neighbourhood']))
    df['room_type'] = pd.Series(le.fit_transform(df['room_type']))
    df['last_review'] = pd.Series(le.fit_transform(df['last_review']))
    
    # Remove NaN and non-definite values from table
    with pd.option_context('mode.use_inf_as_null', True):
        df = df.dropna()
    
    # Ensure category values are integer
    df['neighbourhood_group'] = df['neighbourhood_group'].astype(int)
    df['neighbourhood'] = df['neighbourhood'].astype(int)
    df['room_type'] = df['room_type'].astype(int)
    df['last_review'] = df['last_review'].astype(int)
    
    # Separate prices from features
    features_df = df.loc[:, df.columns != str(target)] 
    target_df = df[str(target)]
    
    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(features_df, target_df, test_size=0.2)
    
    # Normalize the data
    X_train = tf.keras.utils.normalize(X_train, axis=1)
    X_test = tf.keras.utils.normalize(X_test, axis=1)
    
    print("Shapes - X_train: {}, X_test: {}".format(X_train.shape, X_test.shape) +
          "\n         y_train: {}. y_test: {}".format(y_train.shape, y_test.shape))
    
    return X_train, X_test, y_train, y_test


# In[137]:


def create_model(feature_dim, output_dim, neurons, hidden_layers=1, activation='relu', loss_func='sparse_categorical_crossentropy', optimizer_param='adam',
                output_activation='softmax'):
    model = Sequential()
    
    # Add input layer, with dimensions equal to input features
    model.add(Dense(neurons, input_dim=feature_dim, activation=activation))
    
    for i in range(hidden_layers):
        model.add(Dense(neurons, activation=activation))
        
    model.add(Dense(output_dim, activation=output_activation))
    
    model.compile(optimizer=optimizer_param,
                loss=loss_func,
                metrics=['accuracy'])
    
    return model


def test_models(X_train, X_test, y_train, y_test, output_dim, n_models=4, neurons=8, epochs=100, activation='relu',
               verbose=0, loss_func='sparse_categorical_crossentropy', optimizer_param='adam', output_activation='softmax'):
    
    feature_dim = X_train.shape[1]
    
    models = {}
    
    histories = []
    
    print("{} neuron(s) per hidden layer\n".format(neurons))
    for i in range(n_models):
        models[i] = create_model(feature_dim, output_dim=output_dim, neurons=neurons, hidden_layers=i+1, activation='relu', loss_func=loss_func, optimizer_param=optimizer_param,
                                output_activation=output_activation)
        print("Fitting model: {}, has {} hidden layer(s)".format(i+1, i+1))
        histories.append(models[i].fit(X_train, y_train, epochs=epochs, verbose=verbose))
        print("")
    
    return models, histories


def plot_histories(histories, number_of_neurons, feature_name, save=False):
    
    if type(feature_name) != str:
        print("Please enter desired feature name as string")
        return
    
    for i in range(len(histories)):
        plt.plot(histories[i].history['acc'], label="{}-Hidden Layers".format(i+1))
    plt.title('{} - Model Accuracy, {} neurons per layer'.format(feature_name, number_of_neurons))
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    if save:
        plt.savefig("{}-{}-hidden-{}-neurons-acc.png".format(feature_name, i+1, number_of_neurons))
    plt.show()
    
    for i in range(len(histories)):
        plt.plot(histories[i].history['loss'], label="{}-Hidden Layers".format(i+1))
    plt.title('{} - Model Loss, {} neurons per layer'.format(feature_name, number_of_neurons))
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    if save:
        plt.savefig("{}-{}-hidden-{}-neurons-loss.png".format(feature_name, i+1, number_of_neurons))
    plt.show()


def gen_graphs(file, target='neighbourhood_group', output_dim=5, epochs=100, list_of_neurons=[8, 16, 32], loss_func='sparse_categorical_crossentropy',
               optimizer_param='adam', output_activation='softmax', save=False, verbose=1):
    
    X_train, X_test, y_train, y_test = process_data(target=target)

    for i in list_of_neurons:
        models, histories = test_models(X_train, X_test, y_train, y_test, output_dim, epochs=epochs, neurons=i, loss_func=loss_func, optimizer_param=optimizer_param,
                    output_activation=output_activation, verbose=verbose)

        plot_histories(histories, i, "Price", save=save)




