import os
import numpy as np
import networkx as nx
import gzip
import pandas as pd
from urllib.request import urlopen
import stellargraph as sg
from stellargraph import StellarGraph
from stellargraph.data import EdgeSplitter
from stellargraph.mapper import GraphSAGELinkGenerator
from stellargraph.layer import GraphSAGE, link_classification
from stellargraph.data import UniformRandomWalk
from stellargraph.data import UnsupervisedSampler
from stellargraph import globalvar
from stellargraph import datasets

import stellargraph as sg
from stellargraph import StellarGraph
from stellargraph.data import EdgeSplitter
from stellargraph.mapper import GraphSAGELinkGenerator
from stellargraph.mapper import HinSAGELinkGenerator
from stellargraph.layer import GraphSAGE, link_classification, HinSAGE
from stellargraph.data import UniformRandomWalk
from stellargraph.data import UnsupervisedSampler
from stellargraph import globalvar
from stellargraph import datasets
from stellargraph.mapper import GraphSAGENodeGenerator, HinSAGENodeGenerator

from sklearn.model_selection import train_test_split
from sklearn import preprocessing, feature_extraction, model_selection
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tensorflow import keras
from sklearn import preprocessing, feature_extraction, model_selection
from sklearn.metrics import mean_absolute_error, mean_squared_error

from stellargraph.mapper import HinSAGELinkGenerator
from stellargraph.layer import HinSAGE, link_regression
from tensorflow.keras import Model, optimizers, losses, metrics

import multiprocessing
from stellargraph import datasets
from stellargraph import StellarGraph
from IPython.display import display, HTML
import matplotlib.pyplot as plt
import numpy as np
from datasets import Amz_Dataset

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print("Running without Metadata")

dataset = Amz_Dataset()

G, edges_with_ratings = dataset.load(metadata=False)

from sklearn import model_selection

edges_train, edges_test = model_selection.train_test_split(
    edges_with_ratings, train_size=0.7, test_size=0.3
)

edgelist_train = list(edges_train[["user_id", "item_id"]].itertuples(index=False))
edgelist_test = list(edges_test[["user_id", "item_id"]].itertuples(index=False))

labels_train = edges_train["rating"]
labels_test = edges_test["rating"]

batch_size = 200
epochs = 20

num_samples = [8, 4]

generator = HinSAGELinkGenerator(
    G, batch_size, num_samples, head_node_types=["user", "item"]
)
train_gen = generator.flow(edgelist_train, labels_train, shuffle=True)
test_gen = generator.flow(edgelist_test, labels_test)
generator.schema.type_adjacency_list(generator.head_node_types, len(num_samples))

hinsage_layer_sizes = [32, 32]
assert len(hinsage_layer_sizes) == len(num_samples)

hinsage = HinSAGE(
    layer_sizes=hinsage_layer_sizes, generator=generator, bias=True, dropout=0.0
)

# Expose input and output sockets of hinsage:
x_inp, x_out = hinsage.in_out_tensors()

# Final estimator layer
score_prediction = link_regression(edge_embedding_method="concat")(x_out)


def root_mean_square_error(s_true, s_pred):
    return K.sqrt(K.mean(K.pow(s_true - s_pred, 2)))


model = Model(inputs=x_inp, outputs=score_prediction)
model.compile(
    optimizer=optimizers.Adam(lr=1e-2),
    loss=losses.mean_squared_error,
    metrics=[root_mean_square_error, metrics.mae],
)

# Specify the number of workers to use for model training
num_workers = 4

generator = HinSAGELinkGenerator(
    G, batch_size, num_samples, head_node_types=["user", "item"]
)
train_gen = generator.flow(edgelist_train, labels_train, shuffle=True)
test_gen = generator.flow(edgelist_test, labels_test)

#
# def get_embeddings(G, nodes):
#     x_inp_src = x_inp[0::2]
#     x_out_src = x_out[0]
#     embedding_model = keras.Model(inputs=x_inp_src, outputs=x_out_src)
#     node_ids = nodes
#     # egd = HinSAGENodeGenerator(G, batch_size, num_samples, head_node_types=["user", "item"]).flow(node_ids)
#
#     node_embeddings = model.predict(test_gen, workers=4, verbose=1)
#
#     return node_embeddings
#
#
# new_embeddings = get_embeddings(G, G.nodes())


test_metrics = model.evaluate(
    test_gen, verbose=1, use_multiprocessing=False, workers=num_workers
)

print("Untrained model's Test Evaluation:")
for name, val in zip(model.metrics_names, test_metrics):
    print("\t{}: {:0.4f}".format(name, val))

history = model.fit(
    train_gen,
    validation_data=test_gen,
    epochs=epochs,
    verbose=1,
    shuffle=False,
    use_multiprocessing=False,
    workers=num_workers,
)

test_metrics = model.evaluate(
    test_gen, use_multiprocessing=False, workers=num_workers, verbose=1
)

print("Test Evaluation:")
for name, val in zip(model.metrics_names, test_metrics):
    print("\t{}: {:0.4f}".format(name, val))

y_true = labels_test
# Predict the rankings using the model:
y_pred = model.predict(test_gen)
# Mean baseline rankings = mean movie ranking:
y_pred_baseline = np.full_like(y_pred, np.mean(y_true))

rmse = np.sqrt(mean_squared_error(y_true, y_pred_baseline))
mae = mean_absolute_error(y_true, y_pred_baseline)
print("Mean Baseline Test set metrics:")
print("\troot_mean_square_error = ", rmse)
print("\tmean_absolute_error = ", mae)

rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mae = mean_absolute_error(y_true, y_pred)
print("\nModel Test set metrics:")
print("\troot_mean_square_error = ", rmse)
print("\tmean_absolute_error = ", mae)

# HinSAGE Variables

print("Running with Metadata")

dataset = Amz_Dataset()
G, edges_with_ratings = dataset.load(metadata=True)

from sklearn import model_selection

edges_train, edges_test = model_selection.train_test_split(
    edges_with_ratings, train_size=0.7, test_size=0.3
)

edgelist_train = list(edges_train[["user_id", "item_id"]].itertuples(index=False))
edgelist_test = list(edges_test[["user_id", "item_id"]].itertuples(index=False))

labels_train = edges_train["rating"]
labels_test = edges_test["rating"]

batch_size = 200
epochs = 20

num_samples = [8, 4]

generator = HinSAGELinkGenerator(
    G, batch_size, num_samples, head_node_types=["user", "item"]
)
train_gen = generator.flow(edgelist_train, labels_train, shuffle=True)
test_gen = generator.flow(edgelist_test, labels_test)
generator.schema.type_adjacency_list(generator.head_node_types, len(num_samples))

hinsage_layer_sizes = [32, 32]
assert len(hinsage_layer_sizes) == len(num_samples)

hinsage = HinSAGE(
    layer_sizes=hinsage_layer_sizes, generator=generator, bias=True, dropout=0.0
)

# Expose input and output sockets of hinsage:
x_inp, x_out = hinsage.in_out_tensors()

# Final estimator layer
score_prediction = link_regression(edge_embedding_method="concat")(x_out)

import tensorflow.keras.backend as K


def root_mean_square_error(s_true, s_pred):
    return K.sqrt(K.mean(K.pow(s_true - s_pred, 2)))


model = Model(inputs=x_inp, outputs=score_prediction)
model.compile(
    optimizer=optimizers.Adam(lr=1e-2),
    loss=losses.mean_squared_error,
    metrics=[root_mean_square_error, metrics.mae],
)

# Specify the number of workers to use for model training
num_workers = 4

generator = HinSAGELinkGenerator(
    G, batch_size, num_samples, head_node_types=["user", "item"]
)
train_gen = generator.flow(edgelist_train, labels_train, shuffle=True)
test_gen = generator.flow(edgelist_test, labels_test)

#
# def get_embeddings(G, nodes):
#     x_inp_src = x_inp[0::2]
#     x_out_src = x_out[0]
#     embedding_model = keras.Model(inputs=x_inp_src, outputs=x_out_src)
#     node_ids = nodes
#     # egd = HinSAGENodeGenerator(G, batch_size, num_samples, head_node_types=["user", "item"]).flow(node_ids)
#
#     node_embeddings = model.predict(test_gen, workers=4, verbose=1)
#
#     return node_embeddings
#
#
# new_embeddings = get_embeddings(G, G.nodes())


test_metrics = model.evaluate(
    test_gen, verbose=1, use_multiprocessing=False, workers=num_workers
)

print("Untrained model's Test Evaluation:")
for name, val in zip(model.metrics_names, test_metrics):
    print("\t{}: {:0.4f}".format(name, val))

history = model.fit(
    train_gen,
    validation_data=test_gen,
    epochs=epochs,
    verbose=1,
    shuffle=False,
    use_multiprocessing=False,
    workers=num_workers,
)

test_metrics = model.evaluate(
    test_gen, use_multiprocessing=False, workers=num_workers, verbose=1
)

print("Test Evaluation:")
for name, val in zip(model.metrics_names, test_metrics):
    print("\t{}: {:0.4f}".format(name, val))

y_true = labels_test
# Predict the rankings using the model:
y_pred = model.predict(test_gen)
# Mean baseline rankings = mean movie ranking:
y_pred_baseline = np.full_like(y_pred, np.mean(y_true))

rmse = np.sqrt(mean_squared_error(y_true, y_pred_baseline))
mae = mean_absolute_error(y_true, y_pred_baseline)
print("Mean Baseline Test set metrics:")
print("\troot_mean_square_error = ", rmse)
print("\tmean_absolute_error = ", mae)

rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mae = mean_absolute_error(y_true, y_pred)
print("\nModel Test set metrics:")
print("\troot_mean_square_error = ", rmse)
print("\tmean_absolute_error = ", mae)
