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
import networkx as nx

import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# from sklearn.metrics import mean_absolute_error, mean_squared_error
#
# import stellargraph as sg
# from stellargraph.mapper import HinSAGELinkGenerator
# from stellargraph.layer import HinSAGE, link_regression
# from tensorflow.keras import Model, optimizers, losses, metrics
#
# import multiprocessing
# from stellargraph import datasets
# from IPython.display import display, HTML
# import matplotlib.pyplot as plt
from stellargraph.datasets import DatasetLoader

# class MovieLens(
#     DatasetLoader,
#     name="MovieLens",
#     directory_name="ml-100k",
#     url="http://files.grouplens.org/datasets/movielens/ml-100k.zip",
#     url_archive_format="zip",
#     expected_files=["u.data", "u.user", "u.item", "u.genre", "u.occupation"],
#     description="The MovieLens 100K dataset contains 100,000 ratings from 943 users on 1682 movies.",
#     source="https://grouplens.org/datasets/movielens/100k/",
# ):
#     def load(self):
#         """
#         Load this dataset into an undirected heterogeneous graph, downloading it if required.
#
#         The graph has two types of nodes (``user`` and ``movie``) and one type of edge (``rating``).
#
#         The dataset includes some node features on both users and movies: on users, they consist of
#         categorical features (``gender`` and ``job``) which are one-hot encoded into binary
#         features, and an ``age`` feature that is scaled to have mean = 0 and standard deviation = 1.
#
#         Returns:
#             A tuple where the first element is a :class:`.StellarGraph` instance containing the graph
#             data and features, and the second element is a pandas DataFrame of edges, with columns
#             ``user_id``, ``movie_id`` and ``rating`` (a label from 1 to 5).
#         """
#         self.download()
#
#         ratings, users, movies, *_ = [
#             self._resolve_path(path) for path in self.expected_files
#         ]
#
#         edges = pd.read_csv(
#             ratings,
#             sep="\t",
#             header=None,
#             names=["user_id", "movie_id", "rating", "timestamp"],
#             usecols=["user_id", "movie_id", "rating"],
#         )
#
#         users = pd.read_csv(
#             users,
#             sep="|",
#             header=None,
#             names=["user_id", "age", "gender", "job", "zipcode"],
#             usecols=["user_id", "age", "gender", "job"],
#         )
#
#         movie_columns = [
#             "movie_id",
#             "title",
#             "release_date",
#             "video_release_date",
#             "imdb_url",
#             # features from here:
#             "unknown",
#             "action",
#             "adventure",
#             "animation",
#             "childrens",
#             "comedy",
#             "crime",
#             "documentary",
#             "drama",
#             "fantasy",
#             "film_noir",
#             "horror",
#             "musical",
#             "mystery",
#             "romance",
#             "sci_fi",
#             "thriller",
#             "war",
#             "western",
#         ]
#         movies = pd.read_csv(
#             movies,
#             sep="|",
#             header=None,
#             names=movie_columns,
#             usecols=["movie_id"] + movie_columns[5:],
#         )
#
#         # manage the IDs
#         def u(users):
#             return "u_" + users.astype(str)
#
#         def m(movies):
#             return "m_" + movies.astype(str)
#
#         users_ids = u(users["user_id"])
#
#         movies["movie_id"] = m(movies["movie_id"])
#         movies.set_index("movie_id", inplace=True)
#
#         edges["user_id"] = u(edges["user_id"])
#         edges["movie_id"] = m(edges["movie_id"])
#
#         # convert categorical user features to numeric, and normalize age
#         feature_encoding = preprocessing.OneHotEncoder(sparse=False)
#         onehot = feature_encoding.fit_transform(users[["gender", "job"]])
#         scaled_age = preprocessing.scale(users["age"])
#         encoded_users = pd.DataFrame(onehot, index=users_ids).assign(
#             scaled_age=scaled_age
#         )
#
#         g = StellarGraph(
#             {"user": encoded_users, "movie": movies},
#             {"rating": edges[["user_id", "movie_id"]]},
#             source_column="user_id",
#             target_column="movie_id",
#         )
#         return g, edges
#
#
# dataset = MovieLens()
# G, edges_with_ratings = dataset.load()
#
#
# debug=1

import networkx as nx


class Amz_Dataset():
    def load(self, metadata=False):
        """
        Load this dataset into an undirected heterogeneous graph, downloading it if required.

        The graph has two types of nodes (``user`` and ``movie``) and one type of edge (``rating``).

        The dataset includes some node features on both users and movies: on users, they consist of
        categorical features (``gender`` and ``job``) which are one-hot encoded into binary
        features, and an ``age`` feature that is scaled to have mean = 0 and standard deviation = 1.

        Returns:
            A tuple where the first element is a :class:`.StellarGraph` instance containing the graph
            data and features, and the second element is a pandas DataFrame of edges, with columns
            ``user_id``, ``movie_id`` and ``rating`` (a label from 1 to 5).
        """
        rating_file = "/Users/msinghal/SourceCodes/Personal/graphranko/LightGCN/Data/amazon-electronics-v2/user-item-rating.csv"
        user_file = "/Users/msinghal/SourceCodes/Personal/graphranko/LightGCN/Data/amazon-electronics-v2/user_list.txt"
        item_file = "/Users/msinghal/SourceCodes/Personal/graphranko/LightGCN/Data/amazon-electronics-v2/item_meta_list.txt"
        category_file = "/Users/msinghal/SourceCodes/Personal/graphranko/LightGCN/Data/amazon-electronics-v2/categories.txt"

        edges = pd.read_csv(rating_file)

        users = pd.read_csv(
            user_file,
            sep=" ",
            header=None,
            names=["user", "user_id"]
        )

        categories = pd.read_csv(
            category_file,
            sep="|",
            header=None,
            names=["category_id", "category"]
        )

        items = pd.read_csv(
            item_file,
            sep=" ",
            header=None,
            names=["item", "item_id"] + list(categories["category"])
        )

        # manage the IDs
        def u(users):
            return "u_" + users.astype(str)

        def m(movies):
            return "i_" + movies.astype(str)

        print(edges.columns)
        print(users.columns)

        edges = edges.merge(users, on="user").merge(items, on="item")

        users["user_id"] = u(users["user_id"])
        users.set_index("user_id", inplace=True)

        #
        items["item_id"] = m(items["item_id"])
        items.set_index("item_id", inplace=True)
        items.drop("item", inplace=True, axis=1)
        #
        edges["user_id"] = u(edges["user_id"])
        edges["item_id"] = m(edges["item_id"])

        # convert categorical user features to numeric, and normalize age
        # feature_encoding = preprocessing.OneHotEncoder(sparse=False)
        # onehot = feature_encoding.fit_transform(users[["user"]])
        # scaled_age = preprocessing.scale(users["age"])
        values = np.zeros((len(users.index), len(users.index)), dtype=np.uint8)
        np.fill_diagonal(values, 1)
        encoded_users = pd.DataFrame(values, index=users.index, columns=users.index)
        if not metadata:
            values = np.zeros((len(items.index), len(items.index)), dtype=np.int8)
            np.fill_diagonal(values, 1)
            encoded_items = pd.DataFrame(values, index=items.index, columns=items.index, dtype=np.uint8)
            return self.generate_nx_graph_with_features(edges, encoded_users, encoded_items), edges
        return self.generate_nx_graph_with_features(edges, encoded_users, items), edges

    def generate_nx_graph_with_features(self, edges, users, items):
        G = nx.Graph()
        # Add nodes
        for user_id in users.index:
            G.add_node(user_id, feature=users.loc[user_id, :], label="user")
        for item_id in items.index:
            G.add_node(item_id, feature=items.loc[item_id, :], label="item")

        for index, row in edges.iterrows():
            src_node_id = row['user_id']
            dst_node_id = row['item_id']
            G.add_edge(src_node_id, dst_node_id, rating=row['rating'])
        return StellarGraph.from_networkx(G, node_features="feature")



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


#HinSAGE Variables

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
