# Created by msinghal at 09/04/22


import os
import numpy as np
import networkx as nx
import pandas as pd
from sklearn import preprocessing
from stellargraph import StellarGraph

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from stellargraph.datasets import DatasetLoader


class MovieLens(
    DatasetLoader,
    name="MovieLens",
    directory_name="ml-100k",
    url="http://files.grouplens.org/datasets/movielens/ml-100k.zip",
    url_archive_format="zip",
    expected_files=["u.data", "u.user", "u.item", "u.genre", "u.occupation"],
    description="The MovieLens 100K dataset contains 100,000 ratings from 943 users on 1682 movies.",
    source="https://grouplens.org/datasets/movielens/100k/",
):
    def load(self):
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
        self.download()

        ratings, users, movies, *_ = [
            self._resolve_path(path) for path in self.expected_files
        ]

        edges = pd.read_csv(
            ratings,
            sep="\t",
            header=None,
            names=["user_id", "movie_id", "rating", "timestamp"],
            usecols=["user_id", "movie_id", "rating"],
        )

        users = pd.read_csv(
            users,
            sep="|",
            header=None,
            names=["user_id", "age", "gender", "job", "zipcode"],
            usecols=["user_id", "age", "gender", "job"],
        )

        movie_columns = [
            "movie_id",
            "title",
            "release_date",
            "video_release_date",
            "imdb_url",
            # features from here:
            "unknown",
            "action",
            "adventure",
            "animation",
            "childrens",
            "comedy",
            "crime",
            "documentary",
            "drama",
            "fantasy",
            "film_noir",
            "horror",
            "musical",
            "mystery",
            "romance",
            "sci_fi",
            "thriller",
            "war",
            "western",
        ]
        movies = pd.read_csv(
            movies,
            sep="|",
            header=None,
            names=movie_columns,
            usecols=["movie_id"] + movie_columns[5:],
        )

        # manage the IDs
        def u(users):
            return "u_" + users.astype(str)

        def m(movies):
            return "m_" + movies.astype(str)

        users_ids = u(users["user_id"])

        movies["movie_id"] = m(movies["movie_id"])
        movies.set_index("movie_id", inplace=True)

        edges["user_id"] = u(edges["user_id"])
        edges["movie_id"] = m(edges["movie_id"])

        # convert categorical user features to numeric, and normalize age
        feature_encoding = preprocessing.OneHotEncoder(sparse=False)
        onehot = feature_encoding.fit_transform(users[["gender", "job"]])
        scaled_age = preprocessing.scale(users["age"])
        encoded_users = pd.DataFrame(onehot, index=users_ids).assign(
            scaled_age=scaled_age
        )

        g = StellarGraph(
            {"user": encoded_users, "movie": movies},
            {"rating": edges[["user_id", "movie_id"]]},
            source_column="user_id",
            target_column="movie_id",
        )
        return g, edges


class Amz_Dataset():

    def load(self, base_path="/Users/msinghal/SourceCodes/Personal/graphranko/Data/amazon-electronics-v4"):
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
        rating_file = base_path + "/user-item-rating.csv"
        user_file = base_path + "/user_list.txt"
        item_file = base_path + "/item_meta_list.txt"
        category_file = base_path + "/categories.txt"

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
        # Create adjacency matrix

        return users, items, edges

    def create_graph_for_training(self, users, items, edges, metadata=False):
        values = np.zeros((len(users.index), len(users.index)), dtype=np.uint8)
        np.fill_diagonal(values, 1)
        encoded_users = pd.DataFrame(values, index=users.index, columns=users.index, dtype=np.uint8)
        if not metadata:
            values = np.zeros((len(items.index), len(items.index)), dtype=np.uint8)
            np.fill_diagonal(values, 1)
            encoded_items = pd.DataFrame(values, index=items.index, columns=items.index, dtype=np.uint8)
            return self.generate_nx_graph_with_features(edges, encoded_users, encoded_items)
        return self.generate_nx_graph_with_features(edges, encoded_users, items)

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