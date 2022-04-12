# Created by msinghal at 09/04/22


from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow.keras.backend as K
from stellargraph.mapper import HinSAGELinkGenerator
from stellargraph.layer import HinSAGE, link_regression
from tensorflow.keras import Model, optimizers, losses, metrics
import numpy as np

def root_mean_square_error(s_true, s_pred):
    return K.sqrt(K.mean(K.pow(s_true - s_pred, 2)))


class GraphicoRango():
    def __init__(self, edgelist_train, edgelist_test, labels_train, labels_test, epochs, batch_size, steller_graph, num_samples):
        self.model = None
        self.train_gen = None
        self.num_workers = 4
        self.edgelist_train = edgelist_train
        self.edgelist_test = edgelist_test
        self.labels_train = labels_train
        self.labels_test = labels_test
        self.epochs = epochs
        self.graph = steller_graph
        self.batch_size = batch_size
        self.num_samples = num_samples

    def compile(self):
        generator = HinSAGELinkGenerator(
            self.graph, self.batch_size, self.num_samples, head_node_types=["user", "item"]
        )
        self.train_gen = generator.flow(self.edgelist_train, self.labels_train, shuffle=True)
        self.test_gen = generator.flow(self.edgelist_test, self.labels_test)

        generator.schema.type_adjacency_list(generator.head_node_types, len(self.num_samples))

        print(generator.schema.schema)

        hinsage_layer_sizes = [32, 32]
        assert len(hinsage_layer_sizes) == len(self.num_samples)

        hinsage = HinSAGE(
            layer_sizes=hinsage_layer_sizes, generator=generator, bias=True, dropout=0.0
        )

        # Expose input and output sockets of hinsage:
        x_inp, x_out = hinsage.in_out_tensors()

        # Final estimator layer
        score_prediction = link_regression(edge_embedding_method="concat")(x_out)

        model = Model(inputs=x_inp, outputs=score_prediction)
        model.compile(
            optimizer=optimizers.Adam(lr=1e-2),
            loss=losses.mean_squared_error,
            metrics=[root_mean_square_error, metrics.mae],
        )

        print(model.summary())

        self.model = model

    def train_model(self):

        history = self.model.fit(
            self.train_gen,
            validation_data=self.test_gen,
            epochs=self.epochs,
            verbose=1,
            shuffle=False,
            use_multiprocessing=False,
            workers=self.num_workers,
        )
        return history

    def validate_model_accuracy(self):
        test_metrics = self.model.evaluate(
            self.test_gen, use_multiprocessing=False, workers=self.num_workers, verbose=1
        )

        print("Test Evaluation:")
        for name, val in zip(self.model.metrics_names, test_metrics):
            print("\t{}: {:0.4f}".format(name, val))

        y_true = self.labels_test
        # Predict the rankings using the model:
        y_pred = self.model.predict(self.test_gen)
        # Mean baseline rankings = mean edge ranking:
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