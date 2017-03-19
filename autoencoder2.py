import numpy as np
import tensorflow as tf


def default_weight_init(in_size, out_size):
    return tf.truncated_normal([in_size, out_size], stddev=0.01)


def default_bias_init(out_size):
    return tf.zeros([out_size])

def default_base_cost_func(prediction, actual):
    return tf.reduce_mean(tf.square(prediction - actual))


# noinspection PyAttributeOutsideInit
class ClusteringAutoencoder:
    encoder = {'weights': [], 'biases': []}
    decoder = {'weights': [], 'biases': []}
    global_step = 0

    def __init__(self, dims,
                 weight_init_func=default_weight_init,
                 bias_init_func=default_bias_init,
                 activation_func=tf.nn.relu,
                 n_clusters=16,
                 clustering_factor=1E-5,
                 cost_func=default_base_cost_func,
                 optimization_func=tf.train.RMSPropOptimizer(0.0005),
                 learning_rate=0.0005,
                 save_path='/res'):
        self.n_clusters = n_clusters
        self.n_layers = len(dims) - 1
        self.input_size = dims[0]
        self.encoded_size = dims[-1]
        self.clustering_factor = clustering_factor
        self.save_path = save_path
        self.learning_rate = learning_rate

        self.create_layers(dims, weight_init_func, bias_init_func)
        self.create_graph(activation_func, optimization_func, cost_func)

        self.saver = tf.train.Saver()

    def create_layers(self, dims, weight_init_func, bias_init_func):

        for layer in range(self.n_layers):
            self.encoder['weights'].append(tf.Variable(weight_init_func(dims[layer], dims[layer + 1]),
                                                       name='encoder_weight_' + str(layer)))
            self.encoder['biases'].append(tf.Variable(bias_init_func(dims[layer + 1]),
                                                      name='encoder_bias_' + str(layer)))

            decoder_layer = self.n_layers - layer - 1
            self.decoder['weights'].append(tf.Variable(weight_init_func(dims[layer + 1], dims[layer]),
                                                       name='decoder_weight_' + str(decoder_layer)))
            self.decoder['biases'].append(tf.Variable(bias_init_func(dims[layer]),
                                                      name='decoder_bias_' + str(decoder_layer)))

        self.input = tf.placeholder(tf.float32, [None, self.input_size])
        self.centroids = tf.Variable(weight_init_func(self.n_clusters, self.encoded_size))
        self.next_centroids = tf.Variable(weight_init_func(self.n_clusters, self.encoded_size))
        self.centroids_count = tf.Variable(tf.zeros([self.n_clusters]))

    def create_graph(self, activation_func, optimization_func, base_cost_function):
        def encode(layer_input):
            for layer in range(self.n_layers):
                layer_input = activation_func(tf.add(tf.matmul(layer_input, self.encoder['weights'][layer]),
                                                     self.encoder['biases'][layer]))
            return layer_input

        def decode(layer_input):
            for layer in reversed(range(self.n_layers)):
                layer_input = activation_func(tf.add(tf.matmul(layer_input, self.decoder['weights'][layer]),
                                                     self.decoder['biases'][layer]))
            return layer_input

        self.encoded_output = encode(self.input)
        self.decoded_output = decode(self.encoded_output)

        # for each sample, get the mean difference with each centroid
        diff_with_centroids = tf.reduce_mean(tf.square(tf.map_fn(lambda x: tf.subtract(self.centroids, x),
                                                                 self.encoded_output)), 2)
        best_centroid_index = tf.arg_min(diff_with_centroids, 1)
        # each encoded sample row becomes the centroid that best matches it
        best_centroids = tf.gather(self.centroids, tf.cast(best_centroid_index, tf.int32))

        # Reward closeness to centroids (centroid accuracy)
        self.clustering_incentive = self.clustering_factor * tf.reduce_mean(
            tf.square(self.encoded_output - best_centroids))

        self.cost = base_cost_function(self.decoded_output, self.input) - self.clustering_incentive

        self.optimizer = optimization_func.minimize(self.cost)

        # update centroids
        self.init_next_centroids = [
            self.next_centroids.assign(tf.zeros([self.n_clusters, self.encoded_size], tf.float32)),
            self.centroids_count.assign(tf.zeros([self.n_clusters], tf.float32))]

        self.update_next_centroids = [tf.scatter_add(self.next_centroids, best_centroid_index, self.encoded_output),
                                      tf.scatter_add(self.centroids_count, best_centroid_index,
                                                     tf.ones_like(best_centroid_index, dtype=tf.float32))]

        self.update_centroids = self.centroids.assign(tf.div(self.next_centroids,
                                                             tf.expand_dims(tf.maximum(self.centroids_count, 1), 1)))

    def initialize(self, sess):
        sess.run(tf.global_variables_initializer())

    # Trains on one epoch
    def train_epoch(self, sess, samples, batch_size):
        sess.run(self.init_next_centroids)

        batch_count = int(samples.shape[0] / batch_size)

        for i in range(batch_count):
            batch_xs = samples[i * batch_size:(i + 1) * batch_size]
            c, _, _ = sess.run([self.cost, self.optimizer, self.update_next_centroids],
                               feed_dict={self.input: batch_xs})
            self.global_step += 1

        sess.run(self.update_centroids)
        return c

    def evaluate(self, sess, samples):
        return sess.run([self.encoded_output, self.decoded_output, self.cost], feed_dict={self.input: samples})
