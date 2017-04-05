import os
import tensorflow as tf
import numpy as np
from tensorflow.contrib.tensorboard.plugins import projector


def default_weight_init(in_size, out_size):
    return tf.truncated_normal([in_size, out_size], stddev=np.sqrt(2.0 / out_size))


def default_bias_init(out_size):
    return tf.zeros([out_size])


def default_base_cost_func(prediction, actual):
    return tf.reduce_mean(tf.square(prediction - actual))

class ClusteringAutoencoder:
    encoder = {'weights': [], 'biases': []}
    decoder = {'weights': [], 'biases': []}
    global_step = 0
    sess = None

    def __init__(self,
                 dims,
                 embedding_count,
                 weight_init_func=default_weight_init,
                 bias_init_func=default_bias_init,
                 activation_func=tf.nn.relu,
                 n_clusters=16,
                 clustering_factor=1E-5,
                 cost_func=default_base_cost_func,
                 optimization_func=tf.train.RMSPropOptimizer(0.0005),
                 learning_rate=0.0005,
                 save_path='/res',
                 debug_mode=False):

        self.embedding_count = embedding_count
        self.n_clusters = n_clusters
        self.n_layers = len(dims) - 1
        self.input_size = dims[0]
        self.encoded_size = dims[-1]
        self.clustering_factor = clustering_factor
        self.save_path = save_path
        self.learning_rate = learning_rate
        self.debug_mode = debug_mode

        self.create_layers(dims, weight_init_func, bias_init_func)
        self.create_graph(activation_func, optimization_func, cost_func)
        self.create_summaries()

        if not os.path.exists(save_path):
            os.makedirs(save_path)
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
        self.centroids = tf.Variable(weight_init_func(self.n_clusters, self.encoded_size), name='centroids')
        self.next_centroids = tf.Variable(weight_init_func(self.n_clusters, self.encoded_size), name='next_centroids')
        self.centroids_count = tf.Variable(tf.zeros([self.n_clusters]), name='centroid_assign_count')

        # Used to assign embeddings at the end of the simulation
        self.embed_placeholder = tf.placeholder(tf.float32, [self.embedding_count, self.encoded_size])
        self.embeddings = tf.Variable(tf.zeros([self.embedding_count, self.encoded_size]), name='embeddings')

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
        self.best_centroid_index = tf.arg_min(diff_with_centroids, 1)
        # each encoded sample row becomes the centroid that best matches it
        best_centroids = tf.gather(self.centroids, tf.cast(self.best_centroid_index, tf.int32))

        # Reward closeness to centroids (centroid accuracy)
        self.clustering_incentive = tf.reduce_mean(tf.square(self.encoded_output - best_centroids))

        self.base_cost = base_cost_function(self.decoded_output, self.input)
        self.cost = self.base_cost + self.clustering_factor * self.clustering_incentive

        self.optimizer = optimization_func.minimize(self.cost)

        # decoded centroids
        self.decoded_centroids = decode(self.centroids)

        # update centroids
        self.init_next_centroids = [
            self.next_centroids.assign(tf.zeros([self.n_clusters, self.encoded_size], tf.float32)),
            self.centroids_count.assign(tf.zeros([self.n_clusters], tf.float32))]

        self.update_next_centroids = [tf.scatter_add(self.next_centroids, self.best_centroid_index,
                                                     self.encoded_output),
                                      tf.scatter_add(self.centroids_count, self.best_centroid_index,
                                                     tf.ones_like(self.best_centroid_index, dtype=tf.float32))]

        self.update_centroids = self.centroids.assign(tf.div(self.next_centroids,
                                                             tf.expand_dims(tf.maximum(self.centroids_count, 1), 1)))

        self.set_embeddings = self.embeddings.assign(self.embed_placeholder)

    def create_summaries(self):
        summaries = [tf.summary.scalar('total_cost', self.cost),
                     tf.summary.scalar('base_cost', self.base_cost),
                     tf.summary.scalar('clustering_incentive', self.clustering_incentive)]

        if self.debug_mode:
            # encoder
            for encoderVariableLists in self.encoder.items():
                for variable in enumerate(encoderVariableLists[1]):
                    # eg. encoder_weights_1 -> 1 is the layer
                    summary_name = 'encoder_{}_{}'.format(encoderVariableLists[0], variable[0])
                    summaries.append(tf.summary.histogram(summary_name, variable[1]))

            # decoder
            for encoderVariableLists in self.decoder.items():
                for variable in enumerate(encoderVariableLists[1]):
                    # eg. encoder_weights_1 -> 1 is the layer
                    summary_name = 'decoder_{}_{}'.format(encoderVariableLists[0], variable[0])
                    summaries.append(tf.summary.histogram(summary_name, variable[1]))

        self.summaries = tf.summary.merge(summaries)

    def initialize(self):
        if self.sess is not None:
            self.sess.close()

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.summary_writer = tf.summary.FileWriter(self.save_path, self.sess.graph)

    # Trains on one epoch
    def train_epoch(self, samples, batch_size, summarize=False):
        self.__check_sess__()
        batch_count = int(samples.shape[0] / batch_size)

        to_run = [self.cost, self.base_cost, self.clustering_incentive, self.optimizer, self.update_next_centroids]
        if summarize:
            to_run.append(self.summaries)

        self.sess.run(self.init_next_centroids)
        output = []
        for i in range(batch_count):
            batch_xs = samples[i * batch_size:(i + 1) * batch_size]
            output = self.sess.run(to_run, feed_dict={self.input: batch_xs})
            self.sess.run(self.update_centroids)

        self.global_step += 1

        if summarize:
            self.summary_writer.add_summary(output[-1], self.global_step)

        # returns only the costs
        return output[:3]

    def get_top_centroids(self, number):
        self.__check_sess__()
        centroids, counts = self.sess.run([self.decoded_centroids, self.centroids_count])
        return centroids[np.argsort(counts)][-number:], np.sort(counts)[-number:]

    def save(self):
        self.__check_sess__()
        self.saver.save(self.sess, os.path.join(self.save_path, 'graph'), global_step=self.global_step)

    def evaluate(self, samples):
        self.__check_sess__()
        return self.sess.run([self.encoded_output, self.decoded_output, self.cost, self.base_cost],
                             feed_dict={self.input: samples})

    def close(self):
        self.__check_sess__()
        self.sess.close()
        self.sess = None

    def save_embeddings(self, data, label_func):
        batch_size = 500
        sample_count = data.shape[0]
        embeddings = np.zeros((sample_count, self.encoded_size), dtype='float32')
        metadata_file = open(os.path.join(self.save_path, 'labels.tsv'), 'w')
        metadata_file.write('idx\tActual\tDecoded\t\cluster\n')

        def compute_partial_embeddings(start_idx, end_idx):
            embeddings[start_idx:end_idx], decoded, best_centroid = \
                self.sess.run([self.encoded_output, self.decoded_output, self.best_centroid_index],
                              feed_dict={self.input: data[start_idx:end_idx, :]})
            for j in range(decoded.shape[0]):
                metadata_file.write('%06d\t%s\t%s\t%02d\n' % (
                    j + start_idx, label_func(data[j + start_idx]), label_func(decoded[j]), best_centroid[j]))

        n_iter = int(sample_count / batch_size)
        for i in range(n_iter):
            start = i * batch_size
            end = (i + 1) * batch_size
            compute_partial_embeddings(start, end)

        start = n_iter * batch_size
        compute_partial_embeddings(start, None)
        self.sess.run(self.set_embeddings, feed_dict={self.embed_placeholder: embeddings})

        config = projector.ProjectorConfig()

        embedding = config.embeddings.add()
        embedding.tensor_name = self.embeddings.name
        embedding.metadata_path = os.path.join(self.save_path, 'labels.tsv')

        self.save()
        projector.visualize_embeddings(self.summary_writer, config)

    def __check_sess__(self):
        if self.sess is None:
            raise Exception('No session running.')
