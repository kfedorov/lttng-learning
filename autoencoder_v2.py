import parsetrace_simple
import numpy as np
import tensorflow as tf
import os
import autoencoder2

dir_path = os.path.dirname(os.path.realpath(__file__))

learning_rate = 0.005
training_epochs = 300
batch_size = 500
display_step = 1
window_size = 100
test_size = 200


# Data
parser = parsetrace_simple.traceParser(trace_file='/home/kfedorov/lttng-traces/firefox/kernel/',
                                trace_name='firefox',
                                window_size=window_size)

all_data_initial = parser.windows
all_data = all_data_initial.astype(float)

data_count = all_data.shape[0]

all_data_shuffled = all_data
np.random.shuffle(all_data_shuffled)
train = all_data_shuffled[:-test_size]
test = all_data_shuffled[-test_size:]

print('Data gathering done')

# Network Parameters
n_hidden_1 = 128  # 1st layer num features
n_hidden_2 = 8  # 2nd layer num features
n_input = all_data[0].size
n_clusters = 16


autoencoder = autoencoder2.ClusteringAutoencoder([n_input, 256, 3],
                                                 activation_func=tf.nn.relu,
                                                 n_clusters=n_clusters,
                                                 clustering_factor=1E-5,
                                                 optimization_func=tf.train.RMSPropOptimizer(0.0001),
                                                 cost_func=lambda pred, actual: tf.reduce_mean(tf.square(pred - actual)) * window_size)

# Launch the graph
with tf.Session() as sess:
    autoencoder.initialize(sess)

    for epoch in range(training_epochs):
        cost = autoencoder.train_epoch(sess, train, batch_size)
        print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(cost))

    print("Optimization Finished!")
    # Applying cost over test set
    _,_, cost = autoencoder.evaluate(sess, test)
    print("Mean square difference over test set: ", cost)
    _, prediction, _ = autoencoder.evaluate(sess, all_data[1:2])
    print('expected (initial): ')
    print(parser.windowToString(all_data[1]))
    print('actual (predicted): ')
    print(parser.windowToString(prediction[0]))
