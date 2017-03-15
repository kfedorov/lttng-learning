import parsetrace_simple
import numpy as np
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
import os
import datetime

dir_path = os.path.dirname(os.path.realpath(__file__))

learning_rate = 0.005
training_epochs = 300
batch_size = 500
display_step = 1
window_size = 100
test_size = 200
log_dir = dir_path + '/res/' + datetime.datetime.now().strftime("%y%m%d_%H%M")
lmda = 1E-5


os.makedirs(log_dir)

# Data
parser = parsetrace_simple.traceParser(trace_file='/home/kfedorov/lttng-traces/firefox/kernel/',
                                trace_name='firefox',
                                window_size=window_size)

all_data_initial = parser.windows
all_data = all_data_initial.astype(float)

# centering
all_data_mean = np.mean(all_data, axis=0)
# all_data -= all_data_mean


# normalization
def div0(a, b):
    """ ignore / 0, div0( [-1, 0, 1], 0 ) -> [0, 0, 0] """
    with np.errstate(divide='ignore', invalid='ignore'):
        res = np.true_divide(a, b)
        res[~ np.isfinite(res)] = 0  # -inf inf NaN
    return res


all_data_std = np.std(all_data, axis=0)
# all_data = div0(all_data, all_data_std)

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
# tf Graph input
X = tf.placeholder(tf.float32, [None, n_input])

def inverse(prediction):
    # prediction *= all_data_std
    # prediction += all_data_mean
    return prediction


weights = {
    'encoder_h1': tf.Variable(tf.truncated_normal([n_input, n_hidden_1], stddev=0.01),
                              name='encoder_h1'),
    'encoder_h2': tf.Variable(tf.truncated_normal([n_hidden_1, n_hidden_2], stddev=0.01),
                              name='encoder_h2'),
    'decoder_h1': tf.Variable(tf.truncated_normal([n_hidden_2, n_hidden_1], stddev=0.01),
                              name='decode_h1'),
    'decoder_h2': tf.Variable(tf.truncated_normal([n_hidden_1, n_input], stddev=0.01),
                              name='decoder_h2'),
}
biases = {
    'encoder_b1': tf.Variable(tf.zeros([n_hidden_1]), name='encoder_b1'),
    'encoder_b2': tf.Variable(tf.zeros([n_hidden_2]), name='encoder_b2'),
    'decoder_b1': tf.Variable(tf.zeros([n_hidden_1]), name='decoder_b1'),
    'decoder_b2': tf.Variable(tf.zeros([n_input]), name='decoder_b2')
}

centroids = tf.Variable(tf.truncated_normal([n_clusters, n_hidden_2], stddev=0.01))
next_centroids = tf.Variable(tf.zeros([n_clusters, n_hidden_2]))
centroids_count = tf.Variable(tf.zeros([n_clusters]))


# Saving
saver = tf.train.Saver({**weights, **biases})


# summarizing
def variable_summaries(name, var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    tf.summary.histogram(name, var)


# Building the encoder
def encoder(x):
    with tf.name_scope('encoder'):
        variable_summaries('Initial_Value', x)
        variable_summaries('encoder_h1', weights['encoder_h1'])
        variable_summaries('encoder_h2', weights['encoder_h2'])
        variable_summaries('encoder_b1', biases['encoder_b1'])
        variable_summaries('encoder_b2', biases['encoder_b2'])
        # Encoder Hidden layer with sigmoid activation #1
        layer_1 = tf.nn.relu(tf.add(tf.matmul(x, weights['encoder_h1']), biases['encoder_b1']))
        variable_summaries('layer_1', layer_1)
        # Decoder Hidden layer with sigmoid activation #2
        layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, weights['encoder_h2']), biases['encoder_b2']))
        variable_summaries('Encoded_Layer', layer_2)
    return layer_2


# Building the decoder
def decoder(x):
    with tf.name_scope('decoder'):
        variable_summaries('decoder_h1', weights['decoder_h1'])
        variable_summaries('decoder_h2', weights['decoder_h2'])

        variable_summaries('decoder_b1', biases['decoder_b1'])
        variable_summaries('decoder_b2', biases['decoder_b2'])
        # Encoder Hidden layer with sigmoid activation #1
        layer_1 = tf.nn.relu(tf.add(tf.matmul(x, weights['decoder_h1']), biases['decoder_b1']))
        variable_summaries('layer_1', layer_1)
        # Decoder Hidden layer with sigmoid activation #2
        layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, weights['decoder_h2']),  biases['decoder_b2']))
        variable_summaries('decoded_layer', layer_2)
    return layer_2


# Construct model
encoder_op = encoder(X)
decoder_op = decoder(encoder_op)

# Prediction
y_pred = decoder_op
# Targets (Labels) are the input data.
y_true = X


diffToCentroid = tf.reduce_mean(tf.square(tf.map_fn(lambda x: tf.subtract(centroids, x), encoder_op)),2)
bestCentroidIdx = tf.arg_min(diffToCentroid, 1)
bestCentroid = tf.gather(centroids, tf.cast(bestCentroidIdx, tf.int32))

cost = tf.reduce_mean(tf.square(y_pred - y_true)) * window_size - lmda * tf.reduce_mean(tf.square(encoder_op - bestCentroid))


# update cluster
init_new_centroids = next_centroids.assign(tf.zeros([n_clusters, n_hidden_2], tf.float32))
init_centroid_counts = centroids_count.assign(tf.zeros([n_clusters], tf.float32))

update_new_centroids = tf.scatter_add(next_centroids, bestCentroidIdx, encoder_op)
count_centroids = tf.scatter_add(centroids_count, bestCentroidIdx, tf.ones_like(bestCentroidIdx, dtype=tf.float32))

update_centroids = centroids.assign(tf.div(next_centroids, tf.expand_dims(tf.maximum(centroids_count, 1), 1)))
tf.summary.histogram("centroids_count", centroids_count)


costSummary = tf.summary.scalar("Cost", cost)

optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)
mergedSummaries = tf.summary.merge_all()

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    train_writer = tf.summary.FileWriter(log_dir, sess.graph)
    print('Session initialized')
    total_batch = int(train.shape[0] / batch_size)
    print('total batch ', total_batch)
    global_step = 0
    # Training cycle
    for epoch in range(training_epochs):
        # Loop over all batches
        sess.run([init_new_centroids, init_centroid_counts])
        for i in range(total_batch):
            batch_xs = train[i * batch_size:(i + 1) * batch_size]
            summariesToRun = costSummary
            if i == total_batch-1:
                summariesToRun = mergedSummaries
            summaries, c, _, _, _ = sess.run([summariesToRun, cost, optimizer, update_new_centroids, count_centroids],
                                             feed_dict={X: batch_xs})
            train_writer.add_summary(summaries, global_step)
            global_step += 1

        #update centroids
        sess.run([update_centroids])

        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch + 1),
                  "cost=", "{:.9f}".format(c))
            saver.save(sess, os.path.join(log_dir, "graph"), global_step=global_step)

    output_dir = log_dir + "_embed/"
    os.makedirs(output_dir)
    N = all_data.shape[0]
    EMB = np.zeros((N, n_hidden_2), dtype='float32')
    metadata_file = open(os.path.join(output_dir, 'labels.tsv'), 'w')
    metadata_file.write('idx\tActual\tDecoded\n')
    for i in range(N):
        EMB[i], decoded = sess.run([encoder_op, decoder_op], feed_dict={X: all_data[i:i + 1, :]})
        metadata_file.write('%06d\t%s\t%s\n' % (i,
                                                parser.windowToString(all_data_initial[i], separator='||'),
                                                parser.windowToString(inverse(decoded[0]), separator='||')))

    embedding_var = tf.Variable(EMB, name='embeddings')
    sess.run(embedding_var.initializer)
    summary_writer = tf.summary.FileWriter(output_dir, sess.graph)
    # Format: tensorflow/contrib/tensorboard/plugins/projector/projector_config.proto
    config = projector.ProjectorConfig()

    sess.run(embedding_var.initializer)

    # You can add multiple embeddings. Here we add only one.
    embedding = config.embeddings.add()
    embedding.tensor_name = embedding_var.name
    embedding.metadata_path = os.path.join(output_dir, 'labels.tsv')

    # You can add multiple embeddings. Here we add only one.
    # embedding = config.embeddings.add()
    # embedding.tensor_name = weights['encoder_h1'].name
    saver = tf.train.Saver([embedding_var])
    saver.save(sess, os.path.join(output_dir, 'model.ckpt'), training_epochs * total_batch)
    projector.visualize_embeddings(summary_writer, config)

    print("Optimization Finished!")
    # Applying cost over test set
    c = sess.run(cost, feed_dict={X: test[:]})
    pred = sess.run(decoder_op, feed_dict={X: all_data[1:2]})
    print("Mean square difference over test set: ", c)
    print('expected (initial): ')
    print(parser.windowToString(all_data_initial[1]))
    print('actual (predicted): ')
    print(parser.windowToString(inverse(pred[0])))
