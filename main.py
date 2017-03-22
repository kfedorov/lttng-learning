import parsetrace_simple
import parsetrace
import numpy as np
import tensorflow as tf
import os
import autoencoder
import datetime


# Data
window_size = 100
test_size = 200

# Training
training_epochs = 300
batch_size = 500
clustering_factor = 1E-5

# Saving
summary_frequency = 1
save_frequency = 40
dir_path = os.path.dirname(os.path.realpath(__file__))
save_dir = os.path.join(dir_path, 'res', datetime.datetime.now().strftime("%y%m%d_%H%M")) # /res/Date_HoursMinutes


# Data
parser = parsetrace_simple.trace_parser(trace_file='/home/kfedorov/lttng-traces/firefox/kernel/',
                                trace_name='firefox',
                                window_size=window_size)

all_data =  parser.windows.astype(float)
data_count = all_data.shape[0]

all_data_shuffled = np.random.permutation(all_data)
train = all_data_shuffled[:-test_size]
test = all_data_shuffled[-test_size:]

print('Data gathering done')

# Network Parameters
n_input = all_data[0].size


autoencoder = autoencoder.ClusteringAutoencoder([n_input, 128, 8],
                                                activation_func=tf.nn.relu6,
                                                n_clusters=32,
                                                clustering_factor=clustering_factor,
                                                optimization_func=tf.train.AdamOptimizer(0.0005),
                                                save_path=save_dir,
                                                debug_mode=False,
                                                cost_func=lambda pred, actual: tf.reduce_mean(tf.square(pred - actual)) * window_size)

# Launch the graph
autoencoder.initialize()

for epoch in range(training_epochs):
    cost, base_cost, clustering_incentive = autoencoder.train_epoch(train, batch_size, epoch % summary_frequency == 0)
    print("Epoch:", '%04d' % (epoch + 1), "cost= {:.9f} = {:.9f} - {:.9f} * {:.9f}".format(cost, base_cost, clustering_factor, clustering_incentive))

    if (epoch+1) % save_frequency == 0:
        autoencoder.save()

print("Training Finished")

autoencoder.save_embeddings(all_data, lambda row: parser.windowToString(row, separator='||'))

centroids, counts = autoencoder.get_top_centroids(5)
print('centroids: ')
for centroid, count in zip(centroids,counts):
    print('With {} samples:\n'.format(count))
    print(parser.windowToString(centroid))

# Applying cost over test set
_,_, _, base_cost = autoencoder.evaluate(test)
print("Mean square difference over test set: ", base_cost)
_, prediction, _, _ = autoencoder.evaluate(all_data[1:2])
print('expected (initial): ')
print(parser.windowToString(all_data[1]))
print('actual (predicted): ')
print(parser.windowToString(prediction[0]))

autoencoder.close()
