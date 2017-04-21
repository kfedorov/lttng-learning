import argparse
import datetime
import os
import sys

import numpy as np
import tensorflow as tf

import autoencoder
import traceparser

arg_parser = argparse.ArgumentParser(description='Train autoencoder for trace embedding')
arg_parser.add_argument('trace', type=str, help='path to trace folder containing CTF data')
arg_parser.add_argument('-n', '--name', type=str, required=True, help='trace name, for saving purposes')

args = arg_parser.parse_args()

# Data
window_size = 100
test_size = 200

# Training
training_epochs = 300
batch_size = 500
clustering_factor = 0

# Saving
summary_frequency = 1
save_frequency = 40
dir_path = os.path.dirname(os.path.realpath(__file__))
save_dir = os.path.join(dir_path, 'res', datetime.datetime.now().strftime("%y%m%d_%H%M")) # /res/Date_HoursMinutes


print('Parsing trace data')

# Data
parser = traceparser.trace_parser(trace_file=args.trace,
                                trace_name=args.name,
                                window_size=window_size)

all_data = parser.windows.astype(float)
data_count = all_data.shape[0]

all_data_shuffled = np.random.permutation(all_data)
train = all_data_shuffled[:-test_size]
test = all_data_shuffled[-test_size:]

print('Data gathering done')

# Network Parameters
n_input = all_data.shape[1]


autoencoder = autoencoder.ClusteringAutoencoder([n_input, 128, 3],
                                                all_data.shape[0],
                                                activation_func=tf.nn.relu6,
                                                n_clusters=15,
                                                clustering_factor=clustering_factor,
                                                optimization_func=tf.train.AdamOptimizer(0.0001),
                                                save_path=save_dir)
# Launch the graph
autoencoder.initialize()

for epoch in range(training_epochs):
    cost, base_cost, clustering_incentive = autoencoder.train_epoch(train, batch_size, epoch % summary_frequency == 0)
    print("Epoch:", '%04d' % (epoch + 1), "cost= {:.9f} = {:.9f} + {:.9f} * {:.9f}".format(cost, base_cost, clustering_factor, clustering_incentive))

    if (epoch+1) % save_frequency == 0:
        autoencoder.save()

print("Training Finished")

autoencoder.save_embeddings(all_data, lambda row: parser.window_to_string(row, separator='||'))

centroids, counts = autoencoder.get_top_centroids(5)
print('centroids: ')
for centroid, count in zip(centroids,counts):
    print('With {} samples:\n'.format(count))
    print(parser.window_to_string(centroid))

# Applying cost over test set
_,_, _, base_cost = autoencoder.evaluate(test)
print("Mean square difference over test set: ", base_cost)
_, prediction, _, _ = autoencoder.evaluate(all_data[1:2])
print('expected (initial): ')
print(parser.window_to_string(all_data[1]))
print('actual (predicted): ')
print(parser.window_to_string(prediction[0]))

autoencoder.close()
