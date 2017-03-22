TODO
====
* Cleanup trace_parsers. See for interfaces in python?
* Try centering data... (again, but correctly this time)
* Run tests on different parameters
    * Layer number and size
    * Learning rate / optimizer
    * Clustering factor
    ...

BEST SCORES
===========
[n_input, 128, 8],                                      || Epoch: 0300 cost= 0.005008779
activation_func=tf.nn.relu6,                            || over test set:  0.00454351
n_clusters=32,
clustering_factor=1E-5,
optimization_func=tf.train.AdamOptimizer(0.0005)