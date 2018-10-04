# Recommender_system

# Data
MovieLens 100K

# Purpose
Explore how different methods for collaborative filtering to compare. Will explore matrix-factorization (MF), MF 
with regularization, NN model (give u and m what is the probability of rating x), hybrid as well as multitask model.

# Running
To grab the necessary data run `source /path/to/get_dataset.sh`.  Once the dataset has been collected and formatted then
you can run python `python /path/to/basic_model.py` to start training the model. To view metrics on tensorboard, perform 
`tensorboard --logdir /path/to/containing_folder`. 