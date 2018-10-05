# Modified from https://github.com/jadianes/spark-movie-lens/blob/master/download_dataset.sh

#!/bin/bash

# Check that the system has all the needed packages
hash wget 2>/dev/null || { echo >&2 "Wget required.  Aborting."; exit 1; }
hash unzip 2>/dev/null || { echo >&2 "unzip required.  Aborting."; exit 1; }

# Downloading the data files, extracting and moving them to the desired location
wget http://files.grouplens.org/datasets/movielens/ml-latest.zip
unzip -o "ml-latest.zip"
DESTINATION=~/data/
[ -d $DESTINATION ] && echo "Directory already exists." || mkdir -p $DESTINATION && echo "Making new directory"
echo "Moving file to ~/data/"
mv ml-latest $DESTINATION
echo "Cleaning up..."
rm "ml-latest.zip"

# Splitting the data into training and testing data
ORIGINAL=`pwd`
cd `dirname $BASH_SOURCE`
echo Splitting the data into training and testing files...
python split_data.py
echo Done with train/test split!

# Sharding the training files
echo Sharding trainining files..
python shard_training_data.py
echo Done with sharding! Cleaning up...

# Cleaning up data file directory
DESTINATION=~/data/ml-latest/shards
[ -d $DESTINATION ] && echo "Directory already exists." || mkdir -p $DESTINATION && echo "Making new directory."
mv train_* $DESTINATION
echo Done cleaning. Shards can be found in $DESTINATION

# Returning to original directory
cd $ORIGINAL
