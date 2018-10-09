# Modified from https://github.com/jadianes/spark-movie-lens/blob/master/download_dataset.sh
#!/bin/bash

# Check that the system has all the needed packages
hash wget 2>/dev/null || { echo >&2 "Wget required.  Aborting."; exit 1; }
hash unzip 2>/dev/null || { echo >&2 "unzip required.  Aborting."; exit 1; }

# Downloading the data files, extracting and moving them to the desired location
# TODO: need to refactor such that the data files are contained within the same directory.
wget http://files.grouplens.org/datasets/movielens/ml-latest.zip
unzip -o "ml-latest.zip"

# Location where to place the data
ROOT= `dirname $BASH_SOURCE`
[ -d $ROOT/data ] && echo "Directory already exists." || (mkdir -p $ROOT && echo "Making new directory")
echo "Moving file to $ROOT/data"
mv ml-latest $ROOT/data
echo "Cleaning up..."
rm "ml-latest.zip"

# Splitting the data into training and testing data
ORIGINAL=`pwd`
cd $ROOT  # Gets the containing directory of the bash script
echo Splitting the data into training and testing files...
python split_data.py && echo "Done with train/test split!" || echo "FAILED!"

# Sharding the training files
echo Sharding trainining files..
python shard_training_data.py && echo "Sharding complete! Cleaning up..." || echo "FAILED!"

# Cleaning up data file directory
SHARD_ROOT=$ROOT/data/ml-latest/shards
[ -d SHARD_ROOT ] && echo "Directory already exists." || (mkdir -p $ROOT && echo "Making new directory.")
mv $ROOT/data/ml-latest/train_* $SHARD_ROOT && echo "Done cleaning. Shards moved to $SHARD_ROOT" || echo "FAILED!"

# Returning to original directory
cd $ORIGINAL
