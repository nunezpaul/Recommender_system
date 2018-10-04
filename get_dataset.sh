# Modified from https://github.com/jadianes/spark-movie-lens/blob/master/download_dataset.sh

#!/bin/bash

# Check that the system has all the needed packages
hash wget 2>/dev/null || { echo >&2 "Wget required.  Aborting."; exit 1; }
hash unzip 2>/dev/null || { echo >&2 "unzip required.  Aborting."; exit 1; }

# Downloading the data files, extracting and moving them to the desired location
wget http://files.grouplens.org/datasets/movielens/ml-latest.zip
unzip -o "ml-latest.zip"
DESTINATION=~/data/
[ -d $DESTINATION ] && echo "Directory already exists." || mkdir -p $DESTINATION
echo "Moving file to ~/data/"
mv ml-latest $DESTINATION
echo "Cleaning up..."
rm "ml-latest.zip"

# Splitting the data into training and testing data
ORIGINAL=`pwd`
cd $BASH_SOURCE
echo Splitting the data into training and testing files...
python split_data.py
echo Done!
cd $ORIGINAL
