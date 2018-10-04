# Modified from https://github.com/jadianes/spark-movie-lens/blob/master/download_dataset.sh

#!/bin/bash

hash wget 2>/dev/null || { echo >&2 "Wget required.  Aborting."; exit 1; }
hash unzip 2>/dev/null || { echo >&2 "unzip required.  Aborting."; exit 1; }

wget http://files.grouplens.org/datasets/movielens/ml-latest.zip
unzip -o "ml-latest.zip"
DESTINATION=~/data/
[ -d $DESTINATION ] && echo "Directory already exists." || mkdir -p $DESTINATION
echo "Moving file to ~/data/"
mv ml-latest $DESTINATION
echo "Cleaning up..."
rm "ml-latest.zip"
python split_data.py