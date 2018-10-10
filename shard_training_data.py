import argparse

from os.path import expanduser
from random import randint

def shard_data(filename, num_shards):
    datafilenames = {}
    datafiles = {}
    shards = [str(i) for i in range(num_shards)]

    # Create filenames and open shard files to be written
    for shard in shards:
        datafilenames[shard] = filename.replace('train', 'train_{shard}'.format(shard=shard))
        datafiles[shard] = open(datafilenames[shard], 'w')

    # Read each line after the header of training.csv
    with open(filename, 'r') as f:
        header = f.readline()
        for shard in shards:
            datafiles[shard].write(header)

        # Split the data line into training or testing
        for line in f:
            which_shard = str(randint(0, num_shards-1))
            datafiles[which_shard].write(line)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Shard training data in N csv files.')
    parser.add_argument('--num_shards', default=10, type=int,
                        help='Set the number of shards the training data will be broken up into.')
    parser.add_argument('--filename', type=str, default='data/ml-latest/train.csv',
                        help='file location for train.csv')
    args = parser.parse_args()

    shard_data(filename=args.filename, num_shards=args.num_shards)