import argparse

from os.path import expanduser
from random import random


def split_data(filename, test_frac, skip_count=False):
    # filenames for train and test data
    datafilenames = {}
    datafiles = {}
    datatypes = ['train', 'test']

    # Placeholder to determine max vocab size
    vocab_size = {'user': 0, 'movie': 0}

    # Create filename, mkdir if needed and open train and test files to be written
    for datatype in datatypes:
        datafilenames[datatype] = filename.replace('ratings', '{dt}'.format(dt=datatype))
        datafiles[datatype] = open(datafilenames[datatype], 'w')

    # Read each line after the header of ratings.csv
    with open(filename, 'r') as f:
        header = f.readline()
        for datatype in datatypes:
            datafiles[datatype].write(header)

        # Split the data line into training or testing
        for line in f:
            if random() < test_frac:
                datafiles['test'].write(line)
            else:
                datafiles['train'].write(line)

            if not(skip_count):
                arr = dict(zip(['user', 'movie', 'rating', 'time'], line.split(',')))
                for key in vocab_size.keys():
                    vocab_size[key] = max(vocab_size[key], int(arr[key]))

    # Close the files
    for datatype in datatypes:
        datafiles[datatype].close()

    # Store the user and movie max vocab
    if not(skip_count):
        for key in vocab_size.keys():
            with open(filename.replace('ratings.csv', '{key}_max.csv'.format(key=key)), 'w') as f:
                f.write(str(vocab_size[key]))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process data file into a train and test file.')
    parser.add_argument('--test_frac', help='Set the frac of the data to be set aside for testing.',
                        default=0.05, type=float)
    parser.add_argument('--skip_count',
                        help="""
                        Add this flag to skip determining max value for users and movies. Enabling this flag speeds 
                        up the file splitting time considerably. You can do this step manually or run without the flag
                        once to get the necessary info.
                        """
                        '',
                        action='store_true')
    args = parser.parse_args()

    filename = '{home}/data/ml-latest/ratings.csv'.format(home=expanduser("~"))
    split_data(filename=filename, test_frac=args.test_frac, skip_count=args.skip_count)