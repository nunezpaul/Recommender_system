from random import random


def split_data(filename, test_percent=0.1):
    # Train and test files to be written
    tr = open(filename.replace('ratings', 'train'), 'w')
    te = open(filename.replace('ratings', 'test'), 'w')

    # Read each line after the header of ratings.csv
    with open(filename, 'r') as f:
        header = f.readline()
        tr.write(header)
        te.write(header)

        # Randomly split the data into training or testing
        for line in f:
            if random() < test_percent:
                te.write(line)
            else:
                tr.write(line)
    tr.close()
    te.close()

if __name__ == '__main__':

    filename = '/Users/pwork/data/ml-latest/ratings.csv'
    split_data(filename=filename)