csv_filename = filename = '/Users/pwork/data/ml-latest/ratings.csv'
header_lines = 1
delim = ','
batch_size = 10

dataset = tf.data.TextLineDataset(filenames=csv_filename).skip(header_lines)


def parse_csv(line):
    cols_types = [
                     tf.constant([], dtype=tf.int32),
                     tf.constant([], dtype=tf.int32),
                     tf.constant([], dtype=tf.float32),
                     tf.constant([], dtype=tf.float32),
                 ]
    columns = tf.decode_csv(line, record_defaults=cols_types, field_delim=delim)
    return columns

tld = dataset.map(parse_csv).batch(batch_size)

tld_next = tld.make_one_shot_iterator().get_next()

sess.run(tld_next)