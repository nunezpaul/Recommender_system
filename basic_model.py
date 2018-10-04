import os
import uuid

import tensorflow as tf
import keras as k


class DataConfigBase(object):
    def __init__(self):
        self.iter_init = None
        self.next_element = None
        self.home = os.path.expanduser("~")

        # params for parsing csv
        self.header_lines = 1
        self.delim = ','
        self.cols_types = [
            tf.constant([], dtype=tf.int32),
            tf.constant([], dtype=tf.int32),
            tf.constant([], dtype=tf.float32),
            tf.constant([], dtype=tf.float32),
        ]

    def create_dataset(self):
        batch_size = self.batch_size
        dataset = tf.data.TextLineDataset(filenames=self.filename).skip(self.header_lines)
        dataset = dataset.map(self.parse_csv)
        dataset = dataset.repeat().shuffle(buffer_size=batch_size * 10)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(batch_size * 10)
        return dataset

    def parse_csv(self, line):
        columns = tf.decode_csv(line, record_defaults=self.cols_types, field_delim=self.delim)
        return columns


class TrainDataConfig(DataConfigBase):
    def __init__(self, batch_size=64):
        super(TrainDataConfig, self).__init__()
        self.batch_size = batch_size
        self.filename = '{home}/data/ml-latest/test.csv'.format(home=self.home)
        self.dataset = self.create_dataset()


class TestDataConfig(DataConfigBase):
    def __init__(self, batch_size=1024):
        super(TestDataConfig, self).__init__()
        self.batch_size = batch_size
        self.filename = '{home}/data/ml-latest/test.csv'.format(home=self.home)
        self.dataset = self.create_dataset()

class DataConfig(object):
    def __init__(self):
        self.configs = {'test': TestDataConfig(), 'train': TrainDataConfig()}
        self.iter_init = None
        self.next_element = None
        self.create_data_init()

    def create_data_init(self):
        keys = ['train', 'test']

        # Collect tf.datasets into a dict
        dataset = dict(zip(keys, [self.configs[key].dataset for key in keys]))

        # Creating iterator, data inits and next_element
        iterator = tf.data.Iterator.from_structure(dataset['train'].output_types, dataset['train'].output_shapes)
        self.iter_init = dict(zip(keys, [iterator.make_initializer(dataset[key], name=key) for key in keys]))
        self.next_element = iterator.get_next()



class ModelParams(DataConfig):
    def __init__(self):
        super(ModelParams, self).__init__()
        self.embed_dim = 256
        self.name = 'matrix_factorization'
        self.id = uuid.uuid4()
        self.model_dir = os.path.abspath(__file__).split(os.path.basename(__file__))[0]
        self.is_training = tf.placeholder_with_default(True, shape=())
        self.vocab_sizes = {}
        self.embeds = {}
        for key in ['user', 'movie']:
            with open(self.configs['test'].filename.replace('test.csv', '{k}_max.csv'.format(k=key)), 'r') as f:
                max_vocab = int(f.readline())
                self.vocab_sizes[key] = max_vocab
                self.embeds[key] = k.layers.Embedding(input_dim=max_vocab, output_dim=self.embed_dim)

    def embed(self, user, movie, check_shapes=True):
        # Get user and movie embeddings. Dot product is the score
        user_embed = self.embeds['user'](user)
        movie_embed = self.embeds['movie'](movie)
        pred_rating = tf.reduce_sum(tf.multiply(user_embed, movie_embed), axis=-1)

        # Place holder for future iterations of MF model
        intermediates = {}

        # Check shapes
        if check_shapes:
            assert user_embed.shape[1:] == movie_embed.shape[1:] == self.embed_dim
            assert pred_rating.shape[1:] == ()

        return pred_rating, intermediates


class TrainLoss(object):
    def __init__(self):
        self.model_params = ModelParams()

    def eval(self):
        metrics = {}

        # Loss will be on the negative log likelihood that the img embed belongs to the correct class
        user, movie, labels, _ = self.model_params.next_element
        predictions, _ = self.model_params.embed(user, movie)

        # Determine the mean squared error between pred and actual rating
        metrics['Mean_squared_error'] = tf.losses.mean_squared_error(labels=labels, predictions=predictions)

        # Check that shapes are as expected
        assert predictions.shape[1:] == labels.shape[1:] == ()

        return metrics, predictions, labels


class TrainRun(object):
    def __init__(self, lr=0.01):
        self.train_loss = TrainLoss()
        self.writer = {}
        self.eval_metrics = self.train_loss.eval()
        self.metrics, self.pred, self.lbl = self.eval_metrics
        self.optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        self.train_op = self.optimizer.minimize(self.metrics['Mean_squared_error'])
        self.step = 0

    def create_writers(self):
        phases = ['train', 'test']
        base_dir = self.train_loss.model_params.model_dir
        id = self.train_loss.model_params.id
        name = self.train_loss.model_params.name
        for phase in phases:
            tensorboard_dir = '{base_dir}.{name}/{id}/{phase}'.format(base_dir=base_dir, name=name, id=id, phase=phase)
            self.writer[phase] = tf.summary.FileWriter(tensorboard_dir, tf.get_default_graph())

    def initialize(self, sess):
        self.count_number_trainable_parameteres()
        self.create_writers()
        init_op = [tf.report_uninitialized_variables(),
                   tf.global_variables_initializer(),
                   self.train_loss.model_params.iter_init['train']]
        init_vals, _, _ = sess.run(init_op)
        print('Initializing Values: \n{init_vals}'.format(init_vals=init_vals))
        print('Finished Initialization.')

    def train(self, sess):
        for step in range(60 * 10**3):
            _ = sess.run([self.train_op])
            self.step += 1
            if step % 10**1 == 0:
                print('Evaluating metrics...')
                self.report_metrics(sess)

    def report_metrics(self, sess):
        # Evaluate using training data here and switch to testing data
        train_store, _ = sess.run(
            [self.eval_metrics, self.train_loss.model_params.iter_init['test']],
        feed_dict={self.train_loss.model_params.is_training: False})
        train_metrics, train_pred, train_lbl = train_store

        # Evaluate using testing data and switch to training data
        test_store, _ = sess.run(
            [self.eval_metrics, self.train_loss.model_params.iter_init['train']],
            feed_dict={self.train_loss.model_params.is_training: False})
        test_metrics, test_pred, test_lbl = test_store

        # Write metrics to tensorboard
        for tag in self.metrics.keys():
            self.tensorboard_logger(self.writer['test'], tag=tag, value=test_metrics[tag])
            self.tensorboard_logger(self.writer['train'], tag=tag, value=train_metrics[tag])
            print('Train {tag}: {train:.3f} \t\t\t Test {tag}: {test:.3f}'.format(
                tag=tag, train=train_metrics[tag], test=test_metrics[tag]))

        print('Train Predict {train} \t Test Predict {test}'.format(
            train=train_pred[:10], test=test_pred[:10]))
        print('Train Correct {train} \t Test Correct {test}'.format(
            train=train_lbl[:10], test=test_lbl[:10]))

    def tensorboard_logger(self, writer, tag, value):
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
        writer.add_summary(summary, self.step)
        writer.flush()

    def count_number_trainable_parameteres(self):
        total_parameters = 0
        for variable in tf.trainable_variables():
            # shape is an array of tf.Dimension
            shape = variable.get_shape()
            print(variable)
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim.value
            print(variable_parameters)
            total_parameters += variable_parameters
        print(total_parameters)



if __name__ == "__main__":
    sess = tf.Session()
    tr = TrainRun()
    tr.initialize(sess)
    tr.train(sess)
