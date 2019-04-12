
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from data_preprocess import get_datasets

INPUT_SIZE = 7
OUTPUT_SIZE = 1
TIME_STEP = 15
HIDDEN_UNIT = 10
BATCH_SIZE = 64
lr = 0.0006
EPOSIDE = 1001

class Lstm:
    def __init__(self, input_size, output_size, time_step, hidden_unit, lr):
        self.input_size = input_size
        self.output_size = output_size
        self.time_step = time_step
        self.hidden_unit = hidden_unit
        self.lr = lr
        self.build_model()
        
    
    def build_model(self):
        self.tf_x = tf.placeholder(tf.float32, [None, self.time_step, self.input_size], name='tf_x')
        self.tf_y = tf.placeholder(tf.float32, [None, self.time_step, self.output_size], name='t_y')
        tf_batch_size = tf.shape(self.tf_x)[0] 
        self.time_step = tf.shape(self.tf_x)[1]
        
        # define the weight and bias of input and output
        weights = {
            'in': tf.Variable(tf.random_normal([self.input_size, self.hidden_unit])),
            'out': tf.Variable(tf.random_normal([self.hidden_unit, self.output_size]))
        }

        bias = {
            'in': tf.Variable(tf.constant(0.1, shape = [self.hidden_unit,])),
            'out':tf.Variable(tf.constant(0.1, shape = [self.output_size,]))
        }

        # calculate the input unit
        tf_x = tf.reshape(self.tf_x, [-1, self.input_size])
        lstm_input = tf.matmul(tf_x, weights['in']) + bias['in']
        lstm_input = tf.reshape(lstm_input, [-1, self.time_step, self.hidden_unit])

        # define the lstm cell and initialize the units of the lstm model.
        lstm_cell = tf.nn.rnn_cell.LSTMCell(self.hidden_unit)
        init_state = lstm_cell.zero_state(tf_batch_size, dtype = tf.float32)
        lstm_output, final_state = tf.nn.dynamic_rnn(
            cell = lstm_cell, 
            inputs = lstm_input, 
            initial_state = init_state, 
            dtype = tf.float32
            )
        
        # define the output units
        lstm_output = tf.reshape(lstm_output, [-1, self.hidden_unit])
        self.predict = tf.matmul(lstm_output, weights['out']) + bias['out']

        # define the loss function
        self.loss = tf.reduce_mean(tf.square(tf.reshape(self.predict, [-1]) - tf.reshape(self.tf_y, [-1])))
        self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    def train_model(self, train_x, train_y):

        saver = tf.train.Saver(tf.global_variables())

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            # module_file = tf.train.latest_checkpoint('./train')
            # saver.restore(sess, module_file)

            for i in range(EPOSIDE):
                for j in range(len(train_x)):
                    _, loss = sess.run(
                        [self.optimizer, self.loss], 
                        feed_dict={self.tf_x:train_x[j], self.tf_y:train_y[j]})

                print("Eposide: {} loss:{}".format(i, loss))
                if i % 200 == 0:
                    saver.save(sess, "./train/lstm.model", global_step = i)
                    print("Save Model")
    
    def predict_model(self, test_x, test_y):
        
        saver = tf.train.Saver(tf.global_variables())
        with tf.Session() as sess:
            module_file = tf.train.latest_checkpoint('./train')
            saver.restore(sess, module_file)

            test_predict = []
            for i in range(len(test_x)):
                predict = sess.run([self.predict], feed_dict={self.tf_x:test_x[i]})
                predict = np.reshape(predict, [-1])
                test_predict.extend(predict)
            
            # calculate the accuracy
            test_y = np.reshape(test_y, [-1])
            rsme = np.sqrt(np.mean(np.square(np.abs(test_predict - test_y))))
            print(rsme)

            # plot the result
            plt.figure()
            plt.plot(list(range(len(test_predict))), test_predict, color = 'r')
            plt.plot(list(range(len(test_y))), test_y, color = 'b')
            plt.show()

if __name__ == "__main__":
    # stock predict
    model = Lstm(INPUT_SIZE, OUTPUT_SIZE, TIME_STEP, HIDDEN_UNIT, lr)
    train_x, train_y, test_x, test_y = get_datasets(BATCH_SIZE, TIME_STEP, INPUT_SIZE)

    model.train_model(train_x, train_y)
    model.predict_model(test_x, test_y)

