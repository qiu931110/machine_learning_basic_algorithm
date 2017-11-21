import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


mnist = input_data.read_data_sets('MNIST_data/',one_hot=True)

input_num = 28
sequence_num = 28
lstm_size = 100
class_num = 10
batchsize = 50
batch_num = mnist.train.num_examples // batchsize

#######

x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

#######
weights = tf.Variable(tf.truncated_normal([lstm_size, class_num], stddev = 0.1))
biases = tf.Variable(tf.constant(0.1, shape=[class_num]))

####

def RNN(X, weights, biases):
    input = tf.reshape(X, [-1, sequence_num, input_num])
    lstm_cell = tf.contrib.rnn.core_rnn_cell.BasicLSTMCell(lstm_size)
    output, final_state = tf.nn.dynamic_rnn(lstm_cell,input,dtype= tf.float32)
    predict = tf.nn.softmax(tf.matmul(final_state[1],weights) + biases)

    return predict

prediction = RNN(x, weights, biases)

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits= prediction, labels= y))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(prediction,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(6):
        for batch in range(batch_num):
            batch_xs, batch_ys = mnist.train.next_batch(batchsize)
            sess.run(train_step,feed_dict={x:batch_xs, y : batch_ys})
        acc = sess.run(accuracy,feed_dict={x:mnist.test.images, y : mnist.test.labels})
        print('Iter' + str(epoch) + ', Testing Acc = ' + str(acc))



###########
'''
C:\Software\PYthon35\python.exe C:/Software/pycharm/LSTM/class7_rnn_orignal/rnn.py
Extracting MNIST_data/train-images-idx3-ubyte.gz
Extracting MNIST_data/train-labels-idx1-ubyte.gz
Extracting MNIST_data/t10k-images-idx3-ubyte.gz
Extracting MNIST_data/t10k-labels-idx1-ubyte.gz

Iter0, Testing Acc = 0.743
Iter1, Testing Acc = 0.835
Iter2, Testing Acc = 0.8966
Iter3, Testing Acc = 0.9106
Iter4, Testing Acc = 0.9178
Iter5, Testing Acc = 0.9296

Process finished with exit code 0
'''