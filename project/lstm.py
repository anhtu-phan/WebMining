import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import pickle

EMBEDDING_DIM = 100
num_nodes = 64
nb_classes = 4
num_unrollings = 50

learning_rate = 0.1
nb_epoch = 1000
batch_size = 64

graph = tf.Graph()

def count_lines(file_name):
    f = open(file_name,'r')
    cnt = 0
    for line in f :
        cnt = cnt+1

    return cnt

f_word2int = open('./data/word2vec/word2int.pkl','rb')
word2int = pickle.load(f_word2int)
word_vecs = np.load('./data/word2vec/vectors.npy')
def generate_data(file_name, lines):

    train_inputs = []
    train_labels = []

    # read sentence
    f = open(file_name, 'r')

    i_l = 0
    i_f = 0
    lines.sort()
    for line in f:
        if i_l >= len(lines) :
            break

        if i_f == lines[i_l] :
            train_input_i = []
            test = 0
            for word in line.split():
                if test == 0:
                    train_label_i = np.zeros(nb_classes)
                    train_label_i[int(word)-1] = 1.0
                    train_labels.append(train_label_i)
                    test = test+1

                if word in word2int:
                    train_input_i.append(word_vecs[word2int[word]])
                else:
                    train_input_i.append(np.random.normal(0,0.1,EMBEDDING_DIM))

            if len(train_input_i) < num_unrollings:
                for i in range(len(train_input_i),num_unrollings):
                    train_input_i.append(np.random.normal(0, 0.1, EMBEDDING_DIM))
            else:
                if len(train_input_i) > num_unrollings:
                    train_input_i = train_input_i[0:num_unrollings]

            i_l = i_l+1
            train_inputs.append(train_input_i)

        i_f = i_f +1

    f.close()
    return np.array(train_inputs), train_labels


with graph.as_default():
    # Parameters of forget gate: input, previous output, bias
    fx = tf.Variable(tf.truncated_normal([EMBEDDING_DIM, num_nodes], stddev=1.e-4, mean=0.0))
    fm = tf.Variable(tf.truncated_normal([num_nodes, num_nodes], stddev=1.e-4, mean=0.0))
    fb = tf.Variable(tf.zeros([1, num_nodes]))

    # Parameters of input gate: input, previous output, and bias
    ix = tf.Variable(tf.truncated_normal([EMBEDDING_DIM, num_nodes], stddev=1.e-4, mean=0.0))
    im = tf.Variable(tf.truncated_normal([num_nodes, num_nodes], stddev=1.e-4, mean=0.0))
    ib = tf.Variable(tf.zeros([1, num_nodes]))

    # Parameters of memory cell: input, previous output, and bias
    cx = tf.Variable(tf.truncated_normal([EMBEDDING_DIM, num_nodes], stddev=1.e-4, mean=0.0))
    cm = tf.Variable(tf.truncated_normal([num_nodes, num_nodes], stddev=1.e-4, mean=0.0))
    cb = tf.Variable(tf.zeros([1, num_nodes]))

    # Parameters of output gate: input, state, and bias
    ox = tf.Variable(tf.truncated_normal([EMBEDDING_DIM, num_nodes], stddev=1.e-4, mean=0.0))
    om = tf.Variable(tf.truncated_normal([num_nodes, num_nodes], stddev=1.e-4, mean=0.0))
    ob = tf.Variable(tf.zeros([1, num_nodes]))

    # Variables saving state across unrollings.
    saved_output = tf.Variable(tf.zeros([batch_size, num_nodes]), trainable=False)
    saved_state = tf.Variable(tf.zeros([batch_size, num_nodes]), trainable=False)

    # Classifier weights and bias
    w = tf.Variable(tf.truncated_normal([num_nodes, nb_classes], stddev=1.e-4, mean=0.0))
    b = tf.Variable(tf.zeros([nb_classes]))


    # Definition of the cell computation
    def lstm_cell(i, o, state):
        forget_state = tf.sigmoid(tf.matmul(i, fx) + tf.matmul(o, fm) + fb)

        input_state = tf.sigmoid(tf.matmul(i, ix) + tf.matmul(o, im) + ib)
        c_t = tf.tanh(tf.matmul(i, cx) + tf.matmul(o, cm) + cb)
        update = tf.matmul(input_state, c_t)

        state = tf.matmul(forget_state, state) + update

        output_state = tf.sigmoid(tf.matmul(i, ox) + tf.matmul(o, om) + ob)
        output = tf.matmul(output_state, tf.tanh(state))

        return output, state

    # Input data
    train_inputs = []
    for _ in range(num_unrollings):
        train_inputs.append(tf.placeholder(tf.float32, shape=[None,EMBEDDING_DIM]))

    train_labels = tf.placeholder(tf.float32, shape=[None,nb_classes])

    output = saved_output
    state = saved_state

    for i in train_inputs:
        output, state = lstm_cell(i, output, state)

    logits = tf.matmul(output, w) + b
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=train_labels))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)

    init = tf.global_variables_initializer()

with tf.Session(graph=graph) as sess:
    sess.run(init)

    nb_lines = count_lines('./data/preprocessing/iphone_train')
    for epoch in range(nb_epoch):
        perm = np.random.permutation(nb_lines)

        train_loss = []
        batch_iter = 0
        nb_batch = int(nb_lines/batch_size)
        for i in range(0,nb_lines,batch_size):
            lines = perm[i: i+batch_size]

            if len(lines) < batch_size:
                break

            train_batch, train_batch_label = generate_data('./data/preprocessing/iphone_train',lines)

            feed_dict = dict()
            for i in range(num_unrollings):
                feed_dict[train_inputs[i]] = train_batch[:,i,:]

            feed_dict[train_labels] = train_batch_label

            _, batch_loss = sess.run([optimizer,loss],feed_dict=feed_dict)
            print("batch_loss = "+str(batch_loss))
            train_loss.append(batch_loss)

        train_loss = np.average(train_loss)

        print('train_loss: '+str(train_loss))