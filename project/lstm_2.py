import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import pickle
import os

EMBEDDING_DIM = 300
nb_classes = 4

learning_rate = 0.1
nb_epoch = 100
batch_size = 64

n_input = 64
#n_steps = 64
n_hidden = 128

logs_path = './log/tensorboard'
writer = tf.summary.FileWriter(logs_path)

word_vecs = dict()

def build_dict():
    print("Build dictionary ..... ")
    f_word2vec = open('./data/word2vec/wiki.vi.vec', 'rb')

    for line in f_word2vec:
        words = line.split()

        vec = np.array(words[1:])
        vec = vec.astype(np.float)
        word_vecs[words[0]] = vec

    f_word2vec.close()
    print("Done!!")

build_dict()

def count_lines(file_name):
    f = open(file_name,'r')
    cnt = 0
    for line in f :
        cnt = cnt+1

    return cnt

#f_word2int = open('./data/word2vec/word2int.pkl','rb')
#word2int = pickle.load(f_word2int)
#word_vecs = np.load('./data/word2vec/vectors.npy')
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

                #if word in word2int:
                #    train_input_i.append(word_vecs[word2int[word]])
                if word in word_vecs:
                    train_input_i.append(word_vecs[word])
                else:
                    train_input_i.append(np.random.normal(0,0.1,EMBEDDING_DIM))

            if len(train_input_i) < n_input:
                for i in range(len(train_input_i),n_input):
                    train_input_i.append(np.random.normal(0, 0.1, EMBEDDING_DIM))
            else:
                if len(train_input_i) > n_input:
                    train_input_i = train_input_i[0:n_input]

            i_l = i_l+1
            train_inputs.append(train_input_i)

        i_f = i_f +1

    f.close()
    return np.array(train_inputs), train_labels


x = tf.placeholder("float", [None, n_input, EMBEDDING_DIM])
y = tf.placeholder("float", [None, nb_classes])

# RNN output node weights and biases
weights = {
    'out': tf.Variable(tf.random_normal([n_hidden, nb_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([nb_classes]))
}

def RNN(x, weights, biases):

    x = tf.transpose(x, [1,0,2])

    x = tf.reshape(x, [-1, EMBEDDING_DIM])

    x = tf.split(x,n_input,0)

    rnn_cell = rnn.MultiRNNCell([rnn.BasicLSTMCell(n_hidden),rnn.BasicLSTMCell(n_hidden), rnn.BasicLSTMCell(n_hidden)])

    outputs, states = rnn.static_rnn(rnn_cell, x, dtype=tf.float32)

    return tf.matmul(outputs[-1], weights['out']) + biases['out']

def BiRNN(x, weights, biases):
    x = tf.transpose(x, [1, 0, 2])

    x = tf.reshape(x, [-1, EMBEDDING_DIM])

    x = tf.split(x, n_input, 0)

    lstm_fw_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, forget_bias=0.8)
    lstm_bw_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, forget_bias=0.8)

    outputs, _, _ = tf.contrib.rnn.static_bidirectional_rnn(lstm_fw_cell,lstm_bw_cell,x,dtype='float32')

    output_fw, output_bw = tf.split(outputs, [n_hidden, n_hidden], 2)

    return tf.matmul(output_fw[-1]+output_bw[-1],weights['out']) + biases['out']


pred = BiRNN(x, weights, biases)

# Loss and optimizer
global_step = tf.contrib.framework.get_or_create_global_step()
learning_rate = tf.train.exponential_decay(0.1, global_step, 1000, 0.65, staircase=False)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost,global_step)

# Model evaluation
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()

saver = tf.train.Saver()
save_dir = 'save/lstm'
save_path = os.path.join(save_dir,'best_validation')

with tf.Session() as sess:
    sess.run(init)

    nb_lines = count_lines('./data/SA_IT4868_20171_data/iphone_train')
    nb_lines_dev = count_lines('./data/SA_IT4868_20171_data/iphone_dev')

    best_acc = 0.0

    writer.add_graph(sess.graph)

    for epoch in range(nb_epoch):
        print("Epoch "+str(epoch)+"/"+str(nb_epoch))
        perm = np.random.permutation(nb_lines)

        train_loss = []
        batch_iter = 0
        nb_batch = int(nb_lines/batch_size)

        for i in range(0,nb_lines,batch_size):
            lines = perm[i: i+batch_size]

            train_batch, train_batch_label = generate_data('./data/SA_IT4868_20171_data/iphone_train',lines)
            #print ("shape train_batch "+str(train_batch))

            _, batch_loss = sess.run([optimizer,cost],feed_dict={x: train_batch, y: train_batch_label})

            #print("batch_loss = "+str(batch_loss))
            train_loss.append(batch_loss)

        train_loss = np.average(train_loss)

        #print('train_loss: '+str(train_loss))

        dev_loss = []
        dev_acc = []
        for i in range(0,nb_lines_dev,batch_size):
            lines = np.array(range(i,min(i+batch_size, nb_lines_dev)))

            dev_batch , dev_batch_label = generate_data('./data/SA_IT4868_20171_data/iphone_dev',lines)

            dev_batch_loss , dev_batch_acc = sess.run([cost,accuracy],feed_dict={x:dev_batch,y:dev_batch_label})

            dev_loss.append(dev_batch_loss)
            dev_acc.append(dev_batch_acc)

        dev_loss = np.average(dev_loss)
        dev_acc = np.average(dev_acc)

        #print('dev_loss: '+str(dev_loss))
        print('train_loss = '+str(train_loss)+'     dev_loss = '+str(dev_loss)+'     dev_acc = '+str(dev_acc))

        if dev_acc > best_acc:
            best_acc = dev_acc
            saver.save(sess=sess, save_path=save_path)
