import numpy as np
import tensorflow as tf

# read data and build vocabulary
data_raw = open('./data/preprocessing/iphone_train_vntokenizer','r')

words = []
raw_sentences = []

for line in data_raw:
    for word in line.split():
        if word != '.' and len(word) > 1:
            words.append(word)

    for sentence in line.split('.'):
        raw_sentences.append(sentence)

data_raw.close()

words = set(words)
vocab_size = len(words)
print('vocabulary size = '+str(vocab_size))


# create a dictionary which translates words to integers and integers to words
word2int = {}
int2word = {}

for i,word in enumerate(words):
    word2int[word] = i
    int2word[i] = word

del words

sentences = []
for sentence in raw_sentences:
    sentences.append(sentence.split())


# generate training data (words)
data = []

WINDOW_SIZE = 2

for sentence in sentences:
    for word_index, word in enumerate(sentence):
        for nb_word in sentence[max(word_index-WINDOW_SIZE,0) :
                        min(word_index+WINDOW_SIZE,len(sentence)+1)]:
            if nb_word != word:
                data.append([word,nb_word])

del sentences

# convert word to one-hot vector
def to_one_hot(data_point_index, vocab_size):
    vec = np.zeros(vocab_size)
    vec[data_point_index] = 1
    return vec

# train data and its label (one-hot vector)
x_train = []
y_train = []

for data_word in data:
    if (data_word[0] in word2int) and (data_word[1] in word2int):
        x_train.append(to_one_hot(word2int[data_word[0]],vocab_size))
        y_train.append(to_one_hot(word2int[data_word[1]],vocab_size))

# convert to numpy array
x_train = np.asarray(x_train)
y_train = np.asarray(y_train)

print(x_train.shape, y_train.shape)

# make tensorflow model
x = tf.placeholder(dtype=tf.float32,shape=(None,vocab_size))
y_ = tf.placeholder(dtype=tf.float32,shape=(None,vocab_size))

EMBEDDING_DIM = 100
W1 = tf.Variable(tf.truncated_normal([vocab_size,EMBEDDING_DIM],stddev=0.01, mean=0.0))
b1 = tf.Variable(tf.truncated_normal([EMBEDDING_DIM],stddev=0.01, mean=0.0))

hidden_presentation = tf.add(tf.matmul(x,W1),b1)

W2 = tf.Variable(tf.random_normal([EMBEDDING_DIM,vocab_size],stddev=0.01, mean=0.0))
b2 = tf.Variable(tf.random_normal([vocab_size],stddev=0.01, mean=0.0))

prediction = tf.nn.softmax(tf.add(tf.matmul(hidden_presentation,W2),b2))

sess = tf.Session()

init = tf.global_variables_initializer()

sess.run(init)

loss = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(prediction), reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

n_epochs = 100
n_train_samples = len(x_train)
batch_size = 256

for epoch in range(n_epochs):

    perm = np.random.permutation(len(x_train))
    ep_loss = 0.0
    for i in range(0, n_train_samples , batch_size):

        x_batch = x_train[perm[i:i+batch_size]]
        y_batch = y_train[perm[i:i+batch_size]]

        _, batch_loss = sess.run([train_step, loss], feed_dict={x:x_batch, y_: y_batch})

        ep_loss += batch_loss*len(x_batch)

    print("epoch "+str(epoch)+"/"+str(n_epochs)+"    loss = "+str(ep_loss/n_train_samples))

vectors = sess.run(W1+b1)