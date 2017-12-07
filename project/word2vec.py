import numpy as np
import pickle
import csv
import tensorflow as tf
import os

# read data and build vocabulary
data_raw = open('./data/preprocessing/iphone_train','r')
data_raw2 = open('./data/preprocessing/train','r')

words = []
raw_sentences = []

for line in data_raw:
    line_words = line.split()
    for i in range(1,len(line_words)):
        words.append(line_words[i])

    sentence = " ".join(line_words[1:])
    raw_sentences.append(sentence)

data_raw.close()


for line in data_raw2:
    line_words = line.split()
    for i in range(1,len(line_words)):
        words.append(line_words[i])

    sentence = " ".join(line_words[1:])
    raw_sentences.append(sentence)

data_raw2.close()


'''
data_raw2 = open('./data/TRAIN.csv')
readCSV2 = csv.reader(data_raw2,delimiter=',')
for row in readCSV2:
    line_words = row[1].lower().split()
    for i in range(len(line_words)):
        words.append(line_words[i])

    raw_sentences.append(row[1].lower())

data_raw2.close()

data_raw3 = open('./data/TEST.csv')
readCSV3 = csv.reader(data_raw3,delimiter=',')
for row in readCSV3:
    line_words = row[1].lower().split()
    for i in range(len(line_words)):
        words.append(line_words[i])

    raw_sentences.append(row[1].lower())

data_raw3.close()
'''

words = set(words)

f_words = open("words.txt",'w')
for word in words:
    f_words.write("%s \n" % word)
f_words.close()

vocab_size = len(words)
print('vocabulary size = '+str(vocab_size))

'''
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

WINDOW_SIZE = 3
print('generate training data (words) ...')
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
b1 = tf.Variable(tf.zeros([EMBEDDING_DIM]))

hidden_presentation = tf.add(tf.matmul(x,W1),b1)

W2 = tf.Variable(tf.random_normal([EMBEDDING_DIM,vocab_size],stddev=0.01, mean=0.0))
b2 = tf.Variable(tf.zeros([vocab_size]))

prediction = tf.nn.softmax(tf.add(tf.matmul(hidden_presentation,W2),b2))

loss = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(prediction), reduction_indices=[1]))

global_step = tf.contrib.framework.get_or_create_global_step()
learning_rate = tf.train.exponential_decay(7.5, global_step, 10000, 0.65, staircase=False)
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step)

n_epochs = 1000
n_train_samples = len(x_train)
batch_size = 256

saver = tf.train.Saver()
save_dir = "save/word2vec"
save_path = os.path.join(save_dir,"best_model")

def train():

    with tf.Session() as sess:

        save_model_path = os.path.join(save_dir,"best_model.index")
        if not os.path.isfile(save_model_path):
            print ("Create new model")
            init = tf.global_variables_initializer()
            sess.run(init)
        else:
            print ("Load exist model")
            saver.restore(sess=sess, save_path=save_path)

        min_loss = 10

        for epoch in range(n_epochs):

            perm = np.random.permutation(len(x_train))
            ep_loss = 0.0
            print("learning_rate = "+str(learning_rate.eval()))
            for i in range(0, n_train_samples , batch_size):

                x_batch = x_train[perm[i:i+batch_size]]
                y_batch = y_train[perm[i:i+batch_size]]

                _, batch_loss = sess.run([train_step, loss], feed_dict={x:x_batch, y_: y_batch})

                ep_loss += batch_loss*len(x_batch)

            ep_loss = ep_loss/n_train_samples
            print("epoch "+str(epoch)+"/"+str(n_epochs)+"    loss = "+str(ep_loss))
            if(ep_loss < min_loss):
                #print("Save model")
                min_loss = ep_loss
                saver.save(sess=sess,save_path=save_path)

def get_result():
    # save result
    with tf.Session() as sess:
        saver.restore(sess=sess, save_path=save_path)
        vectors = sess.run(W1+b1)
        np.save('./data/word2vec/vectors.npy',vectors)
        f_word2int = open('./data/word2vec/word2int.pkl','wb')
        f_int2word = open('./data/word2vec/int2word.pkl','wb')
        pickle.dump(word2int,f_word2int,pickle.HIGHEST_PROTOCOL)
        pickle.dump(int2word,f_int2word,pickle.HIGHEST_PROTOCOL)
        f_int2word.close()
        f_word2int.close()


train()

'''