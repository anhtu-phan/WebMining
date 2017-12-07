import re
import os
import csv

def pre_processing():
    '''f = open('./data/SA_IT4868_20171_data/iphone_dev','r')
    f_p = open('./data/preprocessing/iphone_dev','w')

    for line in f:
        words = line.split()
        for word_index, word in enumerate(words):
            if 'http' in word or 'www' in word:
                words[word_index] = "urllink"

        new_line = ' '.join(words)

        f_p.write(new_line.lower())
        f_p.write('\n')

    f.close()
    f_p.close()

    os.chdir("./data/vn.hus.nlp.tokenizer-4.1.1-bin")
    os.system("sh vnTokenizer.sh -i ../preprocessing/iphone_dev -o ../preprocessing/iphone_dev_tokenizer")
    os.chdir("../..")
    '''

    #f = open('./data/preprocessing/iphone_dev_tokenizer','r')
    #f2 = open('./data/preprocessing/iphone_dev','w')
    f3 = open('./data/stop_words','r')
    stop_words = []
    for word in f3:
        stop_words.append(word)
    f3.close()

    f4w = open('./data/preprocessing/train', 'w')

    f4 = open('./data/TRAIN.csv','r')
    readCSV4 = csv.reader(f4, delimiter=',')
    for row in readCSV4:
        new_line = row[1].lower()
        label = row[0]
        new_line = re.sub('[\.,!-^@:?=#\[\]()/\d"]', '', new_line)
        words = new_line.split()
        for word_index, word in enumerate(words):
            if word in stop_words:
                words[word_index] = " "
            if len(word) < 2 or len(word) > 10:
                words[word_index] = " "

        new_line = " ".join(words)
        f4w.write(label+" "+new_line+"\n")
    f4.close()

    f5 = open('./data/TEST.csv','r')
    readCSV5 = csv.reader(f5,delimiter=',')
    for row in readCSV5:
        new_line = row[1].lower()
        label = row[0]
        new_line = re.sub('[\.,!-^@:?=#\[\]()/\d"]', '', new_line)
        words = new_line.split()
        for word_index, word in enumerate(words):
            if word in stop_words:
                words[word_index] = " "
            if len(word) < 2 or len(word) > 10:
                words[word_index] = " "
        new_line = " ".join(words)
        f4w.write(label+" "+new_line+"\n")

    f5.close()
    f4w.close()

pre_processing()