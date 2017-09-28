import  re

def pre_processing():
    f = open('./data/preprocessing/iphone_train_vntokenizer','r')
    f_p = open('./data/preprocessing/iphone_train_word2vec','w')

    for line in f:
        new_line = re.sub(r'\d###',"",line)
        new_line = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
                          ,'url_link',new_line)
        new_line = re.sub('www[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
                          ,'url_link',new_line)
        f_p.write(new_line.lower())

    f.close()
    f_p.close()

pre_processing()