from __future__ import unicode_literals
import tensorflow as tf
import numpy as np

def vectorize(dataframe):
    WORDS_SIZE=10000
    INPUT_SIZE=500

    x_all = dataframe.iloc[:, 0]
    tokenizer = tf.keras.preprocessing.text.Tokenizer(char_level=False)
    tokenizer.fit_on_texts(list(x_all))
    del(x_all)
    print('Number of indexed word(token) count: ',len(tokenizer.word_counts))
    # Reducing to top N words
    print("Reducing to top N words to ", WORDS_SIZE)
    tokenizer.num_words = WORDS_SIZE

    print("Top 10 words: ")    
    print(sorted(tokenizer.word_counts.items(), key=lambda x:x[1], reverse=True)[0:10])

     ## Tokenizing data and create matrix
    print("Tokenizing data and create matrix")
    list_tokenized_train = tokenizer.texts_to_sequences(dataframe.iloc[:,0])
    print("Padding sequences to the same length.")
    x_train = tf.keras.preprocessing.sequence.pad_sequences(list_tokenized_train, 
                                      maxlen=INPUT_SIZE,
                                      padding='post')
    x_train = x_train.astype(np.int64)
    print(x_train)
    return x_train;