# -*- coding: utf-8 -*-


import sys, random, re, pickle
import numpy as np
import jiebazhc as jieba


# Length of extracted character sequences
MAXLEN = 10

def load_data(path, model_h5=None):
    content = open(path).read().lower()
    content = re.sub(r'[\n|#|0-9|：|（|）|“|”|《|》|‘|’]', '', content)
    text = [i for i in jieba.cut(content) if len(i)>0]
    print('Corpus length:', len(text))

    # We sample a new sequence every `step` characters
    step = 3

    # This holds our extracted sequences
    sentences = []

    # This holds the targets (the follow-up characters)
    next_chars = []

    for i in range(0, len(text) - MAXLEN, step):
        sentences.append(text[i: i + MAXLEN])
        next_chars.append(text[i + MAXLEN])
    print('Number of sequences:', len(sentences))

    # List of unique characters in the corpus
    chars = sorted(list(set(text)))
    print('Unique characters:', len(chars))
    # Dictionary mapping unique characters to their index in `chars`
    char_indices = dict((char, chars.index(char)) for char in chars)

    # Next, one-hot encode the characters into binary arrays.
    print('Vectorization...')
    x = np.zeros((len(sentences), MAXLEN, len(chars)), dtype=np.bool)
    y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
    for i, sentence in enumerate(sentences):
        for t, char in enumerate(sentence):
            x[i, t, char_indices[char]] = 1
        y[i, char_indices[next_chars[i]]] = 1

    # save 
    if model_h5:
        with open(model_h5+'.dat', 'wb') as f:
            pickle.dump((chars, char_indices), f)

    return x, y, chars


def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def train_model(model_h5, data_path):
    import keras
    from keras import layers

    # 装入数据
    x, y, chars = load_data(data_path, model_h5)

    # 建立网络
    model = keras.models.Sequential()
    model.add(layers.LSTM(128, input_shape=(MAXLEN, len(chars))))
    model.add(layers.Dense(len(chars), activation='softmax'))

    optimizer = keras.optimizers.RMSprop(lr=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)

    for epoch in range(1, 60):
        print('epoch', epoch)
        # Fit the model for 1 epoch on the available training data
        model.fit(x, y,
                  batch_size=128,
                  epochs=1)

    model.save(model_h5+'.h5')


def generate_txt(model_h5, text_seed):
    import keras

    model = keras.models.load_model(model_h5+'.h5')
    with open(model_h5+'.dat', 'rb') as f:
        chars, char_indices = pickle.load(f)

    generated_text = [i for i in jieba.cut(text_seed) if len(i)>0]

    for temperature in [0.2, 0.5, 1.0, 1.2]:
        print('------ temperature:', temperature)
        sys.stdout.write(''.join(generated_text))

        # We generate 400 characters
        for i in range(400):
            sampled = np.zeros((1, MAXLEN, len(chars)))
            for t, char in enumerate(generated_text):
                sampled[0, t, char_indices[char]] = 1.

            preds = model.predict(sampled, verbose=0)[0]
            next_index = sample(preds, temperature)
            next_char = chars[next_index]

            #generated_text += next_char
            generated_text.append(next_char)
            if len(generated_text)>MAXLEN:
                generated_text = generated_text[1:]

            sys.stdout.write(next_char)
            sys.stdout.flush()
        print()

if __name__ == '__main__':
    if len(sys.argv)<4:
        print("usage: python3 %s <train|gen> <model_h5> <train-data-path>|<text-seed>" % sys.argv[0])
        sys.exit(0)

    action = sys.argv[1]
    model_h5 = sys.argv[2]

    if action=='train':
        train_model(model_h5, sys.argv[3])   
    else:
        generate_txt(model_h5, sys.argv[3])
