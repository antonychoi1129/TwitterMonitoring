import numpy as np
import tensorflow as tf
from tensorflow import keras

# load all docs in a directory
def data_pre_process(filename, labels, class_index):
    documents = []
    file = open(filename, 'r')
    # Using readlines() 
    Lines = file.readlines() 
    file.close()
    # Strips the newline character 
    for line in Lines: 
        # print(line)
        documents.append(line.replace("\n", ""))
        labels.append(class_index)
    print(len(documents))
    return documents

samples = []
labels = []
class_names = ['negative','neutral','positive']
samples.extend(data_pre_process('/content/drive/MyDrive/labelled_data/neg.txt', labels, 0))
samples.extend(data_pre_process('/content/drive/MyDrive/labelled_data/neu.txt', labels, 1))
samples.extend(data_pre_process('/content/drive/MyDrive/labelled_data/pos.txt', labels, 2))

# samples.extend(data_pre_process('/content/drive/MyDrive/labelled_data/test_data/neg.txt', labels, 0))
# samples.extend(data_pre_process('/content/drive/MyDrive/labelled_data/test_data/neu.txt', labels, 1))
# samples.extend(data_pre_process('/content/drive/MyDrive/labelled_data/test_data/pos.txt', labels, 2))




print("Classes:", class_names)
print("Number of samples:", len(samples))
# print(samples)
# print(labels[:20])
# print(labels[20:])

# Shuffle the data
seed = 1337
rng = np.random.RandomState(seed)
rng.shuffle(samples)
rng = np.random.RandomState(seed)
rng.shuffle(labels)

# Extract a training & validation split
validation_split = 0.2
num_validation_samples = int(validation_split * len(samples))
train_samples = samples[:-num_validation_samples]
val_samples = samples[-num_validation_samples:]
train_labels = labels[:-num_validation_samples]
val_labels = labels[-num_validation_samples:]

from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

vectorizer = TextVectorization(max_tokens=20000, output_sequence_length=200)
text_ds = tf.data.Dataset.from_tensor_slices(train_samples).batch(128)
vectorizer.adapt(text_ds)

vectorizer.get_vocabulary()[:5]

output = vectorizer([["the cat sat on the mat"]])
output.numpy()[0, :6]

voc = vectorizer.get_vocabulary()
word_index = dict(zip(voc, range(len(voc))))

# print(word_index)
test = ["the"]
[word_index[w] for w in test]

path_to_glove_file = "/content/drive/MyDrive/glove/glove.6B.100d.txt"


embeddings_index = {}
with open(path_to_glove_file) as f:
    for line in f:
        word, coefs = line.split(maxsplit=1)
        coefs = np.fromstring(coefs, "f", sep=" ")
        embeddings_index[word] = coefs

print("Found %s word vectors." % len(embeddings_index))

num_tokens = len(voc) + 2
embedding_dim = 100
hits = 0
misses = 0

# Prepare embedding matrix
embedding_matrix = np.zeros((num_tokens, embedding_dim))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # Words not found in embedding index will be all-zeros.
        # This includes the representation for "padding" and "OOV"
        embedding_matrix[i] = embedding_vector
        hits += 1
    else:
        misses += 1
print("Converted %d words (%d misses)" % (hits, misses))

from tensorflow.keras.layers import Embedding

embedding_layer = Embedding(
    num_tokens,
    embedding_dim,
    embeddings_initializer=keras.initializers.Constant(embedding_matrix),
    trainable=False,
)

from tensorflow.keras import layers

int_sequences_input = keras.Input(shape=(None,), dtype="int64")
embedded_sequences = embedding_layer(int_sequences_input)
x = layers.Conv1D(128, 5, activation="relu")(embedded_sequences)
x = layers.MaxPooling1D(5)(x)
x = layers.Conv1D(128, 5, activation="relu")(x)
x = layers.MaxPooling1D(5)(x)
x = layers.Conv1D(128, 5, activation="relu")(x)
x = layers.GlobalMaxPooling1D()(x)
x = layers.Dense(128, activation="relu")(x)
x = layers.Dropout(0.5)(x)
preds = layers.Dense(len(class_names), activation="softmax")(x)
model = keras.Model(int_sequences_input, preds)
model.summary()

x_train = vectorizer(np.array([[s] for s in train_samples])).numpy()
x_val = vectorizer(np.array([[s] for s in val_samples])).numpy()

y_train = np.array(train_labels)
y_val = np.array(val_labels)

model.compile(
    loss="sparse_categorical_crossentropy", optimizer="rmsprop", metrics=["acc"]
)
model.fit(x_train, y_train, batch_size=128, epochs=20, validation_data=(x_val, y_val))

string_input = keras.Input(shape=(1,), dtype="string")
x = vectorizer(string_input)
preds = model(x)
end_to_end_model = keras.Model(string_input, preds)

probabilities = end_to_end_model.predict(
    # [["The beads are pretty but the box was broken and all the different sizes were mixed up when I got it. The box did not stand up to the shipping."]]
    [["perfect"]]
)
print(probabilities[0])
print(np.argmax(probabilities[0]))
class_names[np.argmax(probabilities[0])]
# class_names[np.argmax(probabilities[0])]

print(vectorizer([["this"]]))

import pickle
pickle.dump({'config': vectorizer.get_config(),
             'weights': vectorizer.get_weights()}
            , open('/content/drive/MyDrive/vectorizer.pkl', "wb"))

model.save('/content/drive/MyDrive/my_model')

# vec_model = keras.models.Sequential()
# vec_model.add(keras.Input(shape=(1,), dtype="string"))
# vec_model.add(vectorizer)
# vec_model.save('/content/drive/MyDrive/vec_model', save_format="tf")
