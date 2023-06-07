import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from datasets import load_dataset
from collections import Counter
from src.utils.conlleval import *


# ---------------- Classes ----------------


# --- defining a TransformerBlock layer
class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()
        self.att = keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        self.ffn = keras.Sequential(
            [
                keras.layers.Dense(ff_dim, activation="relu"),
                keras.layers.Dense(embed_dim),
            ]
        )
        self.layernorm1 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = keras.layers.Dropout(rate)
        self.dropout2 = keras.layers.Dropout(rate)

    def call(self, inputs, training=False):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super().__init__()
        self.token_emb = keras.layers.Embedding(
            input_dim=vocab_size, output_dim=embed_dim
        )
        self.pos_emb = keras.layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, inputs):
        maxlen = tf.shape(inputs)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        position_embeddings = self.pos_emb(positions)
        token_embeddings = self.token_emb(inputs)
        return token_embeddings + position_embeddings


# --- Build the NER model class as a keras.Model subclass
class NERModel(keras.Model):
    def __init__(
        self, num_tags,
        vocab_size,
        maxlen=128,
        # embed_dim=32,
        embed_dim=128,
        # num_heads=2,
        num_heads=8,
        ff_dim=32,
        num_transformer_layers=5
    ):
        super().__init__()
        self.embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
        self.transformer_blocks = [TransformerBlock(embed_dim, num_heads, ff_dim) for _ in range(num_transformer_layers)]
        self.dropout1 = layers.Dropout(0.1)
        self.ff = layers.Dense(ff_dim, activation="relu")
        self.dropout2 = layers.Dropout(0.1)
        self.ff_final = layers.Dense(num_tags, activation="softmax")

    def call(self, inputs, training=False):
        x = self.embedding_layer(inputs)
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)
        x = self.dropout1(x, training=training)
        x = self.ff(x)
        x = self.dropout2(x, training=training)
        x = self.ff_final(x)
        return x


class CustomNonPaddingTokenLoss(keras.losses.Loss):
    def __init__(self, name="custom_ner_loss"):
        super().__init__(name=name)

    def call(self, y_true, y_pred):
        loss_fn = keras.losses.SparseCategoricalCrossentropy(
            from_logits=None, reduction=keras.losses.Reduction.NONE
        )
        loss = loss_fn(y_true, y_pred)
        mask = tf.cast((y_true > 0), dtype=tf.float32)
        loss = loss * mask
        return tf.reduce_sum(loss) / tf.reduce_sum(mask)


# ---------------- Helper Functions ----------------

def load_and_prepare_data():
    conll_data = load_dataset("conll2003")
    if not os.path.exists("./data"):
        os.mkdir("./data")
    export_to_file("./data/conll_train.txt", conll_data["train"])
    export_to_file("./data/conll_val.txt", conll_data["validation"])
    return conll_data

def export_to_file(export_file_path, data):
    with open(export_file_path, "w") as f:
        for record in data:
            ner_tags = record["ner_tags"]
            tokens = record["tokens"]
            if len(tokens) > 0:
                f.write(
                    str(len(tokens))
                    + "\t"
                    + "\t".join(tokens)
                    + "\t"
                    + "\t".join(map(str, ner_tags))
                    + "\n"
                )
                
                
def prepare_data_directory():
    if not os.path.exists("./data"):
        os.mkdir("./data")

def make_tag_lookup_table():
    iob_labels = ["B", "I"]
    ner_labels = ["PER", "ORG", "LOC", "MISC"]
    all_labels = [(label1, label2) for label2 in ner_labels for label1 in iob_labels]
    all_labels = ["-".join([a, b]) for a, b in all_labels]
    all_labels = ["[PAD]", "O"] + all_labels
    return dict(zip(range(0, len(all_labels) + 1), all_labels))


def get_vocabulary(conll_data, vocab_size):
    all_tokens = sum(conll_data["train"]["tokens"], [])
    all_tokens_array = np.array(list(map(str.lower, all_tokens)))
    counter = Counter(all_tokens_array)
    vocabulary = [token for token, count in counter.most_common(vocab_size - 2)]
    return vocabulary

def tokenize_and_convert_to_ids(text, lookup_layer):
    tokens = text.split()
    tokens = tf.strings.lower(tokens)
    return lookup_layer(tokens)

def lowercase_and_convert_to_ids(tokens, lookup_layer):
    tokens = tf.strings.lower(tokens)
    return lookup_layer(tokens)

def map_record_to_training_data(record):
    record = tf.strings.split(record, sep="\t")
    length = tf.strings.to_number(record[0], out_type=tf.int32)
    tokens = record[1 : length + 1]
    tags = record[length + 1 :]
    tags = tf.strings.to_number(tags, out_type=tf.int64)
    tags += 1
    return tokens, tags

def prepare_datasets(vocabulary, batch_size):
    lookup_layer = keras.layers.StringLookup(vocabulary=vocabulary)
    train_data = tf.data.TextLineDataset("./data/conll_train.txt")
    val_data = tf.data.TextLineDataset("./data/conll_val.txt")
    train_dataset = (
        train_data.map(map_record_to_training_data)
        .map(lambda x, y: (lowercase_and_convert_to_ids(x, lookup_layer), y))
        .padded_batch(batch_size)
    )
    val_dataset = (
        val_data.map(map_record_to_training_data)
        .map(lambda x, y: (lowercase_and_convert_to_ids(x, lookup_layer), y))
        .padded_batch(batch_size)
    )
    return train_dataset, val_dataset

# ---------------- Model: Training and Predicting ----------------

def create_model(num_tags, vocab_size):
    ner_model = NERModel(num_tags, vocab_size, embed_dim=32, num_heads=4, ff_dim=64)
    return ner_model

def compile_and_fit(model, train_dataset, epochs=10):
    loss = CustomNonPaddingTokenLoss()
    model.compile(optimizer="adam", loss=loss)
    model.fit(train_dataset, epochs=epochs)

def predict_sample(model, text, mapping, lookup_layer):
    sample_input = tokenize_and_convert_to_ids(text, lookup_layer)
    sample_input = tf.reshape(sample_input, shape=[1, -1])
    output = model.predict(sample_input)
    prediction = np.argmax(output, axis=-1)[0]
    prediction = [mapping[i] for i in prediction]
    return prediction

# ---------------- Model: Evaluation ----------------

def calculate_metrics(dataset, ner_model, mapping, verbose=False):
    all_true_tag_ids, all_predicted_tag_ids = [], []
    
    for x, y in dataset:
        output = ner_model.predict(x, verbose=0)  # set verbose to 0
        predictions = np.argmax(output, axis=-1)
        predictions = np.reshape(predictions, [-1])

        true_tag_ids = np.reshape(y, [-1])

        mask = (true_tag_ids > 0) & (predictions > 0)
        true_tag_ids = true_tag_ids[mask]
        predicted_tag_ids = predictions[mask]

        all_true_tag_ids.append(true_tag_ids)
        all_predicted_tag_ids.append(predicted_tag_ids)

    all_true_tag_ids = np.concatenate(all_true_tag_ids)
    all_predicted_tag_ids = np.concatenate(all_predicted_tag_ids)

    predicted_tags = [mapping[tag] for tag in all_predicted_tag_ids]
    real_tags = [mapping[tag] for tag in all_true_tag_ids]
    
    res = evaluate(real_tags, predicted_tags, verbose = verbose)

    return res



def main():

    vocab_size = 20000
    batch_size = 32
    epochs = 15
    sample_text = "eu rejects german call to boycott british lamb"
    
    
    print(f"processing data and preparing vocabulary of size {vocab_size}...")    
    conll_data = load_and_prepare_data()

    mapping = make_tag_lookup_table()

    # vocab_size = 20000
    vocabulary = get_vocabulary(conll_data, vocab_size)

    print(f"preparing datasets...")
    lookup_layer = keras.layers.StringLookup(vocabulary=vocabulary)

    # batch_size = 32
    train_dataset, val_dataset = prepare_datasets(vocabulary, batch_size)

    num_tags = len(mapping)
    
    print(f"creating model...\n")
    ner_model = create_model(num_tags, vocab_size)

    print(f"training model...\n")
    compile_and_fit(ner_model, train_dataset, epochs=epochs)

    print(predict_sample(ner_model, sample_text, mapping, lookup_layer))
    
    print(f"calculating metrics...\n")
    res = calculate_metrics(val_dataset, ner_model, mapping, verbose = True)
    
    # res is a tuple of (precision, recall, f1), print it out beautifully
    print("\n")
    print(f"precision: \t{res[0]:.2f}")
    print(f"   recall: \t{res[1]:.2f}")
    print(f"       f1: \t{res[2]:.2f}")
        
    return res
    
    




