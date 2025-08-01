import tensorflow as tf
tf.autograph.set_verbosity(0)
from tensorflow import keras
from tensorflow.keras import layers
from joblib import load
import os
#from misc import create_tf_dataset
from datetime import datetime
import numpy as np

############
# Code taken, adapted and modified from: 
# https://github.com/keras-team/keras-io/blob/master/examples/audio/transformer_asr.py
# Copyright 2020. The Keras authors and Apoorv Nandan
# Released under Apache 2.0 License.
#############

########## Transformer ###############
# Define the Transformer Input Layer
class TokenEmbedding(layers.Layer):
    def __init__(self, num_vocab=1000, maxlen=100, num_hid=64):
        super().__init__()
        self.emb = tf.keras.layers.Embedding(num_vocab, num_hid)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=num_hid)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        x = self.emb(x)
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        return x + positions


class SpeechFeatureEmbedding(layers.Layer):
    def __init__(self, num_hid=64, maxlen=100):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv1D(
            num_hid, 11, strides=2, padding="same", activation="relu"
        )
        self.conv2 = tf.keras.layers.Conv1D(
            num_hid, 11, strides=2, padding="same", activation="relu"
        )
        self.conv3 = tf.keras.layers.Conv1D(
            num_hid, 11, strides=2, padding="same", activation="relu"
        )

    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return self.conv3(x)


# Transformer Encoder Layer
class TransformerEncoder(layers.Layer):
    def __init__(self, embed_dim, num_heads, feed_forward_dim, rate=0.0):
        super().__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [
                layers.Dense(feed_forward_dim, activation="relu"),
                layers.Dense(embed_dim),
            ]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


# Transformer Decoder Layer
class TransformerDecoder(layers.Layer):
    def __init__(self, embed_dim, num_heads, feed_forward_dim, dropout_rate=0.0):
        super().__init__()
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = layers.LayerNormalization(epsilon=1e-6)
        self.self_att = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        self.enc_att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.self_dropout = layers.Dropout(0.0)
        self.enc_dropout = layers.Dropout(0.0)
        self.ffn_dropout = layers.Dropout(0.0)
        self.ffn = keras.Sequential(
            [
                layers.Dense(feed_forward_dim, activation="relu"),
                layers.Dense(embed_dim),
            ]
        )

    def causal_attention_mask(self, batch_size, n_dest, n_src, dtype):
        """Masks the upper half of the dot product matrix in self attention.

        This prevents flow of information from future tokens to current token.
        1's in the lower triangle, counting from the lower right corner.
        """
        i = tf.range(n_dest)[:, None]
        j = tf.range(n_src)
        m = i >= j - n_src + n_dest
        mask = tf.cast(m, dtype)
        mask = tf.reshape(mask, [1, n_dest, n_src])
        mult = tf.concat(
            [tf.expand_dims(batch_size, -1), tf.constant([1, 1], dtype=tf.int32)], 0
        )
        return tf.tile(mask, mult)

    def call(self, enc_out, target):
        input_shape = tf.shape(target)
        batch_size = input_shape[0]
        seq_len = input_shape[1]
        causal_mask = self.causal_attention_mask(batch_size, seq_len, seq_len, tf.bool)
        target_att = self.self_att(target, target, attention_mask=causal_mask)
        target_norm = self.layernorm1(target + self.self_dropout(target_att))
        enc_out = self.enc_att(target_norm, enc_out)
        enc_out_norm = self.layernorm2(self.enc_dropout(enc_out) + target_norm)
        ffn_out = self.ffn(enc_out_norm)
        ffn_out_norm = self.layernorm3(enc_out_norm + self.ffn_dropout(ffn_out))
        return ffn_out_norm


# Complete the Transformer model 
class Transformer(keras.Model):
    def __init__(
        self,
        num_hid=64,
        num_head=2,
        num_feed_forward=128,
        source_maxlen=100,
        target_maxlen=100,
        num_layers_enc=4,
        num_layers_dec=1,
        num_classes=10,
        name="NanoSpeech",
    ):
        super().__init__()
        self.loss_metric = keras.metrics.Mean(name="loss")
        self.num_layers_enc = num_layers_enc
        self.num_layers_dec = num_layers_dec
        self.target_maxlen = target_maxlen
        self.num_classes = num_classes
        
        self.enc_input = SpeechFeatureEmbedding(num_hid=num_hid, maxlen=source_maxlen)
        self.dec_input = TokenEmbedding(
            num_vocab=num_classes, maxlen=target_maxlen, num_hid=num_hid
        )

        self.encoder = keras.Sequential(
            [self.enc_input]
            + [
                TransformerEncoder(num_hid, num_head, num_feed_forward)
                for _ in range(num_layers_enc)
            ]
        )

        for i in range(num_layers_dec):
            setattr(
                self,
                f"dec_layer_{i}",
                TransformerDecoder(num_hid, num_head, num_feed_forward),
            )

        self.classifier = layers.Dense(num_classes)

    def decode(self, enc_out, target):
        y = self.dec_input(target)
        for i in range(self.num_layers_dec):
            y = getattr(self, f"dec_layer_{i}")(enc_out, y)
        return y

    def call(self, inputs):
        source = inputs[0]
        target = inputs[1]
        x = self.encoder(source)
        y = self.decode(x, target)
        return self.classifier(y)

    @property
    def metrics(self):
        return [self.loss_metric]

    def train_step(self, batch):
        """Processes one batch inside model.fit()."""
        source = batch[0]
        target = batch[1]
        dec_input = target[:, :-1]
        dec_target = target[:, 1:]
        with tf.GradientTape() as tape:
            preds = self([source, dec_input])
            one_hot = tf.one_hot(dec_target, depth=self.num_classes)
            mask = tf.math.logical_not(tf.math.equal(dec_target, 0))
            loss = self.compiled_loss(one_hot, preds, sample_weight=mask)
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        self.loss_metric.update_state(loss)
        return {"loss": self.loss_metric.result()}

    def test_step(self, batch):
        source = batch[0]
        target = batch[1]
        dec_input = target[:, :-1]
        dec_target = target[:, 1:]
        preds = self([source, dec_input])
        one_hot = tf.one_hot(dec_target, depth=self.num_classes)
        mask = tf.math.logical_not(tf.math.equal(dec_target, 0))
        loss = self.compiled_loss(one_hot, preds, sample_weight=mask)
        self.loss_metric.update_state(loss)
        return {"loss": self.loss_metric.result()}

    def generate(self, source, target_start_token_idx, return_proba=False):
        """Performs inference over one batch of inputs using greedy decoding."""
        #print("NANOSPEECH GENERATE METHOD DEVELOPMENT MESSAGE - self.target_maxlen:", self.target_maxlen, flush=True)
        bs = tf.shape(source)[0]
        enc = self.encoder(source)
        dec_input = tf.ones((bs, 1), dtype=tf.int32) * target_start_token_idx
        dec_logits = []
        dec_logits_proba = []
        for i in range(self.target_maxlen - 1):
            dec_out = self.decode(enc, dec_input)
            logits = self.classifier(dec_out)
            if return_proba:
                dec_logits_proba.append(logits)
            logits = tf.argmax(logits, axis=-1, output_type=tf.int32)
            last_logit = tf.expand_dims(logits[:, -1], axis=-1)
            dec_logits.append(last_logit)
            dec_input = tf.concat([dec_input, last_logit], axis=-1)
        if return_proba:
            return dec_input, dec_logits_proba[-1]
        else:
            return dec_input

# defining Learning rate schedule
class CustomSchedule(keras.optimizers.schedules.LearningRateSchedule):
    def __init__(
        self,
        init_lr=0.00001,
        lr_after_warmup=0.001,
        final_lr=0.00001,
        warmup_epochs=15,
        decay_epochs=85,
        steps_per_epoch=203,
    ):
        super().__init__()
        self.init_lr = init_lr
        self.lr_after_warmup = lr_after_warmup
        self.final_lr = final_lr
        self.warmup_epochs = warmup_epochs
        self.decay_epochs = decay_epochs
        self.steps_per_epoch = steps_per_epoch

    def calculate_lr(self, epoch):
        """ linear warm up - linear decay """
        warmup_lr = (
            self.init_lr
            + ((self.lr_after_warmup - self.init_lr) / (self.warmup_epochs - 1)) * epoch
        )
        decay_lr = tf.math.maximum(
            self.final_lr,
            self.lr_after_warmup
            - (epoch - self.warmup_epochs)
            * (self.lr_after_warmup - self.final_lr)
            / (self.decay_epochs),
        )
        return tf.math.minimum(warmup_lr, decay_lr)

    def __call__(self, step):
        epoch = step // self.steps_per_epoch
        epoch = tf.cast(epoch, "float32")
        return self.calculate_lr(epoch)


# Preprocess the dataset
class VectorizeChar:
    def __init__(self, max_len=50):
        self.vocab = (
            ["-","<", ">", "a", "c", "g", "t", "i"]
        )
        self.max_len = max_len
        self.char_to_idx = {}
        for i, ch in enumerate(self.vocab):
            self.char_to_idx[ch] = i

    def __call__(self, text):
        text = text.lower()
        text = text[: self.max_len - 2]
        text = "<" + text + ">"
        pad_len = self.max_len - len(text)
        return [self.char_to_idx.get(ch, 1) for ch in text] + [0] * pad_len

    def get_vocabulary(self):
        return self.vocab
    
class VectorizeCharMulti:
    def __init__(self, max_len=50, vocab=["-","<", ">", "a", "c", "g", "t", "i"]):
        self.vocab = (vocab)
        self.max_len = max_len
        self.char_to_idx = {}
        for i, ch in enumerate(self.vocab):
            self.char_to_idx[ch] = i

    def __call__(self, text):
        text = text.lower()
        text = text[: self.max_len - 2]
        text = "<" + text + ">"
        pad_len = self.max_len - len(text)
        return [self.char_to_idx.get(ch, 1) for ch in text] + [0] * pad_len

    def get_vocabulary(self):
        return self.vocab

# start model
def initialize_model(m_idx, model_weigths, num_classes=8, target_maxlen=250, pad_len=971, 
                     num_hid=250,
                     num_head=10,
                     num_feed_forward=450,
                     num_layers_enc=4,
                     num_layers_dec=2):
    print("\nINITIALIZE MODEL INPUT ARGS:")     ### DEVELOPMENT ###
    print("\t- m_idx:", m_idx)                  ### DEVELOPMENT ###
    print("\t- model_weigths:", model_weigths)  ### DEVELOPMENT ###
    print("\t- num_classes:", num_classes)      ### DEVELOPMENT ###
    print("\t- target_maxlen:", target_maxlen)  ### DEVELOPMENT ###
    print("\t- pad_len:", pad_len)              ### DEVELOPMENT ###
    print("\t- num_hid:", num_hid)
    print("\t- num_head:", num_head)
    print("\t- num_feed_forward:", num_feed_forward)
    print("\t- num_layers_enc:", num_layers_enc)
    print("\t- num_layers_dec:", num_layers_dec)
    # vectorizer = VectorizeChar(target_maxlen)
    # the model's architecture
    model = Transformer(
        num_hid=num_hid,
        num_head=num_head,
        num_feed_forward=num_feed_forward,
        target_maxlen=target_maxlen,
        num_layers_enc=num_layers_enc,
        num_layers_dec=num_layers_dec,
        num_classes=num_classes,
    )
    model._name=f"NanoSpeech_{m_idx}"

    loss_fn = tf.keras.losses.CategoricalCrossentropy(
        from_logits=True, label_smoothing=0.1,
    )
    init_lr_ = 0.00003
    epochs = 1000
    learning_rate = CustomSchedule(
        init_lr=init_lr_,
        lr_after_warmup=init_lr_*10,
        final_lr=init_lr_,
        warmup_epochs=epochs*0.10,
        decay_epochs=epochs-(epochs*0.10),
        steps_per_epoch=64,
    )

    optimizer = keras.optimizers.Adam(learning_rate)
    model.compile(optimizer=optimizer, loss=loss_fn)

    # load and initialize pretrained model using a fake sample to initialize the input shapes
    print(f"[{datetime.now()}] Initializing the model...", flush=True)
    X_fake = np.array([np.random.randint(-7,7,126*pad_len).astype(float).reshape(pad_len, 126) for i in range(10)])
    y_fake = np.array([np.random.randint(0,6,target_maxlen,dtype="int32") for i in range(X_fake.shape[0])])    
    # first call of the model and loading
    model.fit(x=X_fake, y=y_fake, epochs=1, verbose=0)
    print(f"\n[{datetime.now()}] [Consumer {m_idx} Message] Model Summary:", flush=True)
    print(model.summary(), flush=True)
    print("\n", flush=True)
    # load model weigths
    model.load_weights(model_weigths)
    return model