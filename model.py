from tensorflow import keras
from tensorflow.keras import layers
import math


class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, feat_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="gelu"), layers.Dense(feat_dim),]
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


feat_dim = 14  # number of features
embed_dim = 64  # Embedding size for attention
num_heads = 4  # Number of attention heads
ff_dim = 128  # Hidden layer size in feed forward network inside transformer
dropout_rate = 0.3
num_blocks = 2

###################################################################################
#### Building TRANSFROMER model ####
# Embedding section was removed for this model because all features were numeric values
# But, we can add Embedding layer


def build_model():

    inp = layers.Input(shape=(24, 14), name='input')
    # We can add Embedding layer in this section
    # INPUT EMBEDDING LAYER

    x1 = layers.Dense(feat_dim)(inp)

    # TRANSFORMER BLOCKS
    for _ in range(num_blocks):
        x_old1 = x1
        transformer_block1 = TransformerBlock(
            embed_dim, feat_dim, num_heads, ff_dim, dropout_rate)
        x1 = transformer_block1(x1)
        x1 = 0.9*x1 + 0.1*x_old1  # SKIP CONNECTION

    # CLASSIFICATION HEAD
    x1 = layers.Dense(32, activation="relu")(x1[:, -1, :])
    output1 = layers.Dense(1, activation="sigmoid", name='comp1_fail')(x1)

    x2 = layers.Dense(feat_dim)(inp)

    # TRANSFORMER BLOCKS
    for _ in range(num_blocks):
        x_old2 = x2
        transformer_block2 = TransformerBlock(
            embed_dim, feat_dim, num_heads, ff_dim, dropout_rate)
        x2 = transformer_block2(x2)
        x2 = 0.9*x2 + 0.1*x_old2  # SKIP CONNECTION

    # CLASSIFICATION HEAD
    x2 = layers.Dense(32, activation="relu")(x2[:, -1, :])
    output2 = layers.Dense(1, activation="sigmoid", name='comp2_fail')(x2)

    x3 = layers.Dense(feat_dim)(inp)

    # TRANSFORMER BLOCKS
    for _ in range(num_blocks):
        x_old3 = x3
        transformer_block3 = TransformerBlock(
            embed_dim, feat_dim, num_heads, ff_dim, dropout_rate)
        x3 = transformer_block3(x3)
        x3 = 0.9*x3 + 0.1*x_old3  # SKIP CONNECTION

    # CLASSIFICATION HEAD
    x3 = layers.Dense(32, activation="relu")(x3[:, -1, :])
    output3 = layers.Dense(1, activation="sigmoid", name='comp3_fail')(x3)

    x4 = layers.Dense(feat_dim)(inp)

    # TRANSFORMER BLOCKS
    for _ in range(num_blocks):
        x_old4 = x4
        transformer_block4 = TransformerBlock(
            embed_dim, feat_dim, num_heads, ff_dim, dropout_rate)
        x4 = transformer_block4(x4)
        x4 = 0.9*x4 + 0.1*x_old4  # SKIP CONNECTION

    # CLASSIFICATION HEAD
    x4 = layers.Dense(32, activation="relu")(x4[:, -1, :])
    output4 = layers.Dense(1, activation="sigmoid", name='comp4_fail')(x4)

    model = keras.Model(inputs=inp, outputs=[
                        output1, output2, output3, output4])
    model.compile(loss={'comp1_fail': 'binary_crossentropy', 'comp2_fail': 'binary_crossentropy',
                        'comp3_fail': 'binary_crossentropy', 'comp4_fail': 'binary_crossentropy'},
                  optimizer='adam',
                  loss_weights={'comp1_fail': 0.1, 'comp2_fail': 0.1,
                                'comp3_fail': 0.1, 'comp4_fail': 0.1},
                  metrics={'comp1_fail': 'accuracy', 'comp2_fail': 'accuracy',
                           'comp3_fail': 'accuracy', 'comp4_fail': 'accuracy'}
                  )

    return model

###############################################################
# FUNCTION FOR LEARNING RATE


LR_START = 1e-6
LR_MAX = 1e-3
LR_MIN = 1e-6
LR_RAMPUP_EPOCHS = 0
LR_SUSTAIN_EPOCHS = 0
EPOCHS = 8


def lrfn(epoch):
    if epoch < LR_RAMPUP_EPOCHS:
        lr = (LR_MAX - LR_START) / LR_RAMPUP_EPOCHS * epoch + LR_START
    elif epoch < LR_RAMPUP_EPOCHS + LR_SUSTAIN_EPOCHS:
        lr = LR_MAX
    else:
        decay_total_epochs = EPOCHS - LR_RAMPUP_EPOCHS - LR_SUSTAIN_EPOCHS - 1
        decay_epoch_index = epoch - LR_RAMPUP_EPOCHS - LR_SUSTAIN_EPOCHS
        phase = math.pi * decay_epoch_index / decay_total_epochs
        cosine_decay = 0.5 * (1 + math.cos(phase))
        lr = (LR_MAX - LR_MIN) * cosine_decay + LR_MIN
    return lr
