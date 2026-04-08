"""Model architectures for fall detection."""

import os
import sys

import tensorflow as tf
import torch
from tensorflow.keras import Model
from tensorflow.keras.layers import (Bidirectional, Dense, Dropout,
                                     GlobalAveragePooling1D, Input, Layer,
                                     LSTM)
from tensorflow.keras.optimizers import Adam

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import SEQ_LEN


class TemporalAttention(Layer):
    """
    Temporal attention over sequence outputs.
    Learns frame-wise weights to emphasize high-motion falling frames.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.score_dense = Dense(1, activation="tanh")

    def call(self, inputs, **kwargs):
        # inputs: [batch, time, hidden]
        scores = self.score_dense(inputs)       # [B, T, 1]
        weights = tf.nn.softmax(scores, axis=1) # [B, T, 1]
        attended = inputs * weights             # [B, T, H]
        return attended


def build_bilstm_attention_model(
    n_features: int,
    learning_rate: float = 5e-4,
) -> Model:
    """
    Functional API model:
    Input(75, K)
      -> BiLSTM(64, return_sequences=True)
      -> BiLSTM(32, return_sequences=True)
      -> Attention
      -> GlobalAveragePooling1D
      -> Dropout(0.4)
      -> Dense(1, sigmoid)
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _ = device  # keep explicit device logic for runtime diagnostics
    inp = Input(shape=(SEQ_LEN, n_features), name="feature_sequence")
    x = Bidirectional(LSTM(64, return_sequences=True), name="bilstm_64")(inp)
    x = Bidirectional(LSTM(32, return_sequences=True), name="bilstm_32")(x)
    x = TemporalAttention(name="temporal_attention")(x)
    x = GlobalAveragePooling1D(name="global_avg_pool")(x)
    x = Dropout(0.4, name="dropout_04")(x)
    out = Dense(1, activation="sigmoid", name="fall_prob")(x)

    model = Model(inputs=inp, outputs=out, name="stacked_bilstm_attention")
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_bilstm_no_attention_model(
    n_features: int,
    learning_rate: float = 5e-4,
) -> Model:
    """Ablation model without attention layer."""
    inp = Input(shape=(SEQ_LEN, n_features), name="feature_sequence")
    x = Bidirectional(LSTM(64, return_sequences=True), name="bilstm_64")(inp)
    x = Bidirectional(LSTM(32, return_sequences=True), name="bilstm_32")(x)
    x = GlobalAveragePooling1D(name="global_avg_pool")(x)
    x = Dropout(0.4, name="dropout_04")(x)
    out = Dense(1, activation="sigmoid", name="fall_prob")(x)
    model = Model(inputs=inp, outputs=out, name="stacked_bilstm_no_attention")
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model
