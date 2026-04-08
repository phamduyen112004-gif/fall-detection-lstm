"""Custom training losses."""

import tensorflow as tf


def binary_focal_loss(gamma: float = 2.0, alpha: float = 0.25):
    def loss(y_true, y_pred):
        y_true_cast = tf.cast(y_true, tf.float32)
        eps = tf.keras.backend.epsilon()
        y_pred_clip = tf.clip_by_value(y_pred, eps, 1.0 - eps)

        p_t = y_true_cast * y_pred_clip + (1.0 - y_true_cast) * (1.0 - y_pred_clip)
        alpha_t = y_true_cast * alpha + (1.0 - y_true_cast) * (1.0 - alpha)
        focal_term = -alpha_t * tf.pow((1.0 - p_t), gamma) * tf.math.log(p_t)
        return tf.reduce_mean(focal_term)

    return loss
