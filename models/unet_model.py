"""
Compact U-Net segmentation model for local training and demo use.
"""
import tensorflow as tf
from tensorflow.keras import layers, Model

def double_conv_block(x, n_filters):
    # Conv2D -> BatchNorm -> ReLU -> Conv2D -> BatchNorm -> ReLU
    x = layers.Conv2D(n_filters, 3, padding = "same", kernel_initializer = "he_normal")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    
    x = layers.Conv2D(n_filters, 3, padding = "same", kernel_initializer = "he_normal")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    return x

def downsample_block(x, n_filters, dropout_rate):
    f = double_conv_block(x, n_filters)
    p = layers.MaxPool2D(2)(f)
    p = layers.Dropout(dropout_rate)(p)
    return f, p

def upsample_block(x, conv_features, n_filters, dropout_rate):
    # Upsample -> Concatenate -> Double Conv
    x = layers.Conv2DTranspose(n_filters, 3, 2, padding="same")(x)
    x = layers.concatenate([x, conv_features])
    x = layers.Dropout(dropout_rate)(x)
    x = double_conv_block(x, n_filters)
    return x

def create_unet_model(input_shape=(128, 128, 3), base_filters=16, dropout_rate=0.2):
    inputs = layers.Input(shape=input_shape)
    
    # --- ENCODER (Contracting Path) ---
    # Normalizing inputs to [0,1] is usually expected if raw is [0,255]
    # But let's assume preprocessing handles that or add a Rescaling layer
    s = layers.Rescaling(1./255)(inputs) 

    f1, p1 = downsample_block(s, base_filters, dropout_rate)
    f2, p2 = downsample_block(p1, base_filters * 2, dropout_rate)
    f3, p3 = downsample_block(p2, base_filters * 4, dropout_rate)
    f4, p4 = downsample_block(p3, base_filters * 8, dropout_rate)

    # --- BOTTLENECK ---
    bottleneck = double_conv_block(p4, base_filters * 16)

    # --- DECODER (Expansive Path) ---
    u6 = upsample_block(bottleneck, f4, base_filters * 8, dropout_rate)
    u7 = upsample_block(u6, f3, base_filters * 4, dropout_rate)
    u8 = upsample_block(u7, f2, base_filters * 2, dropout_rate)
    u9 = upsample_block(u8, f1, base_filters, dropout_rate)

    # --- OUTPUT ---
    outputs = layers.Conv2D(1, 1, padding="same", activation="sigmoid")(u9)

    model = Model(inputs, outputs, name="U-Net")
    return model

# --- CUSTOM METRICS (Research Standard) ---

def dice_coefficient(y_true, y_pred):
    smooth = 1e-6
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
    return score

def dice_loss(y_true, y_pred):
    return 1 - dice_coefficient(y_true, y_pred)
