"""
U-Net Segmentation Model (Research Grade)
Standard architecture for biomedical/industrial image segmentation (e.g. WM-811K).
"""
import tensorflow as tf
from tensorflow import keras
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

def downsample_block(x, n_filters):
    f = double_conv_block(x, n_filters)
    p = layers.MaxPool2D(2)(f)
    p = layers.Dropout(0.3)(p)
    return f, p

def upsample_block(x, conv_features, n_filters):
    # Upsample -> Concatenate -> Double Conv
    x = layers.Conv2DTranspose(n_filters, 3, 2, padding="same")(x)
    x = layers.concatenate([x, conv_features])
    x = layers.Dropout(0.3)(x)
    x = double_conv_block(x, n_filters)
    return x

def create_unet_model(input_shape=(128, 128, 3)):
    inputs = layers.Input(shape=input_shape)
    
    # --- ENCODER (Contracting Path) ---
    # Normalizing inputs to [0,1] is usually expected if raw is [0,255]
    # But let's assume preprocessing handles that or add a Rescaling layer
    s = layers.Rescaling(1./255)(inputs) 

    f1, p1 = downsample_block(s, 64)
    f2, p2 = downsample_block(p1, 128)
    f3, p3 = downsample_block(p2, 256)
    f4, p4 = downsample_block(p3, 512)

    # --- BOTTLENECK ---
    bottleneck = double_conv_block(p4, 1024)

    # --- DECODER (Expansive Path) ---
    u6 = upsample_block(bottleneck, f4, 512)
    u7 = upsample_block(u6, f3, 256)
    u8 = upsample_block(u7, f2, 128)
    u9 = upsample_block(u8, f1, 64)

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
