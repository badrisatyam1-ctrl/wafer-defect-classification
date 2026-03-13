
import tensorflow as tf
from tensorflow import keras
from pathlib import Path
import sys

# Define custom metrics strictly as they are in the app/training
def dice_coefficient(y_true, y_pred):
    smooth = 1e-6
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
    return score

def dice_loss(y_true, y_pred):
    return 1 - dice_coefficient(y_true, y_pred)

model_path = Path("models/checkpoints/unet_best.keras")
print(f"Checking {model_path.resolve()}...")

if not model_path.exists():
    print("❌ File does not exist!")
    sys.exit(1)
    
print(f"File size: {model_path.stat().st_size / 1e6:.2f} MB")

try:
    custom_objects = {'dice_loss': dice_loss, 'dice_coefficient': dice_coefficient}
    model = keras.models.load_model(str(model_path), custom_objects=custom_objects)
    print("✅ Model loaded successfully!")
    model.summary()
except Exception as e:
    print(f"❌ Load failed: {e}")
    import traceback
    traceback.print_exc()
