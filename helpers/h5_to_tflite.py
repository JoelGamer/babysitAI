import tensorflow as tf
from keras.models import load_model

ROOT_DIR = '/home/dr-joel/drjoel-projects/babysitAI/'
MODELS_PATH = ROOT_DIR + 'models/'
H5S_PATH =  MODELS_PATH + 'h5s/'
TFLITES_PATH =  MODELS_PATH + 'tflites/'

emotion_model = load_model(H5S_PATH + 'emotion_detection_model_100epochs.h5', compile=False)
age_model = load_model(H5S_PATH + 'age_model_50epochs.h5', compile=False)
age_model_100 = load_model(H5S_PATH + 'age_model_100epochs.h5', compile=False)

converter = tf.lite.TFLiteConverter.from_keras_model(emotion_model)
# converter.optimizations = [tf.lite.Optimize.DEFAULT] #Uses default optimization strategy to reduce the model size
tflite_model = converter.convert()
open(TFLITES_PATH + "emotion_detection_model_100epochs_no_opt.tflite", "wb").write(tflite_model)

converter = tf.lite.TFLiteConverter.from_keras_model(age_model)
# converter.optimizations = [tf.lite.Optimize.DEFAULT] #Uses default optimization strategy to reduce the model size
tflite_model = converter.convert()
open(TFLITES_PATH + "age_detection_model_50epochs_no_opt.tflite", "wb").write(tflite_model)

converter = tf.lite.TFLiteConverter.from_keras_model(age_model_100)
# converter.optimizations = [tf.lite.Optimize.DEFAULT] #Uses default optimization strategy to reduce the model size
tflite_model = converter.convert()
open(TFLITES_PATH + "age_detection_model_100epochs_no_opt.tflite", "wb").write(tflite_model)