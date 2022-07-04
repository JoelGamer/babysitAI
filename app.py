import cv2
import time
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import img_to_array

ROOT_DIR = os.path.dirname(os.path.abspath(__file__)) + '/'
MODELS_PATH = ROOT_DIR + 'models/'
TFLITES_PATH =  MODELS_PATH + 'tflites/'

face_classifier = cv2.CascadeClassifier('./models/haarcascade_frontalface_default.xml')

# Load the TFLite model and allocate tensors.
emotion_interpreter = tf.lite.Interpreter(model_path=TFLITES_PATH + "emotion_detection_model_100epochs_no_opt.tflite")
emotion_interpreter.allocate_tensors()

age_interpreter = tf.lite.Interpreter(model_path=TFLITES_PATH + "age_detection_model_100epochs_no_opt.tflite")
age_interpreter.allocate_tensors()

# Get input and output tensors.
emotion_input_details = emotion_interpreter.get_input_details()
emotion_output_details = emotion_interpreter.get_output_details()

age_input_details = age_interpreter.get_input_details()
age_output_details = age_interpreter.get_output_details()

# Test the model on input data.
emotion_input_shape = emotion_input_details[0]['shape']
age_input_shape = age_input_details[0]['shape']

class_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

cap = cv2.VideoCapture(2)
frame_rate = 24
prev = 0

while True:
  time_elapsed = time.time() - prev
  res, image = cap.read()

  if time_elapsed > 1./frame_rate:
    ret,frame = cap.read()
    labels = []
    
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray,1.3,5)

    for (x,y,w,h) in faces:
      cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

      roi_gray = gray[y:y+h,x:x+w]
      roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)

      roi = roi_gray.astype('float')/255.0
      roi = img_to_array(roi)
      roi = np.expand_dims(roi,axis=0)
      
      emotion_interpreter.set_tensor(emotion_input_details[0]['index'], roi)
      emotion_interpreter.invoke()
      emotion_preds = emotion_interpreter.get_tensor(emotion_output_details[0]['index'])

      emotion_label=class_labels[emotion_preds.argmax()]
      emotion_label_position=(x,y)
      cv2.putText(frame,emotion_label,emotion_label_position,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
      
      roi_color = frame[y:y+h,x:x+w]
      roi_color = cv2.resize(roi_color,(200,200),interpolation=cv2.INTER_AREA)
      roi_color = np.array(roi_color).reshape(-1,200,200,3)
      roi_color = roi_color.astype(np.float32)

      # Age
      age_interpreter.set_tensor(age_input_details[0]['index'], roi_color)
      age_interpreter.invoke()
      age_preds = age_interpreter.get_tensor(age_output_details[0]['index'])

      age = round(age_preds[0,0])
      age_label_position = (x+h, y+h)
      cv2.putText(frame,"Age="+str(age),age_label_position,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

  cv2.imshow('Emotion and Age Detector', frame)
  if cv2.waitKey(1) & 0xFF == ord('q'):
      break

cap.release()
cv2.destroyAllWindows()