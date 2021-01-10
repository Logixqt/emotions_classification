import cv2
import numpy as np
from copy import deepcopy
from tensorflow.keras.applications import EfficientNetB1
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.preprocessing.image import smart_resize

def predict_draw_emotion(img, face):
    """
    Predicts emotion for the face at the image and draws the bounding box with emotion on the image.
    img: np.array of shape (heith, width, channels), the image.
    face: a list of [x, y, w, h] coordinates of the bounding box of the face to predict emotions for.
    returns: the image with the predicted emotion and the bounding box drawn on it.
    """
    x, y, w, h = face
    face_boundingbox_rgb = img[y:y + h, x:x + w]

    rgb_image_with_boundingbox = deepcopy(img)
    rgb_image_with_boundingbox = cv2.rectangle(
        rgb_image_with_boundingbox, (x, y), (x + w, y + h), (0,255,0), 3)

    img = smart_resize(face_boundingbox_rgb, (240, 240), interpolation='nearest')

    # could be replaced with the commented lines for better prediction
    emotion = mapping[np.argmax(model.predict(img[None, ...]), axis=-1)[0]]

#         pred = model.predict(img[None, ...])
#         pred_f = model.predict(np.flip(img, axis=1)[None, ...])
#         tta_pred = np.stack((pred, pred_f)).mean(axis=0).argmax(axis=-1)[0]
#         emotion = mapping[tta_pred]

    rgb_image_with_boundingbox_and_text = deepcopy(rgb_image_with_boundingbox)
    rgb_image_with_boundingbox_and_text = cv2.putText(
        rgb_image_with_boundingbox_and_text, emotion, (x + 10, y - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    return rgb_image_with_boundingbox_and_text

# Defining the trained model and loading our weights
model = Sequential([
    EfficientNetB1(
        input_shape=(240, 240, 3), # default size for EffnetB1
        weights=None,
        include_top=False),
    GlobalAveragePooling2D(),
    Dense(9, activation='softmax') # we've trained to predict 9 emotions
])

model.load_weights('./model_weights/model.h5')

cam = cv2.VideoCapture(0) #getting camera for video capturing
face_detector = cv2.CascadeClassifier(
    './face_detector/haarcascade_frontalface_default.xml') # default face detector, ref: https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml

mapping = {0: 'anger', 1: 'contempt', 2: 'disgust',
           3: 'fear', 4: 'happy', 5: 'neutral',
           6: 'sad', 7: 'surprise', 8: 'uncertain'} # classes to emotions mapping

# here starts an infinite loop (press 'q' to interrupt)
while(True): 
    ret, frame = cam.read() #getting the frame
    
    # cv2 default channels are GRB, we change the frame to RGB here, as the
    # model is trained on RGB images
    rgb_frame = frame[:, :, ::-1] 
    
    # detector requires a greyscale image, so we convert colors
    grayscale_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
    faces = face_detector.detectMultiScale(grayscale_image, 1.3, 5) # detecting faces
    
    # drawing bounding boxes with predicted emotions for each detected face
    for face in faces: 
        rgb_frame = predict_draw_emotion(img=rgb_frame, face=face)
    cv2.imshow("facial emotion recognition", rgb_frame[:,:,::-1])

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break