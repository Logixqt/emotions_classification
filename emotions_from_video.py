import cv2
import numpy as np
import pandas as pd
import os
from pathlib import Path
from copy import deepcopy
from tensorflow.keras.applications import EfficientNetB1
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.preprocessing.image import smart_resize
import moviepy.editor as mpe
import time

# Defining the trained model and loading our weights
model = Sequential([
    EfficientNetB1(
        input_shape=(240, 240, 3), # default size for EffnetB1
        weights=None,
        include_top=False),
    GlobalAveragePooling2D(),
    Dense(9, activation='softmax') # we've trained to predict 9 emotions
])

model.load_weights('C:/Users/mbabaev/Desktop/ML_projects/Skillbox/diploma_neural/model/model.h5')

VIDEO_PATH = 'C:/Users/mbabaev/Desktop/ML_projects/Skillbox/diploma_neural/bril_hand.mp4'
DETECTOR_PATH = 'C:/Users/mbabaev/Desktop/ML_projects/Skillbox/diploma_neural/face_detector/haarcascade_frontalface_default.xml' 
# default face detector
# ref: https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml

MAPPING = {0: 'anger', 1: 'contempt', 2: 'disgust',
           3: 'fear', 4: 'happy', 5: 'neutral',
           6: 'sad', 7: 'surprise', 8: 'uncertain'} # classes to emotions mapping

def predict_draw_emotion(img, face, model, mapping=MAPPING):
    """
    Predicts emotion for the face at the image and draws the bounding box 
    with emotion on the image.
    img: np.array of shape (heith, width, channels), the image.
    face: a list of [x, y, w, h] coordinates of the bounding box of 
    the face to predict emotions for.
    model: the model predicting emotions
    mapping: predictions decoding dict
    returns > the image with the predicted emotion and the bounding box drawn on it.
    """
    x, y, w, h = face
    face_boundingbox_rgb = img[y:y + h, x:x + w]

    rgb_image_with_boundingbox = deepcopy(img)
    rgb_image_with_boundingbox = cv2.rectangle(
        rgb_image_with_boundingbox, (x, y), (x + w, y + h), (0,255,0), 3)

    img = smart_resize(face_boundingbox_rgb, (240, 240), interpolation='nearest')

    emotion = mapping[np.argmax(model.predict(img[None, ...]), axis=-1)[0]]

    rgb_image_with_boundingbox_and_text = deepcopy(rgb_image_with_boundingbox)
    rgb_image_with_boundingbox_and_text = cv2.putText(
        rgb_image_with_boundingbox_and_text, emotion, (x + 10, y - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    return rgb_image_with_boundingbox_and_text, emotion

def analyse_video(video_path, classifier, detector_path, compact=True, 
                  show_process=False, write_video=True, output_filename=None):
    """
    Creates a dataframe for a video, where data on people's emotions on the
    video is stored. If write_video = True, writes a video with detected faces
    and predicted emotions for them.
    video_path: path to the video to analyse.
    classifier: a model, which classifies emotions.
    detector: path to haarcascade detector.
    compact: whether to return a compact dataframe. If True, returns a dataframe
    where periods of emotions are stored; if False, returns a dataframe with
    emotions for each frame/timestamp are stored
    write_video: whether to write the processed video
    output_filename: output video filename, only to be specified if write_video = True
    returns > dataframe with emotions schedule
    """
    base_filename = '.'.join(output_filename.split('.')[:-1])
    files_dir = Path(video_path).parent
    tmp_path = os.path.join(files_dir, base_filename+f'_tmp_{time.time()}.mp4')
    output_path = os.path.join(files_dir, output_filename)
    
    face_detector = cv2.CascadeClassifier(detector_path) # building detector
    cap = cv2.VideoCapture(video_path) # getting the video
    
    if write_video: 
        # creating a VideoWriter if write_video = True
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        writer = cv2.VideoWriter(tmp_path, 0, fourcc=fourcc, fps=fps, 
                                 frameSize=(frame_width, frame_height),
                                 params=[2, -1])
    ret = True # when video ends, ret will turn to False
    data = {'ms': [], 'frame': [], 'emotions': []} # data for df
    while(ret):
        ret, frame = cap.read() # getting next frame
        
        if ret:
            # cv2 default channels are GRB, we change the frame to RGB here, 
            # as models are usually trained on RGB images
            rgb_frame = frame[:, :, ::-1] 
            
            # base cv2 detector requires a greyscale image, so we convert colors
            grayscale_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
            faces = face_detector.detectMultiScale(grayscale_image, 1.3, 5) # detecting faces
            
            # drawing bounding boxes with predicted emotions for each detected face
            emotions = []
            for face in faces: 
                rgb_frame, emotion = predict_draw_emotion(
                    img=rgb_frame, face=face, model=classifier)
                emotions.append(emotion)
                
            if write_video:
                writer.write(rgb_frame[:, :, ::-1])
    
            pos_msec = cap.get(cv2.CAP_PROP_POS_MSEC) # cur msec position
            pos_frames = cap.get(cv2.CAP_PROP_POS_FRAMES) # cur frame num
            data['ms'].append(pos_msec)
            data['frame'].append(pos_frames)
            data['emotions'].append(emotion if len(emotions) == 1 else emotions)
            
            if show_process:
                cv2.imshow("video analysis process", rgb_frame[:,:,::-1])
        
    cap.release()
    writer.release()
    cv2.destroyAllWindows()
            
    if compact:
        compact = {'ms_start': np.array([]), 'ms_end': np.array([]), 
                   'frame_start': np.array([]), 'frame_end': np.array([]), 
                   'emotions': []} # data for compact df
        curr = None
        for ms, frame, emotions in zip(*data.values()):
          curr, prev = emotions, curr # curr frame emotions update
          if curr != prev: 
            # if emotions changed, create a new future df row
            compact['ms_start'] = np.append(compact['ms_start'], ms)
            compact['ms_end'] = np.append(compact['ms_end'], ms)
            compact['frame_start'] = np.append(compact['frame_start'], frame)
            compact['frame_end'] = np.append(compact['frame_end'], frame)
            compact['emotions'].append(emotions)
          else:
            # if emotion remains the same, update current emotion end data
            compact['ms_end'][-1] = ms
            compact['frame_end'][-1] = frame
        df = pd.DataFrame(compact)
        df['frame_start'] = df.frame_start.astype(int)
        df['frame_end'] = df.frame_end.astype(int)
    else:
        df = pd.DataFrame(data)
        df['frame'] = df.frame.astype(int)
    
    df.to_csv(os.path.join(files_dir, base_filename+'_emotions_data.csv'), index=False)
    
    if write_video:
        # first we need the initial video for getting its sound as we didn't
        # process any sound while writing the video with cv2 
        init_video = mpe.VideoFileClip(video_path) # reading initial video 
        processed_video = mpe.VideoFileClip(tmp_path) # reading new video
        # setting initial video's sound to the new video
        processed_video_with_sound = processed_video.set_audio(init_video.audio)
        # writing the new video with sound to a file
        processed_video_with_sound.write_videofile(output_path)
        
        # closing the videofiles and deleting the tmp file
        processed_video.close()
        init_video.close()
        processed_video_with_sound.close()
        os.remove(tmp_path)
        
    return df

if __name__ == '__main__':
    analyse_video(VIDEO_PATH, model, DETECTOR_PATH, 
                  output_filename='bril_hand_with_emotions.mp4')