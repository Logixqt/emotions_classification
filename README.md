# Emotions classification

This is the final project at Skillbox "Data Scientist" course. The task is to train 
an emotion classification model on a dataset of 50 000 labeled images with 9 classes. The results 
are evaluated at Kaggle on 5 000 test images (more detailes regarding the data can be found at 
[the competition page](https://www.kaggle.com/c/skillbox-computer-vision-project/overview)). 
Here are some examples of the images: 
![samples](https://github.com/Logixqt/emotions_classification/blob/main/examples/samples.PNG)\
Should the model achieve over 0.4 `categorization accuracy`, 
it is to be used to classify people's emotions in real-time via webcam. 

While looking for the best model, I tried different pipelines, using EfficientNet as the 
classifier, starting with direct training, further implementing imagenet weights initialization 
an adding some augmentations. Different pipelines with interim results can be found at 
`draft_pipelines` folder.

Finally EfficientNetB1 turned to be the best classifier, initialized with imagenet weights and trained 
on an augmented dataset for 20 epochs. All layers were unfrozen as the imagenet dataset is not so similar 
to ours and I wanted the model to extract exactly facial features. The metric with Test-Time Augmentation was 
0.5736 at private leaderboard, which is a good result taking into account 9 possible classes (a random prediction 
would score about 0.11). The pipeline with the best model is located at `best_pipeline.ipynb` file. The training 
process is also availiable as a script `model_training.py`.

The real-time emotions classification using webcam video is realized at `emotions_from_webcam.py` file. 
I use open-cv library to both read webcam and detect faces, then the faces' images are given to the 
trained model and the real-time predictions go along with the video. Here you can watch a 
![video example](https://github.com/Logixqt/emotions_classification/blob/main/examples/video_example.mp4)\

Further I wrote a program, which processes an mp4 videofile, writes a new video with emotions drawn on it
and writes a .csv file with video timestamp/frame-emotions mapping. You can watch an 
![output videofile example](https://github.com/Logixqt/emotions_classification/blob/main/examples/bril_hand_with_emotions.mp4)  
and the corresponding  
![output .csv file](https://github.com/Logixqt/emotions_classification/blob/main/examples/bril_hand_with_emotions_emotions_data.csv).

## How to run the webcam real-time emotions recognition
1. Install the necessary libraries: 
>tensorflow (2.3.1), open-cv (4.4.0), numpy (1.18.5)

2. Download `emotions_from_webcam.py`, `face_detector/haarcascade_frontalface_default.xml` and 
`model_weights/model.h5` files
3. Either maintain the files structure the same or write full path to `haarcascade_frontalface_default.xml` 
and `model.h5` in .py script (`MODEL_WEIGHTS_PATH`, `DETECTOR_PATH` variables)
4. Run `emotions_from_webcam.py`
5. Press `q` to cancel run

## How to run the video analysis
1. Install the necessary libraries: 
>tensorflow (2.3.1), open-cv (4.4.0), numpy (1.18.5), pandas (1.0.5), moviepy (1.0.3)

2. Download `emotions_from_video.py`, `face_detector/haarcascade_frontalface_default.xml` and 
`model_weights/model.h5` files
3. Write the full path to `haarcascade_frontalface_default.xml`, `model.h5` and the video you want to process 
in .py script (`MODEL_WEIGHTS_PATH`, `DETECTOR_PATH`, `VIDEO_PATH` variables)
4. Run `emotions_from_video.py`
You may read the functions description for more options