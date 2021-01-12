# Emotions classification

This is the final project at Skillbox "Data Scientist" course. The task is to train 
an emotion classification model on a dataset of 50 000 labeled images with 9 classes. The results 
are evaluated at Kaggle on 5 000 test images (more detailes regarding the data can be found at 
[the competition page](https://www.kaggle.com/c/skillbox-computer-vision-project/overview)). 
Here are some examples of the images: 
![samples](https://github.com/Logixqt/emotions_classification/blob/main/examples/samples.PNG)\
Should the model achieve over 0.4 `categorization accuracy`, 
it is to be used to classify emotions of people via webcam. 

While looking for the best model, I tried different pipelines, using EfficientNet as the 
classifier, starting with direct training, further implementing imagenet weights initialization 
an adding some augmentations. Different pipelines with interim results cat be found at 
`draft_pipelines` folder.

Finally EfficientNetB1 turned to be the best classifier, initialized with imagenet weights 
and trained on augmented dataset for 20 epochs. All layers were unfrozen as the imagenet dataset is not so similar 
to ours and I wanted the model to extract exactly face features. The metric with Test-Time Augmentation was 
0.5736 at private leaderboard, which is a good result taking into account 9 possible classes (a random prediction 
will hit about 0.11). The pipeline with the best model is located at `best_pipeline.ipynb` file. The training 
process is also availiable as a script `model_training.py`.

The real-time emotions classification using webcam video is realized at `emotions_from_webcam.py` file. 
I use open-cv library to both read webcam and detect faces, then the faces' pictures are given to the 
trained model and the real-time predictions go along with the video. Here you can watch a 
![video_ example*](https://github.com/Logixqt/emotions_classification/blob/main/examples/video_example.mp4)\
*the script was run on poor hardware with no GPU, that's why fps is low. Free Collab's inference time 
is 32 ms (78 ms with a flip TTA), which is quite enough for high quality videos.

## How to run the code at your hardware
1. Install the necessary libraries: 
>open-cv (4.4.0)
>numpy (1.18.5)
>tensorflow (2.3.1)
2. Download `emotions_from_webcam.py`, `face_detector/haarcascade_frontalface_default.xml` and 
`model_weights/model.h5` files
3. Either maintain the files structure the same or write full path to `haarcascade_frontalface_default.xml` 
and `model.h5` in .py script (`MODEL_WEIGHTS_PATH`, `DETECTOR_PATH` variables)
4. Run `emotions_from_webcam.py` file
5. Press `q` to cancel run