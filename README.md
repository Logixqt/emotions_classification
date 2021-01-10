# Emotions classification

This is the final project at Skillbox "Data Scientist" course. The task is to train 
an emotion classification model on a dataset of 50 000 labeled images. The results are evaluated 
at Kaggle on 5 000 test images (more detailes regarding the data can be found at 
[the competition page](https://www.kaggle.com/c/skillbox-computer-vision-project/overview)). 
Should the model achieve over 0.4 `categorization accuracy`, 
it is to be used to classify emotions of people via webcam. 

While looking for the best model, I tried different pipelines, using EfficientNet as the 
classifier, starting with direct training, further implementing imagenet weights initialization 
an adding some augmentations. Different pipelines with interim results cat be found at 
`draft_pipelines` folder.

Finally EfficientNetB1 turned to be the best classifier, initialized with imagenet weights 
and trained on augmented dataset for 20 epochs. The metric with Test-Time Augmentation was 
0.5736 at private leaderboard. The pipeline with the best model is located at `model_training.py` file.

The real-time emotions classification using webcam video is realized at `emotions_from_webcam.py` file. 
I use open-cv library to both read webcam and detect faces, then the faces' pictures are given to the 
trained model and the real-time predictions go along with the video.