import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow.keras.applications.efficientnet as efn

EFNETS = (efn.EfficientNetB0, efn.EfficientNetB1, efn.EfficientNetB2, efn.EfficientNetB3, 
          efn.EfficientNetB4, efn.EfficientNetB5, efn.EfficientNetB6, efn.EfficientNetB7)
IMSIZES = (224, 240, 260, 300, 380, 456, 528, 600)
# made for convenience, EFNETS[i] corresponds to EfficientNetBi
# and IMSIZES[i] corresponds to EfficientNetBi Input image size

EFNET_NO = 1 # EfficientNetB1 performed the best
BATCH_SIZE = 64
IMAGE_SIZE = IMSIZES[EFNET_NO]

N_CLASSES = 9 # we have 9 emotions to predict
SEED = 17 # just for train_test_split, where I define validation dataset
IMAGES_DIR = '/kaggle/input/skillbox-emotions/'
TEST_DIR = IMAGES_DIR + '/test_kaggle'

train_df = pd.read_csv('/kaggle/input/skillbox-computer-vision-project/train.csv').iloc[:, 1:]
sub = pd.read_csv('/kaggle/input/skillbox-computer-vision-project/sample_submission.csv')

train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=SEED)
# this split is made only for validation part extraction, we have a test set 
# to evaluate our model on via Kaggle

# now we create data_generators adding some augmentations for training
# which helped to improve generalization power of the model
train_data_gen = tf.keras.preprocessing.image.ImageDataGenerator(
                                    horizontal_flip=True, rotation_range=15, 
                                    width_shift_range=0.15, height_shift_range=0.15)
val_data_gen = tf.keras.preprocessing.image.ImageDataGenerator()
test_data_gen = tf.keras.preprocessing.image.ImageDataGenerator()

train_data = train_data_gen.flow_from_dataframe(
              train_df, directory=IMAGES_DIR, x_col='image_path', y_col='emotion', class_mode='sparse',
              target_size=(IMAGE_SIZE, IMAGE_SIZE), batch_size=BATCH_SIZE, shuffle=True
          )
val_data = val_data_gen.flow_from_dataframe(
              val_df, directory=IMAGES_DIR, x_col='image_path', y_col='emotion', class_mode='sparse',
              target_size=(IMAGE_SIZE, IMAGE_SIZE), batch_size=BATCH_SIZE, shuffle=False
          )
test_data = test_data_gen.flow_from_dataframe(
              sub, directory=TEST_DIR, x_col='image_path', y_col=None, class_mode=None,
              target_size=(IMAGE_SIZE, IMAGE_SIZE), batch_size=BATCH_SIZE, shuffle=False
          )

MAPPING = {k: i for i, k in train_data.class_indices.items()} # to decode predictions

# I wrote a child class of tf.keras.Sequential just adding a TTA prediction 
# method which averages the predictions on base image and it's flipped copy
class MySequential(tf.keras.Sequential):
    def predict_tta(self, data_generator):
        tta_preds = []
        i = 0 # counter
        images_count = data_generator.n # to be able to stop the loop
        for curr_im_batch in data_generator:
            for curr_img in curr_im_batch:
                pred = self.predict(curr_img[None, ...])
                pred_f = self.predict(np.flip(curr_img, axis=1)[None, ...])
                tta_pred = np.stack((pred, pred_f)).mean(axis=0)[0]
                tta_preds.append(tta_pred)
                
                i += 1 # for process monitoring
                if i % 1000 == 0:
                    print(f'{i}/{images_count} images processed')
            
            # now we check whether it's time to stop the loop
            # otherwise we will iterate the generator forever
            if i == images_count:
                print('Done')
                break
            
        return np.array(tta_preds)
    
# defining the model with imagenet weights makes the training process faster
model = MySequential([
    EFNETS[EFNET_NO](
        input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
        weights='imagenet',
        include_top=False),
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(N_CLASSES, activation='softmax')
])
model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss='sparse_categorical_crossentropy',
    metrics='accuracy')

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    './model_weights/model.h5', save_best_only=True, monitor='val_accuracy', mode='max', verbose=1)
lr_reducer = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_accuracy", patience=3, min_lr=3e-7, mode='max')

EPOCHS = 20
history = model.fit(
    train_data, validation_data=val_data, 
    epochs=EPOCHS, callbacks=[checkpoint, lr_reducer])