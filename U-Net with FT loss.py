import numpy as np
import cv2
import pandas as pd

import tensorflow as tf
import random
import scipy


import matplotlib.pyplot as plt


path = "C:/Users/Yiming/Desktop/archive/kaggle_3m/"

df = pd.read_csv(path + "data.csv")
df.head(5)

from glob import glob
all_images = []

tumor_images = []
masks_images = []

non_tumor_original = []
non_tumor_masks = []


all_images = glob(path + "*/*_mask*")

for mask in all_images:
    ### if there is no tumor in the image:
    if np.all(cv2.imread(mask) == 0):
        non_tumor_masks.append(mask)
        non_tumor_original.append(mask.replace("_mask",''))
    else:
        masks_images.append(mask)
        tumor_images.append(mask.replace("_mask",''))
print("Tumor Original Images : {0}\nTumor Mask Images : {1}\nNon Tumor Original Images : {2}\nNon Tumor Mask Images : {3}".format(len(tumor_images),len(masks_images),len(non_tumor_original),len(non_tumor_masks)))

df = pd.DataFrame({'Images':tumor_images + non_tumor_original,'Masks':masks_images + non_tumor_masks})

df.head()

from sklearn.model_selection import train_test_split


train_dateset, test_dataset = train_test_split(df, test_size=0.2, random_state=42)
print(len(train_dateset))
print(len(test_dataset))

from tensorflow.keras.preprocessing.image import ImageDataGenerator

def data_generator(df, augmentation):
    image_generator = augmentation.flow_from_dataframe(
        df,
        x_col = "Images",
        target_size = (256,256),
        color_mode = "rgb",
        batch_size = 8,
        seed = 1,
        class_mode = None
    )
    mask_geneartor = augmentation.flow_from_dataframe(
        df,
        x_col = "Masks",
        target_size = (256,256),
        color_mode = "grayscale",
        batch_size = 8,
        seed = 1,
        class_mode = None
    )

    gen = zip(image_generator, mask_geneartor)

    for image, mask in gen:
        mask[mask >= 0.5] = 1.0
        mask[mask < 0.5] = 0.0
        yield image, mask


train_aug = ImageDataGenerator(
    horizontal_flip = True,
    rescale = 1./255,
    rotation_range = 0.2,
    width_shift_range=0.05,
    height_shift_range=0.05,
    shear_range=0.05,
    zoom_range=0.05
)


test_aug = ImageDataGenerator(
    horizontal_flip = True,
    rescale = 1./255
)


train_gen = data_generator(train_dateset, train_aug)
test_gen = data_generator(test_dataset, test_aug)

import keras.backend
def dice_metric(y_true, y_pred):
    smooth = 1.
    y_true_f = keras.backend.flatten(y_true)
    y_pred_f = keras.backend.flatten(y_pred)
    intersection = keras.backend.sum(y_true_f * y_pred_f)
    print(intersection)
    score = (2. * intersection + smooth) / (keras.backend.sum(y_true_f) + keras.backend.sum(y_pred_f) + smooth)

    return score

def dice_loss(y_true, y_pred):
    loss = 1 - dice_metric(y_true, y_pred)
    return loss

def iou(y_true, y_pred):
    smooth = 1.
    intersection = keras.backend.sum(y_true*y_pred)
    sum = keras.backend.sum(y_true + y_pred)
    iou = (intersection + smooth) / (sum - intersection + smooth)
    return iou


def tversky(y_true, y_pred):
    smooth = 1.
    y_true_pos = keras.backend.flatten(y_true)
    y_pred_pos = keras.backend.flatten(y_pred)
    true_pos = keras.backend.sum(y_true_pos * y_pred_pos)
    false_neg = keras.backend.sum(y_true_pos * (1 - y_pred_pos))
    false_pos = keras.backend.sum((1 - y_true_pos) * y_pred_pos)
    alpha = 0.7
    return (true_pos + smooth) / (true_pos + alpha * false_neg + (1 - alpha) * false_pos + smooth)


def focal_tversky(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    pt_1 = tversky(y_true, y_pred)
    gamma = 0.75
    return keras.backend.pow((1 - pt_1), gamma)


def tversky_loss(y_true, y_pred):
    return 1 - tversky(y_true, y_pred)


from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint\

checkpointer = ModelCheckpoint(filepath = "/kaggle/output/Unet.hdf5",
                               verbose = 1,
                               save_best_only = True)

earlystopping = EarlyStopping(monitor = "val_loss",
                              mode = "auto",
                              verbose = 1,
                              patience = 15)

reduce_lr = ReduceLROnPlateau(monitor = 'val_loss',
                              mode = 'auto',
                              verbose = 1,
                              patience = 10,
                              min_delta = 0.0001,
                              factor = 0.2)

def Unet(dims):
    inp = Input(dims)

    conv_1 = Conv2D(filters=64, kernel_size=3, padding='same', activation='relu')(inp)
    conv_1 = Conv2D(filters=64, kernel_size=3, padding='same', activation='relu')(conv_1)
    max_pool1 = MaxPool2D(pool_size=(2, 2))(conv_1)

    conv_2 = Conv2D(filters=128, kernel_size=3, padding='same', activation='relu')(max_pool1)
    conv_2 = Conv2D(filters=128, kernel_size=3, padding='same', activation='relu')(conv_2)
    bn_2 = BatchNormalization(axis=3)(conv_2)
    max_pool2 = MaxPool2D(pool_size=(2, 2))(bn_2)

    conv_3 = Conv2D(filters=256, kernel_size=3, padding='same', activation='relu')(max_pool2)
    conv_3 = Conv2D(filters=256, kernel_size=3, padding='same', activation='relu')(conv_3)
    max_pool3 = MaxPool2D(pool_size=(2, 2))(conv_3)

    conv_4 = Conv2D(filters=512, kernel_size=3, padding='same', activation='relu')(max_pool3)
    conv_4 = Conv2D(filters=512, kernel_size=3, padding='same', activation='relu')(conv_4)
    bn_4 = BatchNormalization(axis=3)(conv_4)
    max_pool4 = MaxPool2D(pool_size=(2, 2))(bn_4)

    conv_5 = Conv2D(filters=1024, kernel_size=3, padding='same', activation='relu')(max_pool4)
    conv_5 = Conv2D(filters=1024, kernel_size=3, padding='same', activation='relu')(conv_5)
    drop_5 = Dropout(0.3)(conv_5)

    up_6 = Conv2DTranspose(filters=512, kernel_size=3, strides=(2, 2), padding="same")(drop_5)
    concat_6 = concatenate([up_6, bn_4], axis=3)
    conv_6 = Conv2D(filters=512, kernel_size=3, padding='same', activation='relu')(concat_6)
    conv_6 = Conv2D(filters=512, kernel_size=3, padding='same', activation='relu')(conv_6)
    bn_6 = BatchNormalization(axis=3)(conv_6)

    up_7 = Conv2DTranspose(filters=256, kernel_size=3, strides=(2, 2), padding="same")(bn_6)
    concat_7 = concatenate([up_7, conv_3], axis=3)
    conv_7 = Conv2D(filters=256, kernel_size=3, padding='same', activation='relu')(concat_7)
    conv_7 = Conv2D(filters=256, kernel_size=3, padding='same', activation='relu')(conv_7)

    up_8 = Conv2DTranspose(filters=128, kernel_size=3, strides=(2, 2), padding="same")(conv_7)
    concat_8 = concatenate([up_8, bn_2], axis=3)
    conv_8 = Conv2D(filters=128, kernel_size=3, padding='same', activation='relu')(concat_8)
    conv_8 = Conv2D(filters=128, kernel_size=3, padding='same', activation='relu')(conv_8)
    bn_8 = BatchNormalization(axis=3)(conv_8)

    up_9 = Conv2DTranspose(filters=64, kernel_size=3, strides=(2, 2), padding="same")(bn_8)
    concat_9 = concatenate([up_9, conv_1], axis=3)
    conv_9 = Conv2D(filters=64, kernel_size=3, padding='same', activation='relu')(concat_9)
    conv_9 = Conv2D(filters=64, kernel_size=3, padding='same', activation='relu')(conv_9)

    conv_10 = Conv2D(filters=1, kernel_size=1, activation='sigmoid')(conv_9)

    return Model(inputs=inp, outputs=conv_10)


unet_f = Unet((256,256,3))
opt = tf.keras.optimizers.Adam(lr=0.0001)
unet_f.compile(optimizer = opt, loss = focal_tversky, metrics = [tversky, dice_metric, iou,'acc'])

history2 = unet_f.fit(train_gen,epochs = 100, steps_per_epoch=len(train_dateset)/8, validation_data = test_gen, validation_steps= len(test_dataset)/8, callbacks = [checkpointer, earlystopping, reduce_lr])



def show(model):
  for i in range(30):
    index=np.random.randint(1,len(test_dataset.index))
    img = cv2.imread(test_dataset['Images'].iloc[index])
    img = cv2.resize(img ,(256, 256))
    img = img / 255
    img = img[np.newaxis, :, :, :]
    pred= model.predict(img)

    plt.figure(figsize=(12,12))
    plt.subplot(1,3,1)
    plt.imshow(np.squeeze(img))
    plt.title('Original Image')
    plt.subplot(1,3,2)
    plt.imshow(np.squeeze(cv2.imread(test_dataset['Masks'].iloc[index])))
    plt.title('Original Mask')
    plt.subplot(1,3,3)
    plt.imshow(np.squeeze(pred) > .5)
    plt.title('Prediction')
    plt.show()

show(model)



def plot(train,test,title, metric):
  plt.plot(train)
  plt.plot(test)
  plt.title(title)
  plt.ylabel(metric)
  plt.xlabel('Epochs')
  plt.legend(['train', 'test'])
  plt.show()


a = history2.history

plot(a['tversky'],a['val_tversky'],"Tversky vs Epoch", "Dice_coeff")
plot(a['dice_metric'],a['val_dice_metric'],"Dice_coeff vs Epoch", "Dice_coeff")
plot(a['iou'],a['val_iou'],"IOU vs Epoch", "iou")
plot(a['loss'],a['val_loss'],"Dice Loss vs Epoch", "Dice Loss")