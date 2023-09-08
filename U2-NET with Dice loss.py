import numpy as np
import cv2
import pandas as pd

import tensorflow as tf
import random
import scipy

from tensorflow.keras.layers import *
from tensorflow.keras.models import Model

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
from keras.layers import *
from keras.models import *

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

checkpointer = ModelCheckpoint(filepath = "./U2net-focal_2.hdf5",
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


def Conv(x, out_ch=3, rate=1):
    x = layers.Conv2D(out_ch, 3, padding='same', dilation_rate=1 * rate)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    return x


def upsampling(src, tar):
    h = int(tar.shape[1] // src.shape[1])
    w = int(tar.shape[2] // src.shape[2])
    src = layers.UpSampling2D((h, w), interpolation='nearest')(src)
    return src


def en_1(x, mid_ch=12, out_ch=3):
    x0 = Conv(x, out_ch, 1)
    x1 = Conv(x0, mid_ch, 1)

    x = MaxPool2D(2, 2)(x1)
    x2 = Conv(x, mid_ch, 1)

    x = MaxPool2D(2, 2)(x2)
    x3 = Conv(x, mid_ch, 1)

    x = MaxPool2D(2, 2)(x3)
    x4 = Conv(x, mid_ch, 1)

    x = MaxPool2D(2, 2)(x4)
    x5 = Conv(x, mid_ch, 1)

    x = MaxPool2D(2, 2)(x5)
    x6 = Conv(x5, mid_ch, 1)

    x = Conv(x6, mid_ch, 2)

    x = Conv(tf.concat([x, x6], axis=-1), mid_ch, 1)

    x = upsampling(x, x5)
    x = Conv(tf.concat([x, x5], axis=-1), mid_ch, 1)

    x = upsampling(x, x4)
    x = Conv(tf.concat([x, x4], axis=-1), mid_ch, 1)

    x = upsampling(x, x3)
    x = Conv(tf.concat([x, x3], axis=-1), mid_ch, 1)

    x = upsampling(x, x2)
    x = Conv(tf.concat([x, x2], axis=-1), mid_ch, 1)

    x = upsampling(x, x1)
    x = Conv(tf.concat([x, x1], axis=-1), out_ch, 1)

    return x + x0


def en_2(x, mid_ch=12, out_ch=3):
    x0 = Conv(x, out_ch, 1)
    x1 = Conv(x0, mid_ch, 1)

    x = MaxPool2D(2, 2)(x1)
    x2 = Conv(x, mid_ch, 1)

    x = MaxPool2D(2, 2)(x2)
    x3 = Conv(x, mid_ch, 1)

    x = MaxPool2D(2, 2)(x3)
    x4 = Conv(x, mid_ch, 1)

    x = MaxPool2D(2, 2)(x4)
    x5 = Conv(x, mid_ch, 1)

    x = Conv(x, mid_ch, 2)

    x = Conv(tf.concat([x, x5], axis=-1), mid_ch, 1)

    x = upsampling(x, x4)
    x = Conv(tf.concat([x, x4], axis=-1), mid_ch, 1)

    x = upsampling(x, x3)
    x = Conv(tf.concat([x, x3], axis=-1), mid_ch, 1)

    x = upsampling(x, x2)
    x = Conv(tf.concat([x, x2], axis=-1), mid_ch, 1)

    x = upsampling(x, x1)
    x = Conv(tf.concat([x, x1], axis=-1), out_ch, 1)

    return x + x0


def en_3(x, mid_ch=12, out_ch=3):
    x0 = Conv(x, out_ch, 1)
    x1 = Conv(x0, mid_ch, 1)

    x = MaxPool2D(2, 2)(x1)
    x2 = Conv(x, mid_ch, 1)

    x = MaxPool2D(2, 2)(x2)
    x3 = Conv(x, mid_ch, 1)

    x = MaxPool2D(2, 2)(x3)
    x4 = Conv(x, mid_ch, 1)

    x = Conv(x, mid_ch, 2)

    x = Conv(tf.concat([x, x4], axis=-1), mid_ch, 1)

    x = upsampling(x, x3)
    x = Conv(tf.concat([x, x3], axis=-1), mid_ch, 1)

    x = upsampling(x, x2)
    x = Conv(tf.concat([x, x2], axis=-1), mid_ch, 1)

    x = upsampling(x, x1)
    x = Conv(tf.concat([x, x1], axis=-1), out_ch, 1)

    return x + x0


def en_4(x, mid_ch=12, out_ch=3):
    x0 = Conv(x, out_ch, 1)

    x1 = Conv(x0, mid_ch, 1)
    x = MaxPool2D(2, 2)(x1)

    x2 = Conv(x, mid_ch, 1)
    x = MaxPool2D(2, 2)(x2)

    x3 = Conv(x, mid_ch, 1)

    x = Conv(x, mid_ch, 2)

    x = Conv(tf.concat([x, x3], axis=-1), mid_ch, 1)
    x = upsampling(x, x2)

    x = Conv(tf.concat([x, x2], axis=-1), mid_ch, 1)
    x = upsampling(x, x1)

    x = Conv(tf.concat([x, x1], axis=-1), out_ch, 1)

    return x + x0


def en_5(x, mid_ch=12, out_ch=3):
    x0 = Conv(x, out_ch, 1)

    x1 = Conv(x0, mid_ch, 1)
    x2 = Conv(x1, mid_ch, 2)
    x3 = Conv(x2, mid_ch, 4)

    x4 = Conv(x3, mid_ch, 8)

    x = Conv(tf.concat([x4, x3], axis=-1), mid_ch, 4)
    x = Conv(tf.concat([x, x2], axis=-1), mid_ch, 2)
    x = Conv(tf.concat([x, x1], axis=-1), out_ch, 1)

    return x + x0


def U2NETP(x, out_ch=1):
    x1 = en_1(x, 16, 64)
    x = layers.MaxPool2D(2, 2)(x1)

    x2 = en_2(x, 16, 64)
    x = layers.MaxPool2D(2, 2)(x2)

    x3 = en_3(x, 16, 64)
    x = layers.MaxPool2D(2, 2)(x3)

    x4 = en_4(x, 16, 64)
    x = layers.MaxPool2D(2, 2)(x4)

    x5 = en_5(x, 16, 64)
    x = layers.MaxPool2D(2, 2)(x5)

    x6 = en_5(x, 16, 64)
    x = upsampling(x6, x5)

    # decoder

    x5 = en_5(tf.concat([x, x5], axis=-1), 16, 64)
    x = upsampling(x5, x4)

    x4 = en_4(tf.concat([x, x4], axis=-1), 16, 64)
    x = upsampling(x4, x3)

    x3 = en_3(tf.concat([x, x3], axis=-1), 16, 64)
    x = upsampling(x3, x2)

    x2 = en_2(tf.concat([x, x2], axis=-1), 16, 64)
    x = upsampling(x2, x1)

    x1 = en_1(tf.concat([x, x1], axis=-1), 16, 64)

    x = layers.ZeroPadding2D((1, 1))(x1)
    s1 = layers.Conv2D(out_ch, 3)(x)
    s1 = layers.Activation('sigmoid')(s1)

    x = layers.ZeroPadding2D((1, 1))(x2)
    x = layers.Conv2D(out_ch, 3)(x)
    s2 = upsampling(x, s1)
    s2 = layers.Activation('sigmoid')(s2)

    x = layers.ZeroPadding2D((1, 1))(x3)
    x = layers.Conv2D(out_ch, 3)(x)
    s3 = upsampling(x, s1)
    s3 = layers.Activation('sigmoid')(s3)

    x = layers.ZeroPadding2D((1, 1))(x4)
    x = layers.Conv2D(out_ch, 3)(x)
    s4 = upsampling(x, s1)
    s4 = layers.Activation('sigmoid')(s4)

    x = layers.ZeroPadding2D((1, 1))(x5)
    x = layers.Conv2D(out_ch, 3)(x)
    s5 = upsampling(x, s1)
    s5 = layers.Activation('sigmoid')(s5)

    x = layers.ZeroPadding2D((1, 1))(x6)
    x = layers.Conv2D(out_ch, 3)(x)
    s6 = upsampling(x, s1)
    s6 = layers.Activation('sigmoid')(s6)

    s0 = layers.Conv2D(out_ch, 1)(tf.concat([s1, s2, s3, s4, s5, s6], axis=-1))
    s0 = layers.Activation('sigmoid')(s0)

    return s0

from tensorflow.keras import layers

uinput = Input(shape=(256,256,3))
uoutput = U2NETP(uinput)
model = Model(inputs = uinput, outputs = uoutput)
opt =  tf.keras.optimizers.Adam(lr=0.0001)
model.compile(optimizer = opt, loss = dice_loss, metrics= [dice_metric, iou,'acc'])
model.summary()

history = model.fit(train_gen,epochs = 100, steps_per_epoch=len(train_dateset)/8, validation_data = test_gen, validation_steps= len(test_dataset)/8, callbacks = [checkpointer, earlystopping, reduce_lr])


def show(model):
    for i in range(30):
        index = np.random.randint(1, len(test_dataset.index))
        img = cv2.imread(test_dataset['Images'].iloc[index])
        img = cv2.resize(img, (256, 256))
        img = img / 255.0
        img = img[np.newaxis, :, :, :]
        label = cv2.imread(test_dataset['Masks'].iloc[index], cv2.IMREAD_GRAYSCALE)
        label = cv2.resize(label, (256, 256))
        label = label / 255.0
        label = label[np.newaxis, :, :]

        pred = model.predict(img)

        count = (pred > 1).sum()
        print(pred)
        print(pred.shape)
        print(label.shape)
        print("The dice is: ")
        print(dice_metric(tf.cast(pred, dtype=np.float32), tf.cast(label, dtype=np.float32)))
        print("finished")
        print(count)

        plt.figure(figsize=(12, 12))
        plt.subplot(1, 3, 1)
        plt.imshow(np.squeeze(img))
        plt.title('Original Image')
        plt.subplot(1, 3, 2)
        plt.imshow(np.squeeze(cv2.imread(test_dataset['Masks'].iloc[index])))
        plt.title('Original Mask')
        plt.subplot(1, 3, 3)
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

  a = history.history

  plot(a['dice_metric'], a['val_dice_metric'], "Dice_coeff vs Epoch", "Dice_coeff")
  plot(a['iou'], a['val_iou'], "IOU vs Epoch", "iou")
  plot(a['loss'], a['val_loss'], "Dice Loss vs Epoch", "Dice Loss")