from keras.optimizers import Adam
from keras.initializers import RandomNormal
from keras.models import Model
from keras.models import Input
from keras.layers import Conv2D
from keras.layers import LeakyReLU
from keras.layers import Activation
from keras.layers import Concatenate
from keras.layers import BatchNormalization
from keras.layers import MaxPooling2D, Flatten, Dense
from keras.utils.vis_utils import plot_model
import cv2
import numpy as np
import glob

test_images = []
test_poses = []
filelist1 = sorted(glob.glob("/content/test_data/real_image/*.png"))
filelist2 = sorted(glob.glob("/content/test_data/real_label/*.png"))
filelist3 = sorted(glob.glob("/content/test_data/syn_image/*.png"))
filelist4 = sorted(glob.glob("/content/test_data/syn_label/*.png"))

# load data
for fname1, fname2 in zip(filelist1, filelist2):
    im = cv2.imread(fname1, 1)
    im = np.array(im)
    im = im / np.amax(im)
    pose = cv2.imread(fname2, 1)
    pose = np.array(pose)
    pose = pose / np.amax(pose)
    test_images.append(im)
    test_poses.append(pose)

for fname1, fname2 in zip(filelist3, filelist4):
    im = cv2.imread(fname1, 1)
    im = np.array(im)
    im = im / np.amax(im)
    pose = cv2.imread(fname2, 1)
    pose = np.array(pose)
    pose = pose / np.amax(pose)
    test_images.append(im)
    test_poses.append(pose)

test_images = np.asarray(train_images)
test_poses = np.asarray(train_poses)

# define the discriminator model
def define_discriminator(image_shape):
    # weight initialization
    init = RandomNormal(stddev=0.02)
    # source image input
    in_src_image = Input(shape=image_shape)
    # target image input
    in_target_image = Input(shape=image_shape)
    # concatenate images channel-wise
    merged = Concatenate()([in_src_image, in_target_image])
    # C64
    d = Conv2D(64, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(merged)
    d = LeakyReLU(alpha=0.2)(d)
    # C128
    d = Conv2D(128, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    # C256
    d = Conv2D(256, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    # C512
    d = Conv2D(512, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    # second last output layer
    d = Conv2D(512, (4, 4), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    # patch output
    d = Conv2D(1, (4, 4), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    # pooling
    d = MaxPooling2D(pool_size=(2, 2), strides=None, padding="valid")(d)
    d = Flatten()(d)
    d = Dense(2, activation=None)(d)
    pred_out = Activation('sigmoid')(d)
    # define model
    model = Model([in_src_image, in_target_image], pred_out)
    # compile model
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt, loss_weights=[0.5])
    return model

# define image shape
image_shape = (512, 512, 3)
# create the model
model = define_discriminator(image_shape)

# load pre-trained weights
model.load_weights('~/model.h5')

pred_probs = []

for image, pose in zip(test_images, test_poses):
    pred_out = model.predict([pose, image])
    pred_probs.append(pred_out)

pred_probs = np.asarray(pred_probs)
np.savetxt('pred_probs.txt', pred_probs)

