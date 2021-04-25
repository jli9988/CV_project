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

# train_images = []
# train_poses = []
# labels = []
# filelist1 = sorted(glob.glob("~/*.png"))
# filelist2 = sorted(glob.glob("~/*.png"))
# filelist3 = sorted(glob.glob("~/*.png"))
#
# # load data
# for fname1, fname2 in zip(filelist1, filelist3):
#     im = cv2.imread(fname1, 1)
#     im = np.array(im)
#     pose = cv2.imread(fname2, 1)
#     pose = np.array(pose)
#     train_images.append(im)
#     train_poses.append(pose)
#     lb = np.array([1, 0])
#     labels.append(lb)
#
# for fname1, fname2 in zip(filelist2, filelist3):
#     im = cv2.imread(fname1, 1)
#     im = np.array(im)
#     pose = cv2.imread(fname2, 1)
#     pose = np.array(pose)
#     train_images.append(im)
#     train_poses.append(pose)
#     lb = np.array([0, 1])
#     labels.append(lb)
#
# train_images = np.asarray(train_images)/255.0
# train_poses = np.asarray(train_poses)/255.0

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
image_shape = (256, 256, 3)
batch_size = 1
epochs = 140
# create the model
model = define_discriminator(image_shape)
# summarize the model
model.summary()
# plot the model
plot_model(model, to_file='discriminator_model_plot.png', show_shapes=True, show_layer_names=True)

history_callback = model.fit([train_poses, train_images], labels, batch_size=batch_size, epochs=epochs, shuffle=True)
training_loss = history_callback.history['loss']

loss_history = np.array(training_loss)
np.savetxt('loss.txt', loss_history, delimiter=",")
model.save_weights('model.h5')

