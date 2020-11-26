import numpy as np
import keras
import argparse
import os
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Conv2D, MaxPooling2D, Flatten, GlobalAveragePooling2D   # required for CNN
from keras import backend as K
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from sklearn.model_selection import train_test_split

# input args
def parse_args():
    desc="run test"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--savedir', type=str, default=None, help='/path/to/save/dir/')
    parser.add_argument('--name', type=str, default='train', help='name your session')
    parser.add_argument('--gpuid', type=str, default="1")
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--planes', type=int, default=32)
    return parser.parse_args()

# load args
args = parse_args()
save_dir = args.savedir
gpuid = args.gpuid
planes = args.planes
batch_size = args.batch_size
epochs = args.epochs
name = args.name

os.environ["CUDA_VISIBLE_DEVICES"] = gpuid

# load npys
print('... start loading data ...')
npy_dir = '/Netdata/2020/ziang/data/guangdong194/dataset/rest_25/specs/npys/'
X_train = np.load(npy_dir+'X_train.npy')
y_train = np.load(npy_dir+'y_train.npy')
X_val = np.load(npy_dir+'X_val.npy')
y_val = np.load(npy_dir+'y_val.npy')
print('... data loading finished ...')
print('shape of training data: ', X_train.shape)
print('shape of target data: ', y_train.shape)

# input image dimensions
img_rows, img_cols = 256, 512

# add an extra dimension to adapt training image to CNN
X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
X_val = X_val.reshape(X_val.shape[0], img_rows, img_cols, 1)
# X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
print(X_train.shape)

# normalize data to float in range 0..1
X_train = X_train.astype('float32')
X_val = X_val.astype('float32')
# X_train /= 255
# X_val /= 255

# convert target values to one hot vectors
y_train = keras.utils.to_categorical(y_train, 30)
y_val = keras.utils.to_categorical(y_val, 30)

# build the net
print('... constructing the net ...')
model = Sequential()
model.add(Conv2D(planes, kernel_size=(5,5),
                 activation='relu',
                 input_shape=(img_rows,img_cols,1)))
model.add(MaxPooling2D(pool_size=(2, 2),strides=2))
model.add(Dropout(0.25))

model.add(Conv2D(2*planes, (5,5),activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2),strides=2))
model.add(Dropout(0.25))

model.add(GlobalAveragePooling2D())

model.add(Dense(1024, activation='relu'))
model.add(Dense(30, activation='softmax'))

model.summary()

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])

# the model should stop training when it won't improve anymore
logdir = "../logs/%s_"%name + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=logdir)
early_stopping_monitor = EarlyStopping(monitor='val_loss', patience=20)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                              patience=10, min_lr=0.00001, verbose=1)
check_pointer = ModelCheckpoint(save_dir+'%s_ckpt_best.pkl'%name,monitor='val_accuracy',
                                                          verbose=1, save_best_only=True,
                                                          save_weights_only=True, mode='max', period=5)


history = model.fit(X_train, y_train,
                   batch_size=batch_size,
                   epochs=epochs,
                   verbose=1,
                   validation_data=(X_val, y_val),
                   callbacks=[early_stopping_monitor, lr_scheduler, check_pointer, tensorboard_callback])

# save the model
model.save_weights(save_dir + 'CNN_%d_epochs_%d_batch_Adam.h5'%(epochs, batch_size))