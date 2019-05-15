import pandas as pd
import numpy as np
import sys
from keras import backend
from keras.utils import np_utils
from sklearn.model_selection import train_test_split

desired_width = 520
pd.set_option('display.width', desired_width)

sys.path.insert(0,'keras-resnet')
from resnet import ResnetBuilder

print('Set variables')
data = pd.read_pickle('data.pkl')
nr_labels = len(data['label'].unique())
nr_samples = data.shape[0]
nr_channels = 3
nr_pixels = 224
nr_samples = len(data['image'])
# image_shape = (nr_samples, nr_pixels, nr_pixels, nr_channels)
image_shape = (nr_channels, nr_pixels, nr_pixels)
# print(data.image[0])



# set keras model
print('Build resnet and compile')
DIM_ORDERING = {'th', 'tf'}
model = ResnetBuilder.build_resnet_18(input_shape=image_shape, num_outputs=nr_labels)
# backend.set_image_dim_ordering('th')
# model.compile(optimizer='sgd')
model.compile(optimizer="sgd")
# model.compile(loss="categorical_crossentropy", optimizer="sgd")

# train test banana split
print('Create train test split')
X = np.array(data['image'].tolist())

# print(len(X[0]))
# print(len(X))
# print(X.shape)
#
# for i in X:
#     print(i)

# X = X.reshape((len(X), nr_pixels, nr_pixels, nr_channels))
# print(X[0])
y = np.array(data['label'].tolist())
# print(y)
# y = y.reshape(-1, 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
y_train = np_utils.to_categorical(y_train, nr_labels)
y_test = np_utils.to_categorical(y_test, nr_labels)
# print(X_train)
# print(Y_train)


print(X_train.shape)
print(len(X_train))

print(y_train.shape)
print(len(y_train))
# train model
print('Fitting model...')
# history = model.fit(X_train, Y_train, verbose=1) # validation_data=(X_test, Y_test))
# history = model.fit(X_train, Y_train, verbose=1) # validation_data=(X_test, Y_test))
history = model.fit(X_test, y_test, verbose=1)
# score = model.evaluate(X_test, y_test, verbose=0)
# print(score)