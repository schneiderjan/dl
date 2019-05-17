import numpy as np
import pandas as pd
import keras_resnet.models
from keras.layers import Input
from keras.utils import np_utils
from keras.callbacks import CSVLogger
from keras.optimizers import SGD
from sklearn.model_selection import train_test_split

desired_width = 520
pd.set_option('display.width', desired_width)


print('Set variables')
data = pd.read_pickle('data.pkl')
nr_labels = len(data['label'].unique())
nr_samples = data.shape[0]
print(nr_samples)
nr_channels = 3
nr_pixels = 224
image_shape = (nr_pixels, nr_pixels, nr_channels)
input_shape = Input(image_shape)

# train test banana split
print('Create train test split')
X = np.array(data['image'].tolist())
y = np.array(data['label'].tolist())
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2)
Y_train = np_utils.to_categorical(Y_train, nr_labels)
Y_test = np_utils.to_categorical(Y_test, nr_labels)

# train model
print('Create ResNet models')
nr_epochs = 40
learning_rate = 0.001
momentum = 0.9
decay_rate = learning_rate / nr_epochs
sgd = SGD(lr=learning_rate, decay=1e-6, momentum=momentum)

# model_18 = keras_resnet.models.ResNet18(input_shape, classes=nr_labels)
# model_34 = keras_resnet.models.ResNet34(input_shape, classes=nr_labels)
# model_50 = keras_resnet.models.ResNet50(input_shape, classes=nr_labels)
# model_101 = keras_resnet.models.ResNet101(input_shape, classes=nr_labels)
# model_152 = keras_resnet.models.ResNet152(input_shape, classes=nr_labels)
# model_202 = keras_resnet.models.ResNet200(input_shape, classes=nr_labels)
#
# models = [model_18, model_34, model_50, model_101, model_152, model_202]
# model_names = ['ResNet18', 'ResNet34', 'ResNet50', 'ResNet101', 'ResNet152', 'ResNet200']
#
# print('Begin training')
# for idx, model in enumerate(models):
#     log_dir = 'logs\\{}.log'.format(model_names[idx])
#     model_dir = 'models\\{}.hdf5'.format(model_names[idx])
#
#     model.compile(optimizer=sgd, loss="categorical_crossentropy", metrics=["accuracy"])
#     csv_logger = CSVLogger(log_dir, separator=',', append=False)
#     history = model.fit(X_train, Y_train, verbose=1, callbacks=[csv_logger], epochs=nr_epochs)
#     score = model.evaluate(X_test, Y_test, verbose=0)
#     model.save(model_dir)
