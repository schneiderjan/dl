import numpy as np
import pandas as pd
import keras_resnet.models
from keras.layers import Input
from keras.utils import np_utils
from keras.callbacks import CSVLogger
from sklearn.model_selection import train_test_split

desired_width = 520
pd.set_option('display.width', desired_width)


print('Set variables')
data = pd.read_pickle('data.pkl')
nr_labels = len(data['label'].unique())
nr_samples = data.shape[0]
nr_channels = 3
nr_pixels = 224
image_shape = (nr_pixels, nr_pixels, nr_channels)

input_shape = Input(image_shape)

# train test banana split
print('Create train test split')
X = np.array(data['image'].tolist())
y = np.array(data['label'].tolist())
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, train_size=10)
Y_train = np_utils.to_categorical(Y_train, nr_labels)
Y_test = np_utils.to_categorical(Y_test, nr_labels)

# train model
print('Create ResNet models')
nr_epochs = 40

model_50 = keras_resnet.models.ResNet50(input_shape, classes=nr_labels)
model_50.compile("adam", "categorical_crossentropy", ["accuracy"])
csv_logger = CSVLogger('logs\\training_resnet50.log', separator=',', append=False)
history = model_50.fit(X_train, Y_train, verbose=1, callbacks=[csv_logger], epochs=nr_epochs)
score = model_50.evaluate(X_test, Y_test, verbose=0)
print('ResNet50 score:')
print(score)

print('Saving ResNet50 to hdf5.')
model_50.save('models\\resnet50.hdf5')