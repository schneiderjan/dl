import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.engine.saving import load_model

# model = load_model('models\\resnet50.hdf5')
model_history = pd.read_csv('logs\\training_resnet50.log', sep=',', engine='python')

print(model_history)

# print(history.history.keys())
# summarize history for accuracy
plt.plot(model_history['acc'])
# plt.plot(model_history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
# plt.plot(model_history['loss'])
# # plt.plot(model_history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()
# plt.savefig('plots\\test.png')