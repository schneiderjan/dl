from keras.models import Sequential
from keras.layers import Dense

import sys
sys.path.insert(0,'keras-resnet')
from resnet import ResnetBuilder

image_shape = (3,224,224)
nr_labels = 31

b = ResnetBuilder(input_shape=image_shape, num_outputs=nr_labels, block_fn='bottleneck')
print(b)




# model = Sequential()
# model.add(Dense(units=64, activation='relu', input_dim=100))
# model.compile()
b.build_resnet_18()


