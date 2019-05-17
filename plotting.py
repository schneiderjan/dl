import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import re


df = pd.read_csv('logs\\test_accuracy.csv')
test_means = df.groupby(by='model').mean().reset_index()
# test_means['layers'] = test_means['model'].replace('ResNet','')
test_means.loc[len(test_means)+1] = ['ResNet26 (Sun et al.)', -1, 0.9965]

for idx, row in test_means.iterrows():
    test_means.at[idx, 'layers'] = str( re.sub("[^0-9]", "", row['model']))
test_means.at[6, 'layers'] = '26 (Sun et al.)'
test_means = test_means.sort_values(by=['acc'], ascending=False)


model_names = ['ResNet18', 'ResNet34', 'ResNet50', 'ResNet101', 'ResNet152']

# plot acc
for name in model_names:
    log_file = 'logs\\best\\{}.log'.format(name)
    model_history = pd.read_csv(log_file, sep=',')

    plt.plot(model_history['acc'])
    # plt.plot(model_history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(model_names, loc='lower right')
    plt.savefig('plots\\acc_epoch.png'.format(name))

plt.clf()
#
# # plot loss
# for name in model_names:
#     log_file = 'logs\\best\\{}.log'.format(name)
#     model_history = pd.read_csv(log_file, sep=',')
#
#     plt.plot(model_history['loss'])
#     # plt.plot(model_history['val_loss'])
#     plt.title('model loss')
#     plt.ylabel('Loss')
#     plt.xlabel('Epoch')
#     plt.legend(model_names, loc='upper left')
#     plt.savefig('plots\\loss_{}.png'.format(name))
#
# plt.clf()
#
# bat plot acc
x = np.arange(len(test_means))
accuracies = test_means.acc.tolist()
layers = test_means.layers.tolist()
fig, ax = plt.subplots()
plt.bar(x, accuracies)
plt.xticks(x, layers)
plt.ylabel('Accuracy')
plt.xlabel('Layers')
plt.savefig('plots\\bar_acc.png')
