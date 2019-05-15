import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

model_names = ['ResNet18', 'ResNet34', 'ResNet50']#, 'ResNet101', 'ResNet152', 'ResNet200']

# plot acc
for name in model_names:
    log_file = 'logs\\{}.log'.format(name)
    model_history = pd.read_csv(log_file, sep=',')

    plt.plot(model_history['acc'])
    # plt.plot(model_history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(model_names, loc='upper left')
    plt.savefig('plots\\acc_{}.png'.format(name))

plt.clf()

# plot loss
for name in model_names:
    log_file = 'logs\\{}.log'.format(name)
    model_history = pd.read_csv(log_file, sep=',')

    plt.plot(model_history['loss'])
    # plt.plot(model_history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(model_names, loc='upper left')
    plt.savefig('plots\\loss_{}.png'.format(name))

plt.clf()

# bat plot acc
x = np.arange(len(model_names))
accuracies = []
for name in model_names:
    log_file = 'logs\\{}.log'.format(name)
    model_history = pd.read_csv(log_file, sep=',')
    accuracies.append(model_history.at[39, 'acc'])
fig, ax = plt.subplots()
plt.bar(x, accuracies)
plt.xticks(x, model_names)
plt.savefig('plots\\bar_acc.png')
