import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
from keras.callbacks import History
from keras.utils import plot_model

history_path = './model_history'

# Model accuracy and loss plots
def plot_model_history(model_details):
    # Create sub-plots
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))

    # Summarize history for accuracy
    # axs[0].plot(range(1, len(model_details.history['acc']) + 1),
    #             model_details.history['acc'])
    # axs[0].plot(range(1, len(model_details.history['val_acc']) + 1),
    #             model_details.history['val_acc'])
    # axs[0].set_title('Model Accuracy')
    # axs[0].set_ylabel('Accuracy')
    # axs[0].set_xlabel('Epoch')
    # axs[0].set_xticks(np.arange(1, len(model_details.history['acc']) + 1),
    #                   len(model_details.history['acc']) / 10)
    # axs[0].legend(['train', 'val'], loc='best')

    # Summarize history for loss
    axs[0].plot(range(1, len(model_details.history['loss']) + 1),
                model_details.history['loss'])
    # axs[0].plot(range(1, len(model_details.history['val_loss']) + 1),
    #             model_details.history['val_loss'])
    axs[0].set_title('Model Loss')
    axs[0].set_ylabel('Loss')
    axs[0].set_xlabel('Epoch')
    axs[0].set_xticks(np.arange(1, len(model_details.history['loss']) + 1),
                      len(model_details.history['loss']) / 10)
    axs[0].legend(['train', 'loss'], loc='best')
    plt.show()


# Plot saved model history
if os.path.exists(history_path):
    # Save model history
    with open(history_path, 'rb') as file_pi:
        model_history = History()
        model_history.history = pickle.load(file_pi)
        plot_model_history(model_history)
else:
    raise ValueError('No model history found')

# plot model by keras. pydot is required
# model = im.get_model()
# plot_model(model, to_file='./Doc/model.png')
