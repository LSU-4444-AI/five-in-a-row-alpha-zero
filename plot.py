import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
from keras.callbacks import History
from policy_value_net_keras import his_path
from keras.utils import plot_model


# Model accuracy and loss plots
def plot_model_history(model_details):
    # Create sub-plots
    # fig, axs = plt.subplots(1, 2, figsize=(15, 5))

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
    # axs[1].plot(range(1, len(model_details.history['loss']) + 1),
    #             model_details.history['loss'])
    # axs[0].plot(range(1, len(model_details.history['val_loss']) + 1),
    #             model_details.history['val_loss'])
    # axs[1].set_title('Model Loss')
    # axs[1].set_ylabel('Loss')
    # axs[1].set_xlabel('Epoch')
    # axs[1].set_xticks(np.arange(1, len(model_details.history['loss']) + 1),
    #                   len(model_details.history['loss']) / 10)
    # axs[1].legend(['train', 'loss'], loc='best')

    # Summarize history for loss
    plt.plot(range(1, len(model_details.history['loss']) + 1),
             model_details.history['loss'])
    # axs[0].plot(range(1, len(model_details.history['val_loss']) + 1),
    #             model_details.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    # plt.xticks(np.arange(1, len(model_details.history['loss']) + 1),
    #            len(model_details.history['loss']) / 10)
    plt.legend(['train', 'loss'], loc='best')
    plt.show()


def plot(his_paths):
    # Plot saved model history
    for history_path in his_paths:
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


if __name__ == '__main__':
    plot(['./saved/model_6307_11782_history',
          './saved/model_22221_27518_history'])
