import keras
from IPython.display import clear_output
import matplotlib.pyplot as plt

#function to view training and validation losses
class PlotLosses(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        self.fig = plt.figure()
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.i += 1
        clear_output(wait=True)
        plt.plot(self.x, self.losses, label="loss", c = 'green')
        plt.plot(self.x, self.val_losses, label="val_loss", c = 'red')
        plt.legend()
        plt.show()
        
#function to view multiple losses
def plotAllLosses(loss1, loss2):         
    N, m1f = loss1.shape
    _, m2f = loss2.shape
    
    print(loss1.shape)
    print(loss2.shape)
    
    fig = plt.figure(figsize=(6, 12))
    plt.subplot(2, 1, 1)
    plt.plot(loss1[:, 0], label='loss1_check1')
    plt.plot(loss1[:, 1], label='loss1_check2')
    plt.plot(loss1[:, 2], label='loss1_check3')
    plt.plot(loss1[:, 3], label='loss1_check4')
    plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.plot(loss2[:, 0], label='loss2_check1')
    plt.plot(loss2[:, 1], label='loss2_check2')
    plt.plot(loss2[:, 2], label='loss2_check3')
    plt.plot(loss2[:, 3], label='loss2_check4')
    plt.legend()
    
    return fig
