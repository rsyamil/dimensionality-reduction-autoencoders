import numpy as np
import util
import keras
from keras.models import Model
from keras.layers import Layer, Flatten, LeakyReLU
from keras.layers import Input, Reshape, Dense, Lambda
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D

from keras import backend as K
from keras.engine.base_layer import InputSpec

from keras.optimizers import Adam, SGD, RMSprop
from keras.layers.normalization import BatchNormalization
from keras.losses import mse, binary_crossentropy
from keras import regularizers, activations, initializers, constraints
from keras.constraints import Constraint
from keras.callbacks import History, EarlyStopping

from keras.utils import plot_model
from keras.models import load_model

from keras.utils.generic_utils import get_custom_objects

def sampling(args):
    
    epsilon_std = 1.0
    
    z_mean, z_log_sigma = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    
    epsilon = K.random_normal(shape=(batch, dim),
                              mean=0., stddev=epsilon_std)
    return z_mean + K.exp(z_log_sigma) * epsilon

class Autoencoder:

    def __init__(self, x_sz, y_sz, z_dim= 64, variational = False, name=[]):
        self.name = name
        self.x_sz = x_sz
        self.y_sz = y_sz
        self.AE_m2m = []
        self.AE_m2z = []
        self.AE_z2m = []
        self.variational = variational
        self.z_dim = z_dim
        
        self.AE_m2m_reg = []

    def encoder2D(self):
        #define the simple autoencoder
        input_image = Input(shape=(self.x_sz, self.y_sz, 1)) 

        #image encoder
        _ = Conv2D(4, (3, 3), padding='same', name='enc')(input_image)
        _ = LeakyReLU(alpha=0.3)(_)
        _ = MaxPooling2D((2, 2))(_)

        _ = Conv2D(8, (4, 4), padding='same')(_)
        _ = BatchNormalization()(_)
        _ = LeakyReLU(alpha=0.3)(_)
        _ = MaxPooling2D((2, 2))(_)

        _ = Conv2D(16, (5, 5), padding='same')(_)
        _ = BatchNormalization()(_)
        _ = LeakyReLU(alpha=0.3)(_)
        _ = MaxPooling2D((2, 2))(_)

        _ = Flatten()(_)

        if not self.variational:
            encoded_image = Dense(self.z_dim)(_)
        else:
            _ = Dense(self.z_dim)(_)
            z_mean_m = Dense(self.z_dim)(_)
            z_log_var_m = Dense(self.z_dim)(_)
            encoded_image = Lambda(sampling)([z_mean_m, z_log_var_m])
            return input_image, encoded_image, z_mean_m, z_log_var_m

        return input_image, encoded_image

    def decoder2D(self, encoded_image):
        #image decoder
        _ = Dense((256))(encoded_image)
        _ = Reshape((4, 4, 16))(_)

        _ = Conv2D(16, (5, 5), padding='same')(_)
        _ = BatchNormalization()(_)
        _ = LeakyReLU(alpha=0.3)(_)
        _ = UpSampling2D((2, 2))(_)

        _ = Conv2D(8, (4, 4), padding='same')(_)
        _ = BatchNormalization()(_)
        _ = LeakyReLU(alpha=0.3)(_)
        _ = UpSampling2D((2, 2))(_)

        _ = Conv2D(4, (3, 3))(_)
        _ = LeakyReLU(alpha=0.3)(_)
        _ = UpSampling2D((2, 2))(_)

        decoded_image = Conv2D(1, (3, 3), padding='same')(_)

        return decoded_image
        
    def train_autoencoder2D_dual(self, x_train_reg, x_train, load = False):
        #autoencoder for regression
        input_image_reg, encoded_image_reg = self.encoder2D()
        decoded_image_reg = self.decoder2D(encoded_image_reg)
        
        self.AE_m2m_reg = Model(input_image_reg, decoded_image_reg)
        opt = keras.optimizers.Adam(lr=1e-3)
        self.AE_m2m_reg.compile(optimizer=opt, 
                        loss="mse", 
                        metrics=['mse'])
                        
        self.AE_m2m_reg.summary()
        
        #autoencoder for reconstruction
        input_image, encoded_image = self.encoder2D()
        decoded_image = self.decoder2D(encoded_image)
        
        self.AE_m2m = Model(input_image, decoded_image)
        opt = keras.optimizers.Adam(lr=1e-3)
        self.AE_m2m.compile(optimizer=opt, 
                        loss="mse", 
                        metrics=['mse'])
        
        self.AE_m2m.summary()
        
        #train the neural network alternatingly
        totalEpoch = 150
        plot_losses1 = util.PlotLosses()
        plot_losses2 = util.PlotLosses()
        history1 = History()
        history2 = History()
        
        AE_reg = np.zeros([totalEpoch, 4])
        AE = np.zeros([totalEpoch, 4])
        
        for i in range(totalEpoch):
            #train main reg model
            self.AE_m2m_reg.fit(x_train_reg, x_train,        
                        epochs=1,
                        batch_size=128,
                        shuffle=True,
                        validation_split=0.2,
                        callbacks=[plot_losses1, EarlyStopping(monitor='loss', patience=60), history1])
            #copy loss
            AE_reg[i, :] = np.squeeze(np.asarray(list(history1.history.values())))
            
            #copy weights from the reg model to the recons model
            copy_idxs = range(14, 27+1)
            for c in copy_idxs:
                self.AE_m2m.layers[c].set_weights(self.AE_m2m_reg.layers[c].get_weights())
         
            #train model recons AE
            self.AE_m2m.fit(x_train, x_train,        
                        epochs=1,
                        batch_size=128,
                        shuffle=True,
                        validation_split=0.2,
                        callbacks=[plot_losses2, EarlyStopping(monitor='loss', patience=60), history2])
            
            #copy recons into the reg model 
            for c in copy_idxs:
                self.AE_m2m_reg.layers[c].set_weights(self.AE_m2m.layers[c].get_weights())
                
            #copy loss
            AE[i, :] = np.squeeze(np.asarray(list(history2.history.values())))
            
            #write to folder for every 10th epoch for monitoring
            figs = util.plotAllLosses(AE_reg, AE)
            figs.savefig('Dual_Losses.png')
        
    def train_autoencoder2D(self, x_train_reg, x_train, load = False):
        #set loss function, optimizer and compile
        input_image, encoded_image = self.encoder2D()
        decoded_image = self.decoder2D(encoded_image)

        self.AE_m2m = Model(input_image, decoded_image)
        opt = keras.optimizers.Adam(lr=1e-3)
        self.AE_m2m.compile(optimizer=opt, 
                        loss="mse", 
                        metrics=['mse'])

        #get summary of architecture parameters and plot arch. diagram
        self.AE_m2m.summary()
        plot_model(self.AE_m2m, to_file='AE_m2m.png')

        #train the neural network
        if not load:
            plot_losses = util.PlotLosses()
            self.AE_m2m.fit(x_train_reg, x_train,        
                            epochs=100,
                            batch_size=32,
                            shuffle=True,
                            validation_split=0.3,
                            callbacks=[plot_losses])
            #save trained model
            self.AE_m2m.save('AE_m2m.h5')
        else:
            #load an already trained model
            #some bug here, self.AE_m2z and self.AE_z2m not working
            #when model is loaded.
            print("Trained model loaded")
            self.AE_m2m = load_model('AE_m2m.h5')

        #set the encoder model
        self.AE_m2z = Model(input_image, encoded_image)

        #set the decoder model
        zm_dec = Input(shape=(self.z_dim, )) 
        _ = self.AE_m2m.layers[14](zm_dec)
        for i in range(15, 27):
            _ = self.AE_m2m.layers[i](_)
        decoded_image_ = self.AE_m2m.layers[27](_)
        self.AE_z2m = Model(zm_dec, decoded_image_)

    def train_var_autoencoder2D(self, x_train, load = False):
        #set loss function, optimizer and compile
        input_image, encoded_image, z_mean, z_log_var = self.encoder2D()
        decoded_image = self.decoder2D(encoded_image)

        #define the variational loss and mse loss (equal weighting)
        def vae_loss(input_image, decoded_image):
            recons_loss = K.sum(mse(input_image, decoded_image))                
            kl_loss = (- 0.5) * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
            return K.mean(recons_loss + kl_loss)

        #add custom loss 
        get_custom_objects().update({"vae_loss": vae_loss})

        self.AE_m2m = Model(input_image, decoded_image)
        opt = keras.optimizers.Adam(lr=1e-3)
        self.AE_m2m.compile(optimizer=opt, 
                        loss=vae_loss)

        #get summary of architecture parameters and plot arch. diagram
        self.AE_m2m.summary()
        plot_model(self.AE_m2m, to_file='AE_m2m_var.png')

        #train the neural network
        if not load:
            plot_losses = util.PlotLosses()
            self.AE_m2m.fit(x_train, x_train,        
                            epochs=100,
                            batch_size=32,
                            shuffle=True,
                            validation_split=0.3,
                            callbacks=[plot_losses])
            #save trained model
            self.AE_m2m.save('AE_m2m_var.h5')
        else:
            #load an already trained model
            print("Trained model loaded")
            self.AE_m2m = load_model('AE_m2m_var.h5')

        #set the encoder model
        self.AE_m2z = Model(input_image, encoded_image)

        #set the decoder model
        zm_dec = Input(shape=(self.z_dim, )) 
        _ = self.AE_m2m.layers[17](zm_dec)
        for i in range(18, 30):
            _ = self.AE_m2m.layers[i](_)
        decoded_image_ = self.AE_m2m.layers[30](_)
        self.AE_z2m = Model(zm_dec, decoded_image_)


