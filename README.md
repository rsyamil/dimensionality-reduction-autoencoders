# dimensionality-reduction-autoencoders

This repository contains a simple implementation of 2D convolutional autoencoders. [This Jupyter Notebook](https://github.com/rsyamil/dimensionality-reduction-autoencoders/blob/master/simple_autoencoder.ipynb) demonstrates a vanilla autoencoder (AE) and the variational (VAE) version is in [this notebook](https://github.com/rsyamil/dimensionality-reduction-autoencoders/blob/master/simple_autoencoder_variational.ipynb). For comparison purposes, dimensionality reduction with PCA is [here](https://github.com/rsyamil/dimensionality-reduction-autoencoders/blob/master/dimension_reduction_pca.ipynb). Let's reduce the dimension of the digit-MNIST dataset to latent variables of dimension three (3) and compare the image reconstructions from PCA, AE and VAE.

Once the latent space (i.e. variables represented as *z*) is constructed (i.e. learnt either using PCA, AE or VAE), we uniformly sample (i.e. 10 sample points for each dimension) all three dimensions and get the image reconstruction for each of the sampling point (each of dimension 3). There are 10x10x10 points altogether for the three dimensions and the GIFs represent the reconstructed images where the frames represent the variation in the third dimension. 

**Reconstructions with AE**
![AE](/readme/AE.gif)

**Reconstructions with VAE**
![VAE](/readme/AE_var.gif)

**Reconstructions with PCA**
![PCA](/readme/PCA.gif)

Overall, the digit-MNIST image reconstructions from the convolutional autoencoders are better than the simple PCA method (RMSE-wise). Visually, the latent spaces learnt by the autoencoders are able to represent more digits with more complex spatial features. The image reconstructions from PCA show that the types of spatial features present are limited to only features within the first three eigenspaces (i.e. large scale features). For VAE, there is a trade-off between enforcing the Gaussian prior constraint on the latent variables and the quality of image reconstruction. 

![xpl](/readme/explore_compare.png)

In this figure above, we take two image samples from the testing data set and reduce them to the latent variables. Then we uniformly sample between these two variables and proceed to obtain the image reconstructions. Again, the successive sampled points between the two images are smoother for the latent spaces constructed by the autoencoders. 

![xpl_z](/readme/explore_compare_z.png)

The figure above shows the histograms of the latent variables. For each of the dimensions, the distribution of the latent variable for VAE honors the Gaussian prior constraint. The distribution of the latent variables for AE and PCA can be multimodal or skewed as no constraint is imposed. While VAE allows random sampling from the latent variables as the distribution is known, the quality of the reconstructed image may be poorer when compared to reconstructed image from a regular autoencoder. VAE may be useful to convert high-dimensional data to low-dimensional Gaussian variables that are amenable to many existing algorithms (Kalman filter for example). 
