# Load libraries
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Input,Dense
from keras.models import Model,Sequential
from keras.datasets import mnist
 
# Load data
(X_train,_), (X_test,_)=mnist.load_data()
 
#Scaling the data to 0 and 1
X_train=np.random.rand(100, 13)
X_test=np.random.rand(100, 13)
 
#Inspect our data (The training data has 60K images while testing data has 10K images. All the sets have resolution of 28x28)
print("Training set : ",X_train.shape)
print("Testing set : ",X_test.shape)
 
#Reshaping our images into matrices
print("Training set : ",X_train.shape) #The resolution has changed
print("Testing set : ",X_test.shape)
 
# Creating an autoencoder model
input_dim=X_train.shape[1]
encoding_dim=32
compression_factor=float(input_dim/encoding_dim)
 
autoencoder=Sequential()
autoencoder.add(Dense(encoding_dim, input_shape=(input_dim,),activation='relu'))
autoencoder.add(Dense(input_dim,activation='sigmoid'))
 
input_img=Input(shape=(input_dim,))
encoder_layer=autoencoder.layers[0]
encoder=Model(input_img,encoder_layer(input_img))
 
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
autoencoder.summary()
autoencoder.fit(X_train,X_train,epochs=10, batch_size=256, shuffle=True, validation_data=(X_test,X_test))
 
# Test images and prediction
num_images=10
np.random.seed(42)
random_test_images=np.random.randint(X_test.shape[0], size=num_images)
encoded_img=encoder.predict(X_test)
decoded_img=autoencoder.predict(X_test)
 
# Save model
encoder.save('minst_encoder.h5')
autoencoder.save('minst_ae.h5')

# Display the images
#plt.figure(figsize=(18,4))
#for i, image_idx in enumerate(random_test_images):
#    #plot input image
#    ax=plt.subplot(3,num_images,i+1)
#    plt.imshow(X_test[image_idx].reshape(28,28))
#    plt.gray()
#    ax.get_xaxis().set_visible(False)
#    ax.get_yaxis().set_visible(False)
#                      
#    # plot encoded image
#    ax = plt.subplot(3, num_images, num_images + i + 1)
#    plt.imshow(encoded_img[image_idx].reshape(8, 4))
#    plt.gray()
#    ax.get_xaxis().set_visible(False)
#    ax.get_yaxis().set_visible(False)
# 
#    # plot reconstructed image
#    ax = plt.subplot(3, num_images, 2*num_images + i + 1)
#    plt.imshow(decoded_img[image_idx].reshape(28, 28))
#    plt.gray()
#    ax.get_xaxis().set_visible(False)
#    ax.get_yaxis().set_visible(False)
#                      
#plt.show()