
from keras.models import Model
from keras.layers.core import Dropout, Activation
from keras.layers.pooling import MaxPooling2D
from keras.layers import Input, Conv2D,Conv2DTranspose,concatenate
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.layers import LeakyReLU

from keras.layers.merge import Concatenate



from coord import CoordinateChannel2D


class denseUnet():
    def __init__(self,img_width,img_height,activation_unit=None,learning_rate=None,drop_ratio=0.1):
        
        self._img_width = img_width
        self._img_height = img_height
        if activation_unit ==None:
            self._activation_unit = "LeakyRelu"
        else:
            self._activation_unit = activation_unit
            
        if learning_rate == None:
            self._learning_rate = 0.001
        else:
            self._learning_rate = learning_rate
        
        self.start_neurons = 16 
        self.drop_ratio = drop_ratio
            
            
    def convolution_block(self,x, filters, size, strides=(1,1), padding='same', activation=True):
        x = Conv2D(filters, size, strides=strides, padding=padding,kernel_initializer='Orthogonal')(x)
        if activation == True:
            x = self.batch_activate(x)
        return x

    def batch_activate(self,x):
        x = BatchNormalization(axis=3, momentum=0.0,epsilon=0.0001)(x)
        if self._activation_unit =="LeakyRelu":
            x = LeakyReLU(alpha=0.1)(x)
        else:
            x = Activation(self._activation_unit)(x)
        return x
    
    
    def densenet_block(self,blockInput, num_filters=16, batch_activate = False):

        count=3
        list_feature = [blockInput]
        pas=self.convolution_block(blockInput, num_filters,size=(3,3),strides=(1,1),activation= True)
        for i in range(2 , count+1):
            list_feature.append(pas)
            out =  Concatenate(axis = 3)(list_feature) # conctenated out put
            pas=self.convolution_block(out, num_filters,size=(3,3),strides=(1,1),activation=True)
  
        # feature extractor from the dense net
        list_feature.append(pas)
        out = Concatenate(axis =3)(list_feature)
        feat=self.convolution_block(out, num_filters,size=(3,3),strides=(1,1),activation=True)

        return feat
 
    def build(self):
        
    
        inputs = Input((self._img_width,self._img_height,3))
    
        coord0=CoordinateChannel2D()(inputs)

        conv1 = Conv2D(self.start_neurons * 1, (3, 3), activation="elu", padding="same")(coord0)
        conv1 = self.densenet_block(conv1,self.start_neurons * 1)
        conv1 = self.densenet_block(conv1,self.start_neurons * 1, True)
        pool1 = MaxPooling2D((2, 2))(conv1)
        pool1 = Dropout(self.drop_ratio/2)(pool1)
    
        
        conv2 = Conv2D(self.start_neurons * 2, (3, 3), activation="elu", padding="same")(pool1)
        conv2 = self.densenet_block(conv2,self.start_neurons * 2)
        conv2 = self.densenet_block(conv2,self.start_neurons * 2, True)
        pool2 = MaxPooling2D((2, 2))(conv2)
        pool2 = Dropout(self.drop_ratio)(pool2)
    

        conv3 = Conv2D(self.start_neurons * 4, (3, 3), activation="elu", padding="same")(pool2)
        conv3 = self.densenet_block(conv3,self.start_neurons * 4)
        conv3 = self.densenet_block(conv3,self.start_neurons * 4, True)
        pool3 = MaxPooling2D((2, 2))(conv3)
        pool3 = Dropout(self.drop_ratio)(pool3)
 
        conv4 = Conv2D(self.start_neurons * 8, (3, 3), activation="elu", padding="same")(pool3)
        conv4 = self.densenet_block(conv4,self.start_neurons * 8)
        conv4 = self.densenet_block(conv4,self.start_neurons * 8, True)
        pool4 = MaxPooling2D((2, 2))(conv4)
        pool4 = Dropout(self.drop_ratio)(pool4)
    

        convm = Conv2D(self.start_neurons * 16, (3, 3), activation="elu", padding="same")(pool4)
        convm = self.densenet_block(convm,self.start_neurons * 16)
        convm = self.densenet_block(convm,self.start_neurons * 16, True)
      
        
    
        deconv4 = Conv2DTranspose(self.start_neurons * 16, (4, 4), strides=(2, 2), padding="same")(convm)
        uconv4 = concatenate([deconv4,conv4])
        uconv4 = Dropout(self.drop_ratio)(uconv4)
        
        uconv4 = Conv2D(self.start_neurons * 8, (3, 3), activation="elu", padding="same")(uconv4)
        uconv4 = self.densenet_block(uconv4,self.start_neurons * 8)
        uconv4 = self.densenet_block(uconv4,self.start_neurons * 8, True)


        deconv3 = Conv2DTranspose(self.start_neurons * 8, (4, 4), strides=(2, 2), padding="same")(uconv4)
        uconv3 = concatenate([deconv3,conv3])    
        uconv3 = Dropout(self.drop_ratio)(uconv3)
        
        uconv3 = Conv2D(self.start_neurons * 4, (4, 4), activation="elu", padding="same")(uconv3)
        uconv3 = self.densenet_block(uconv3,self.start_neurons * 4)
        uconv3 = self.densenet_block(uconv3,self.start_neurons * 4, True)



        deconv2 = Conv2DTranspose(self.start_neurons * 2, (4, 4), strides=(2, 2), padding="same")(pool2)
        

        uconv2 = concatenate([deconv2,conv2])
            
        uconv2 = Dropout(self.drop_ratio)(uconv2)
        uconv2 = Conv2D(self.start_neurons * 2, (3, 3), activation="elu", padding="same")(uconv2)
        uconv2 = self.densenet_block(uconv2,self.start_neurons * 2)
        uconv2 = self.densenet_block(uconv2,self.start_neurons * 2, True)
    
        
        deconv1 = Conv2DTranspose(self.start_neurons * 1, (3, 3), strides=(2, 2), padding="same")(uconv2)
        uconv1 = concatenate([deconv1,conv1])
        
        uconv1 = Dropout(self.drop_ratio)(uconv1)
        uconv1 = Conv2D(self.start_neurons * 1, (3, 3), activation="elu", padding="same")(uconv1)
        uconv1 = self.densenet_block(uconv1,self.start_neurons * 1)
        uconv1 = self.densenet_block(uconv1,self.start_neurons * 1, True)    
        uconv1 = Dropout(self.drop_ratio/2)(uconv1)

        
        x2 = Conv2D(filters=1, kernel_size=(3,3), strides=(1,1), kernel_initializer='Orthogonal',padding='same')(uconv1)
        output =  Activation('linear', name = 'output')(x2)
    
        model = Model(inputs=[inputs], outputs=[output])
    

        adam0 = Adam(lr=self._learning_rate)
        model.compile(optimizer=adam0, loss="mse", metrics=["mae"])
        
        
        return model
    


        
        
        
        