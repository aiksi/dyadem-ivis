import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, GaussianNoise, Dropout, AlphaDropout
import numpy as np
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import EarlyStopping
from tempfile import TemporaryFile
from tensorflow.keras.models import load_model
tf.keras.backend.set_floatx('float64')



def Annotation(Data, pop, neuralnet=None,epochs=1,patience=10,encoder=[128,128,128,128],drate=[0.2,0.2,0.2,0.2]): 
  pop=tf.keras.utils.to_categorical(pop)
  es = EarlyStopping(monitor='val_loss', mode='min', restore_best_weights=True, patience=patience) #termine training avant overfitting
  
  if neuralnet is None:
    input=Input(shape=(Data.shape[1],)) 
   
    # input_noise=GaussianNoise(.1)(input) 
    x=input
    for i in range(len(encoder)):
      x=Dense(encoder[i], activation='relu')(x)
      x=Dropout(drate[i])(x)
    output = Dense(pop.shape[1], activation='softmax', name='aux_output1')(x) 
    
    neuralnet = Model(input, outputs=output) 
    
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001)
    
    neuralnet.compile(optimizer=optimizer, 
    loss=['categorical_crossentropy'],  
    metrics={'aux_output1':'accuracy'}) 
    
    neuralnet.summary()
    
  
  entrainement = neuralnet.fit(Data, pop, epochs=epochs, batch_size=256, callbacks=[es], shuffle=True, validation_split=0.2, verbose = 1) 
  
  def _classif(data):
    res=neuralnet(data, training=False)
    res=K.argmax(res)
    return res.numpy()

  
  return neuralnet.predict(Data), neuralnet, _classif
  
  

