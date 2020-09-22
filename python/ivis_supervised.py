import numpy as np
import random as rd
import sys
from operator import itemgetter
import itertools
import ipdb
import time
import json

import tensorflow as tf
from tensorflow import keras, math
from tensorflow.keras import backend as K
from tensorflow.keras import Model, layers, regularizers
from tensorflow.keras.layers import Dense, Input, Lambda
from tensorflow.keras.callbacks import EarlyStopping

from sklearn import datasets
from sklearn.neighbors import NearestNeighbors


tf.keras.backend.set_floatx('float64')


#####
# Model subclassing
#####

class BaseModel(keras.Model):
  def __init__(self, input_dim=None, dense_layers=[128,128,128], dropout_layers=[0.1,0.1], embedding_dim=2, embedding_l2=0.1, name='base', **kwargs):
    super(BaseModel, self).__init__(name=name, **kwargs)
    self.input_dim = input_dim
    self.dense_layers = dense_layers
    self.dropout_layers = dropout_layers
    self.embedding_dim = embedding_dim
    self.embedding_l2 = embedding_l2
    
    self.alphadrop = [None]*len(dropout_layers)
    self.dense_proj = [None]*len(dense_layers)
    
    for i,v in enumerate(self.dense_layers):
      self.dense_proj[i] = layers.Dense(v, activation='selu', kernel_initializer='lecun_normal', name = "Dense_"+str(i))
      
    for i,v in enumerate(self.dropout_layers):
      self.alphadrop[i] = layers.AlphaDropout(rate=v, name = "AlphaDropout_"+str(i))
      
    self.embed = layers.Dense(self.embedding_dim, kernel_regularizer=tf.keras.regularizers.l2(self.embedding_l2), name = "embedding_layer")
    return None
    
  def call(self, inputs, training = None):
    x = inputs
    x = self.dense_proj[0](x)
    for i in range(1,len(self.dense_proj)):
      x=self.dense_proj[i](self.alphadrop[i-1](x))
      
    x = self.embed(x)
    return x
    
  def infer(self):
    x = Input(shape=(self.input_dim,), name='main_input')
    return Model(inputs = x, outputs = self.call(x), name='base')
    
class Classifier(tf.keras.Model):
  def __init__(self, input_dim = None, nb_classes = 2, noise_sd = 0.1, name = 'classifier', **kwargs):
    super(Classifier, self).__init__(name=name, **kwargs)
    self.input_dim = input_dim
    self.nb_classes = nb_classes
    self.noise_sd = noise_sd
    
    self.norm = tf.keras.layers.BatchNormalization(axis = 1, trainable = False)
    self.noise = layers.GaussianNoise(stddev = self.noise_sd, name = 'Gaussian_Noise')
    self.classif = layers.Dense(units = nb_classes, activation = 'softmax', name = 'Classification')
    return None
    
  def call(self, inputs, training = None):
    x = inputs
    x = self.classif(self.noise(self.norm(x)))
    return x
    
  def infer(self):
    x = Input(shape = (self.input_dim,), name = 'embedded_output')
    return Model(inputs = x, outputs = self.call(x), name='classifier')
    

class TripletModel(tf.keras.Model):
  def __init__(self, input_dim, embedding_dim=2, nb_classes = 2, dense_layers = [128,128,128], dropout_layers = [0.1,0.1], distance = "euclidean", margin = 1., name="ivis", **kwargs):
    super(TripletModel, self).__init__(name=name, **kwargs)
    self.input_dim = input_dim
    self.embedding_dim = embedding_dim
    self.nb_classes = nb_classes
    self.dense_layers = dense_layers
    self.dropout_layers = dropout_layers
    self.distance = distance
    self.margin = margin
    
    self.base = BaseModel(input_dim = self.input_dim, dense_layers = self.dense_layers, dropout_layers = self.dropout_layers, embedding_dim = self.embedding_dim)
    self.classifier = Classifier(input_dim = self.embedding_dim, nb_classes = self.nb_classes)
    self.stack = tf.keras.layers.Lambda(K.stack, output_shape = (3, self.embedding_dim,), name = "Output_concat.")
    return None
    
  def train(self, dataset_list, classes, epochs, epochs_without_improvement, batch_size, loss_object, optimizers):
    loss_history = []
    minimum_loss = 0.
    c = 0
    formatted_data = data_format(dataset_list, batch_size)
    formatted_classes = data_format([classes], batch_size)[0]
    anchor, positive, negative = formatted_data
    for epoch in range(epochs):
      start = time.time()
      current_epoch_loss = 0.
      
      for (batch, data) in enumerate(anchor):
        inputs = [data, positive[batch], negative[batch]]
        cl = formatted_classes[batch]
        ivis_loss, class_loss = self.train_step(inputs, cl, loss_object, optimizers)
        current_epoch_loss += ivis_loss
        #print("\t Batch", batch)
        
        if batch == len(anchor)-1:
          acc = self.accuracy(cl, inputs)
        
      current_epoch_loss /= len(anchor)
      loss_history.append(current_epoch_loss.numpy())
        
      if epoch == 0:
        minimum_loss = current_epoch_loss
      if current_epoch_loss < minimum_loss:
        minimum_loss = current_epoch_loss
        c = 0
      else:
        c += 1
  
      if c >= epochs_without_improvement:
        print(epochs_without_improvement, "epochs have elapsed with no improvement. Process finished")
        sys.stdout.flush()
        return loss_history
      
      print ('Time for epoch {} is {} sec -'.format(epoch, time.time()-start), 'ivis loss: {} -'.format(round(current_epoch_loss.numpy(), 4)), 'classification accuracy: {}'.format(round(acc.numpy(), 4)))
      sys.stdout.flush()
    return loss_history
  
  @tf.function
  def train_step(self, inputs, classes, loss_object, optimizers):
    loss, loss_array, class_loss = loss_object
    ivis_opti, class_opti = optimizers
    with tf.GradientTape(persistent=True) as tape:
      
      embed, classified = self(inputs, training=True)
      
      ivis_loss_value = loss(y_true = [1.], y_pred = embed)
      class_loss_value = class_loss(y_true = classes, y_pred = classified)

    class_grads = tape.gradient(class_loss_value, self.trainable_variables) 
    ivis_grads = tape.gradient(ivis_loss_value, self.trainable_variables[:8]) 
    
    class_opti.apply_gradients(zip(class_grads, self.trainable_variables)) # le resultat de la classification affecte egalement ivis
    ivis_opti.apply_gradients(zip(ivis_grads, self.trainable_variables[:8]))
    
    return tf.reduce_mean(ivis_loss_value), tf.reduce_mean(class_loss_value)
    
  def get_score_array(self, inputs, loss_function):
    logits = self(inputs, training=True)
    loss_array = loss_function(y_true = [1.], y_pred = logits)
    return loss_array
  
  def get_config(self):
        return {'input_dim': self.input_dim, 'embedding_dim': self.embedding_dim, 'nb_classes': self.nb_classes,
            'dense_layers': self.dense_layers, 'dropout_layers': self.dropout_layers, 'distance': self.distance, 'margin': self.margin}
    
  def save_model(self, config_file, weights_file):
    json_config=self.get_config()
    json.dump(json_config, open(config_file,'w'))
    self.save_weights(weights_file)
    return None
    
  def call(self, inputs, training=None):
    x = inputs
    anc = self.base(x[0], training=training)
    pos = self.base(x[1], training=training)
    neg = self.base(x[2], training=training)
    
    classified = self.classifier(anc)

    embedded_triplet = self.stack([anc, pos, neg])

    return embedded_triplet, classified
    
  def infer(self):
    input_a = Input(shape=(self.input_dim,))
    input_p = Input(shape=(self.input_dim,))
    input_n = Input(shape=(self.input_dim,))
    x = [input_a, input_p, input_n]
    
    return Model(inputs = x, outputs = self.call(x), name='ivis')
    
  def accuracy(self, y_true, data):
    y_true = tf.cast(K.argmax(y_true, axis=1), dtype=tf.int32)
    logits = self.call(data, training=False)[1]
    classes = tf.cast(K.argmax(logits, axis=1), dtype=tf.int32)
    acc = tf.math.reduce_mean(tf.cast(K.equal(classes,y_true), dtype=tf.float32))
    return acc

#####
# Load saved model
#####

def load_model(config_file, weights_file):
  config = json.load(open(config_file))
  new_model = TripletModel(config["input_dim"], config["embedding_dim"], config["nb_classes"], config["dense_layers"], config["dropout_layers"], config["distance"], config["margin"])
  
  optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
  loss = pn_loss_builder(config["distance"], config["margin"])[0]
  new_model.compile(optimizer=optimizer, loss=loss)
  x=[np.random.rand(10,config["input_dim"]), np.random.rand(10,config["input_dim"]), np.random.rand(10,config["input_dim"])]
  y=np.random.randint(0,new_model.nb_classes,10)
  new_model.train_on_batch(x,np.array([x[0],y]))
  new_model.load_weights(weights_file)
  
  def _predict(data):
    return new_model(data).numpy()
        
  return new_model, _predict
  
#####
# Calcule les arrays positif et négatif à partir de la matrice des KNN, puis forme les batchs
#####

def input_compute(x,k,approx=False,knn_matrix=None): #x représente un dataset 

  data = x[:,:-1] # la derniere colonne contient les classes

  if approx:
    if knn_matrix is None:
      build_annoy_index(data,"./ind",build_index_on_disk=True)
      knn_matrix = extract_knn(data,"./ind")
    
  else:
    nbrs = NearestNeighbors(n_neighbors=k, algorithm="ball_tree").fit(data)
    distances, knn_matrix = nbrs.kneighbors(data) #knn_matrix est la matrice des knn
  
  positive = np.empty(np.shape(x))
  negative = np.empty(np.shape(x))
  
  for i in range(knn_matrix.shape[0]):
    positive[i,:] = x[rd.choice(knn_matrix[i,1:]),:]
    negative[i,:] = x[rd.choice(knn_matrix[:,0]),:] #k<<N

  inputs = [x,positive,negative]
      
  return inputs #renvoie les points choisis


def data_format(dataset_list, batch_size): #forme des batchs dans un dataset structuré en 3 parties [anchor, positive, negative]
  formatted_data = []
  for dataset in dataset_list:
    slices = tf.data.Dataset.from_tensor_slices(dataset)
    slices = slices.batch(batch_size)
    formatted_data.append(list(slices.as_numpy_iterator()))
  
  return formatted_data

#####
# Fonction de coût/Loss function
#####

def euclidean_distance(x, y):
  return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), tf.cast(K.epsilon(),dtype="float64")))
  
def cosine_distance(x, y):
  def norm(t):
    return K.sqrt(K.maximum(K.sum(K.square(t), axis=1, keepdims=True), tf.cast(K.epsilon(),dtype="float64")))
  return K.sum(x * y, axis = 1, keepdims = True)/(norm(x)*norm(y)) 

def pn_loss_builder(distance='euclidean', margin=1., weight = 1.):
  if distance=='euclidean':
    distance = euclidean_distance
  elif distance=='cosine':
    distance=cosine_distance
  
  def _pn_loss_array(y_true, y_pred, distance=distance, margin=margin):    
    anchor, positive, negative = tf.unstack(y_pred)
  
    anchor_positive_distance = distance(anchor, positive)
    anchor_negative_distance = distance(anchor, negative)
    positive_negative_distance = distance(positive, negative)
  
    minimum_distance = K.min(K.concatenate([anchor_negative_distance, positive_negative_distance]), axis=1, keepdims=True)
  
    loss_array = K.maximum(anchor_positive_distance - minimum_distance + margin, 0)
  
    return loss_array
    
  def _pn_loss(y_true, y_pred, distance=distance, margin=margin):
    return weight*K.mean(_pn_loss_array(y_true, y_pred, distance=distance, margin=margin))
    
  return [_pn_loss, _pn_loss_array]
  
#####
# R Wrapper
#####
    
def R_wrapper_interface(dataset, classes, epochs, batch_size, Sbatch_draws = 5, epochs_without_improvement = 10., start_imp = 0, dense_layers = [128,128,128], alpha = [0.1,0.1], embedding_dim = 2, k=15, approx = False, distance = "euclidean", margin = 1., pn_weight = 1., verbose = 1, debug = False):
  
  nb_classes = np.unique(classes).shape[0]

  siamese_model = TripletModel(input_dim = dataset.shape[1], embedding_dim = embedding_dim, nb_classes = nb_classes, dense_layers = dense_layers, dropout_layers = alpha, distance = distance, margin = margin)
  d = np.empty((dataset.shape[0], dataset.shape[1] + 1))
  d[:,:-1] = dataset
  d[:,-1] = classes # merge dataset and classes along axis 1

  inputs = input_compute(d, k, approx)
  classes = inputs[0][:,-1]
  classes = tf.keras.utils.to_categorical(np.array(classes).astype(int) - 1, num_classes = nb_classes)
  inputs = [inputs[0][:,:-1],inputs[1][:,:-1],inputs[2][:,:-1]]

  optimizers = [tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07), tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07)]

  loss_object = pn_loss_builder(distance, margin, pn_weight)
  loss_object.append(tf.keras.losses.CategoricalCrossentropy())
  epoch_loss_history = []
    
  if debug:
    epoch_loss_history = siamese_model.train(inputs, classes, epochs, epochs_without_improvement, batch_size, loss_object, optimizers)
  else:
    es = EarlyStopping(monitor='val_loss', mode='min', restore_best_weights=True, patience=epochs_without_improvement)
    siamese_model.compile(optimizer=optimizer, loss=loss_object[0], target_tensors = [1.])
    siamese_model.fit(inputs, epochs, batch_size, callbacks=[es], validation_split=0.2, shuffle=True, verbose = verbose)
  
  epoch_array = np.arange(len(epoch_loss_history))
  epoch_loss_progression = np.column_stack((epoch_array, epoch_loss_history)) 
  
  siamese_model.infer().summary()
  siamese_model.layers[0].infer().summary()
  siamese_model.layers[1].infer().summary()
  output_embeddings = siamese_model.layers[0](dataset).numpy() #Predict embeddings.
  predicted_classes = K.argmax(siamese_model.layers[1](output_embeddings)).numpy() + 1 #Predict classes.
 
  sys.stdout.flush()
  return epoch_loss_progression, output_embeddings, siamese_model, predicted_classes
