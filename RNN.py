#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf


# In[2]:


from keras.datasets import imdb


# In[3]:


from keras import layers, models, losses, optimizers


# In[5]:


from keras.utils import pad_sequences


# In[6]:


vocab_size = 5000
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=vocab_size)


# In[7]:


reviewLengths = [len(x) for x in X_train]


# In[8]:


import matplotlib.pyplot as plt


# In[9]:


plt.hist(reviewLengths, bins=10)
plt.show()


# Build a LSTM network

# In[10]:


tf.random.set_seed(42)


# In[11]:


model = models.Sequential()


# In[12]:


embeddingSize = 32


# In[14]:


model.add(layers.Embedding(vocab_size, embeddingSize, input_length=500))


# In[15]:


model.add(layers.LSTM(50))


# In[16]:


model.add(layers.Dense(1, activation='sigmoid'))


# In[17]:


model.compile(loss=losses.binary_crossentropy, optimizer=optimizers.RMSprop(lr=0.001), metrics=['accuracy'])


# In[19]:


model.fit(X_train, y_train, epochs=10, batch_size=128, validation_data=(X_test, y_test))

