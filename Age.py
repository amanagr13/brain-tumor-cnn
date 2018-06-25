
# coding: utf-8

# In[1]:


#get_ipython().run_line_magic('pylab', 'inline')
import os
import random
import sys
import numpy as np
import sklearn
import keras
import timeit


import pandas as pd
from scipy.misc import imread


# In[2]:


root_dir = os.path.abspath('.')
data_dir = root_dir

train = pd.read_csv(os.path.join(data_dir, 'train.csv'))
test = pd.read_csv(os.path.join(data_dir, 'test.csv'))


# In[5]:


i = random.choice(train.index)

img_name = train.ID[i]
img = imread(os.path.join(data_dir, 'Train', img_name))

print('Age: ', train.Class[i])
#imshow(img)


# In[6]:


from scipy.misc import imresize

temp = []
for img_name in train.ID:
    img_path = os.path.join(data_dir, 'Train', img_name)
    img = imread(img_path)
    img = imresize(img, (128, 128))
    img = img.astype('float32') # this will help us in later stage
    temp.append(img)

train_x = np.stack(temp)



# In[ ]:


temp = []
for img_name in test.ID:
    img_path = os.path.join(data_dir, 'Test', img_name)
    img = imread(img_path)
    img = imresize(img, (128, 128))
    temp.append(img.astype('float32'))

test_x = np.stack(temp)


# In[ ]:


train_x = train_x / 255.
test_x = test_x / 255.
print("half")


# In[ ]:


train.Class.value_counts(normalize=True)


# In[ ]:


test['Class'] = 'MIDDLE'
test.to_csv('sub01.csv', index=False)


# In[ ]:



from sklearn.preprocessing import LabelEncoder

lb = LabelEncoder()
train_y = lb.fit_transform(train.Class)
train_y = keras.utils.np_utils.to_categorical(train_y)


# In[ ]:


input_num_units = (128, 128, 3)
hidden_num_units = 500
output_num_units = 3

epochs = 50
batch_size = 16


# In[ ]:


from keras.models import Sequential
from keras.layers import Dense, Flatten, InputLayer, Conv2D, MaxPooling2D



# In[ ]:


model = Sequential([
  InputLayer(input_shape=input_num_units),
  
  Conv2D(32, (3,3), padding='same', activation='relu'), 
  MaxPooling2D(pool_size=(2,2)),
  
  Conv2D(64, (3,3), padding='same', activation='relu'), 
  MaxPooling2D(pool_size=(2,2)), 
  
  Conv2D(128, (3,3), padding='same', activation='relu'), 
  MaxPooling2D(pool_size=(2,2)), 
  

  Flatten(), 
  
  Dense(units=hidden_num_units, activation='relu'),
  Dense(units=output_num_units, activation='softmax'),
])


# In[ ]:



model.summary()


# In[ ]:

start=timeit.default_timer
model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_x, train_y, batch_size=batch_size,epochs=epochs,verbose=1)
end=timeit.default_timer

i = random.choice(train.index)
img_name = train.ID[i]

img = imread(os.path.join(data_dir, 'Train', img_name)).astype('float32')
#imshow(imresize(img, (128, 128)))
pred = model.predict_classes(train_x)
print('Original:', train.Class[i], 'Predicted:', lb.inverse_transform(pred[i]))

