
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tqdm
#get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df = pd.read_csv("./data_source.csv")
print("Total records = ", len(df))
df.head()


# In[ ]:


# select only the columns that we are interested in
df = df[["_unit_id", "please_select_the_gender_of_the_person_in_the_picture",
    "please_select_the_gender_of_the_person_in_the_picture:confidence", "image_url"]]
 
# rename the columns
df.columns = ["id", "gender", "confidence", "url"]
 
# only select the rows that has confidence of 1.0
df = df[df["confidence"] == 1]
 
print("Total records = ", len(df))


# In[ ]:


# helper function to display image urls
from IPython.display import HTML, display
def display_images(df, category_name="male", count=12):
    filtered_df = df[df["gender"] == category_name]
    p = np.random.permutation(len(filtered_df))
    p = p[:count]
    img_style = "width:180px; margin:0px; float:left;border:1px solid black;"
    images_list = "".join(["<img src={}>".format(img_style, u) for u in filtered_df.iloc[p].url])
    display(HTML(images_list))


# In[ ]:


df_male = df[df["gender"] == "male"]
df_female = df[df["gender"] == "female"]
 
# to make both categories have equal number of samples
# we'll take the counts of the category that has lowest
# number of samples
min_samples = min(len(df_male), len(df_female))
 
# for indexing randomly
p = np.random.permutation(min_samples)
 
df_male = df_male.iloc[p]
df_female = df_female.iloc[p]
 
print("Total male samples = ", len(df_male))
print("Total female samples = ", len(df_female))
 
df = pd.concat([df_male, df_female])


# In[ ]:


import os
import requests
from io import BytesIO
from PIL import Image
 
def download_images(df, data_dir="./data"):
    genders = df["gender"].unique()
    for g in genders:
        g_dir = "{}/{}".format(data_dir, g)
        if not os.path.exists(g_dir):
            os.makedirs(g_dir)
           
    for index, row in tqdm.tqdm_notebook(df.iterrows()):
        filepath = "{}/{}/{}.jpg".format(data_dir, row["gender"], row["id"])
        if os.path.exists(filepath):
            continue
        #try:
           # resp = requests.get(row["url"])
            #im = Image.open(BytesIO(resp.content))
            #im.save(filepath)
        #except:
            #print("Error while downloading %s" % row["url"])
 
DATA_DIR = "./data"
download_images(df, data_dir=DATA_DIR)  
 
# create train/test folder for each gender
import glob
 
TRAIN_DIR = DATA_DIR + "/train"
TEST_DIR = DATA_DIR + "/test"
 
for d in [TRAIN_DIR, TEST_DIR]:
    for g in df["gender"].unique():
        final_dir = "{}/{}".format(d, g)
        if not os.path.exists(final_dir):
            os.makedirs(final_dir)
 
from random import shuffle
import math
import shutil
 
split_ratio = 0.8 # we'll reserve 70% of the images for training set
 
def validate_and_move(files, target_dir):
    for f in tqdm.tqdm_notebook(files):
        # try to open the file to make sure that this is not corrupted
        try:
            im = Image.open(f)
            shutil.copy(f, target_dir)
        except:
            pass
#             os.remove(f)
 
for gender in df["gender"].unique():
    gender_dir = "{}/{}".format(DATA_DIR, gender)
    pattern = "{}/*.jpg".format(gender_dir)
    all_files = glob.glob(pattern)
    shuffle(all_files)
   
    train_up_to = math.ceil(len(all_files) * split_ratio)
    train_files = all_files[:train_up_to]
    test_files = all_files[train_up_to:]
   
   
    validate_and_move(train_files, TRAIN_DIR + "/" + gender)
    validate_and_move(test_files, TEST_DIR + "/" + gender)


# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
#get_ipython().run_line_magic('matplotlib', 'inline')
 
TRAIN_DIR = "./data/train"
TEST_DIR = "./data/test"
IM_WIDTH = 198
IM_HEIGHT = 198
 
def plot_images(images, labels):
    n_cols = min(5, len(images))
    n_rows = len(images) // n_cols
    fig = plt.figure(figsize=(8, 8))
 
    for i in range(n_rows * n_cols):
        sp = fig.add_subplot(n_rows, n_cols, i+1)
        plt.axis("off")
        plt.imshow(images[i], cmap=plt.cm.gray)
        sp.set_title(labels[i])
    plt.show()


# In[ ]:


from keras.layers import Input, Dense, Flatten, Conv2D, MaxPool2D, BatchNormalization, Activation, GlobalAvgPool2D
from keras.models import Model
 
img_input = Input(shape=(IM_HEIGHT, IM_WIDTH, 3))
_ = Conv2D(filters=32, kernel_size=3)(img_input)
_ = Activation("relu")(_)
_ = BatchNormalization()(_)
_ = MaxPool2D()(_)
 
_ = Conv2D(filters=64, kernel_size=3)(_)
_ = Activation("relu")(_)
_ = BatchNormalization()(_)
_ = MaxPool2D()(_)
 
_ = Conv2D(filters=64, kernel_size=3)(_)
_ = Activation("relu")(_)
_ = BatchNormalization()(_)
_ = MaxPool2D()(_)
 
_ = Conv2D(filters=128, kernel_size=3)(_)
_ = Activation("relu")(_)
_ = BatchNormalization()(_)
_ = MaxPool2D()(_)
 
_ = Conv2D(filters=128, kernel_size=3)(_)
_ = Activation("relu")(_)
_ = BatchNormalization()(_)
_ = GlobalAvgPool2D()(_)
 
_ = Dense(128)(_)
_ = Activation("relu")(_)
_ = BatchNormalization()(_)
 
_ = Dense(1)(_)
_ = Activation("sigmoid")(_)
 
model = Model(inputs=img_input, outputs=_)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.summary()


# In[ ]:


from keras.preprocessing.image import ImageDataGenerator
train_data_gen = ImageDataGenerator(rescale=1./255,
                              rotation_range=20,
                              horizontal_flip=True
                             )
 
test_data_gen = ImageDataGenerator(rescale=1./255)
 
batch_size = 32
train_gen = train_data_gen.flow_from_directory(TRAIN_DIR,
                                               target_size=(IM_WIDTH, IM_HEIGHT),
                                               class_mode="binary",
                                               batch_size=batch_size)
test_gen = test_data_gen.flow_from_directory(TEST_DIR,
                                            target_size=(IM_WIDTH, IM_HEIGHT),
                                            class_mode="binary",
                                            batch_size=batch_size)
 
# now we can train the model
history = model.fit_generator(train_gen, steps_per_epoch=len(train_gen.filenames)//batch_size, epochs=16)


# In[ ]:


model.evaluate_generator(test_gen, steps=len(test_gen.filenames)//batch_size)


# In[ ]:


# predict for one batch of test images
imgs, y_true = next(test_gen)
predictions = model.predict(imgs, batch_size=imgs.shape[0])
 
# how do I know if male is 0 or 1? ImageDataGenerator has a property class_indices
# try printing test_gen.class_indices
# since sigmoid produces value between 0 and 1, we create a threshold and say
# any value > 0.5 is a male otherwise it is female
predictions_str = np.where(predictions.flatten() > 0.5, "male", "female")
plot_images(imgs, predictions_str)

