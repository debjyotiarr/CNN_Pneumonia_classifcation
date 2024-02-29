import pandas as pd
import numpy as np
import tensorflow as tf
from src.make_dataset import *
from src.augment_data import *
from src.visualise_image import *
from models.CNN_model import CNNModel, CNNModel2


train_files_pneum = "Covid-19_dataset/train/Viral_Pneumonia"
train_files_normal = "Covid-19_dataset/train/Normal"
test_files_pneum = "Covid-19_dataset/test/Viral_Pneumonia"
test_files_normal = "Covid-19_dataset/test/Normal"


#make_dataset
df_train = make_dataset(train_files_pneum, train_files_normal)
df_test  = make_dataset(test_files_pneum,  test_files_normal)

print("Training data")
print(df_train.head())
print("shape = ",df_train.shape)
print("Testing data")
print(df_test.head())
print("shape =",df_test.shape)


#augment data
df_train_aug = augment_dataframe(df_train)  #the augment_data function returns a generator. one can directly pass this on to train
df_test_aug  = augment_dataframe(df_test, batch_size=1) #don't really augment

#print("some images for training")
#visualize_images(df_test_aug)

###Validation?

#load the model architecture
inputSize = (228,228,3) #size of the image to be input
cnnModel = CNNModel2(input_size = inputSize)
print(cnnModel.summary())

#compile model
#model_params = {loss:'categorial_crossentropy', optimizer:'adam', metrics : ['accuracy']}

cnnModel.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#train/fit model
numEpochs = 50

#callbacks
early_stopping_callback= tf.keras.callbacks.EarlyStopping(monitor='loss',
                                                         patience=5,
                                                         verbose=1,
                                                         restore_best_weights=True)

callbackFlag = False
callbacks = [early_stopping_callback]

if callbackFlag:
    model_history = cnnModel.fit(df_train_aug, epochs = numEpochs, batch_size=32, callbacks = callbacks)
else:
    model_history = cnnModel.fit(df_train_aug, epochs=numEpochs, batch_size=32)

modelName = "models/cnnModel"+ "epoch"+str(numEpochs)+"_callback"+str(callbackFlag)+".h5"
cnnModel.save(modelName)

"""

#predict
#first load
model = tf.keras.models.load_model(modelName)
predProba = model.predict(df_test_aug, verbose = 1)
predLabels = tf.argmax(predProba, axis=1)


"""




### To Do
# 1. Validation dataset
# 2. Plot images
# 3. Plot model history
# 4. separate training and loading models in different modules
# 5. Generalise the model for three output classes - include covid
# 6. compare with given labels