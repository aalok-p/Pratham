# PRATHAM
It classify differnt types of cancer, majorly we focused on classifying:-
<ol>
 
 <ins>**Brain Tumor**</ins><br />
 <ins>**Breast Cancer**</ins><br />
 <ins>**Skin Cancer**</ins><br />

Input Shape: (224, 224, 3) (standard size for MobileNet)
Flatten Layer: Converts 2D matrices into a 1D vector
Output Layer: A Dense layer with a sigmoid activation function for binary classification.

<ins>**🛠️ Installation**</ins><br />

To run this project, please ensure you have the following prerequisites installed:

<li>Python</li> 
<li>TensorFlow</li>
<li>Keras</li>
<li>NumPy</li>
<li>Matplotlib</li>
<p>
 <br>
</p>

<ins>**🛠️ Code Snippet**</ins><br />

```python
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.layers import Flatten,Dense
from keras.models import Model,load_model
from keras.applications.mobilenet import MobileNet,preprocess_input
import keras
     

base_model=MobileNet(input_shape=(224,224,3),include_top=False)

for layer in base_model.layers:
  layer.trainable=False
     

X=Flatten()(base_model.output)
X=Dense(units=1,activation='sigmoid')(X)

model=Model(base_model.input,X)
model.summary()
```

<ins>**📊 Results**</ins><br />

![Image](https://github.com/user-attachments/assets/a8e59efa-4117-4506-a558-69bd64f12b25)

<h1>🩺 Breast Cancer Detection using Machine Learning</h1>

<ins>**🛠️ Technologies Used**</ins><br />

<li>Python: The primary programming language used for implementation.</li>
<li>Libraries:</li>
<li>NumPy: For numerical computations.</li>
<li>Pandas: For data manipulation and analysis.</li>
<li>Matplotlib: For data visualization.</li>
<li>Scikit-learn: For dataset handling, preprocessing, and model evaluation.</li>
<li>TensorFlow/Keras: For building and training the neural network.</li>
<p>
 <br>
</p>
<ins>**🧠 Model Architecture**</ins><br />
<p>
 <br>
</p>
The neural network model is built using Keras and consists of the following layers:

<li>Input Layer: 30 input features.</li>
<li>Hidden Layer: 20 neurons with ReLU activation.</li>
<li>Output Layer: 2 neurons with sigmoid activation for binary classification.</li>
<p>
 <br>
</p>
<ins>**🛠️ Installation**</ins><br />
<p>
 <br>
</p>
To run this project, please ensure you have the following prerequisites installed:

<li>Python</li> 
<li>TensorFlow/Keras</li>
<li>NumPy</li>
<li>Pandas</li>
<li>Matplotlib</li>
<li>scikit-learn</li>
<p>
 <br>
</p>

<ins>**🛠️ Code Snippet**</ins><br />

```python
input_data = (11.76,21.6,74.72,427.9,0.08637,0.04966,0.01657,0.01115,0.1495,0.05888,0.4062,1.21,2.635,28.47,0.005857,0.009758,0.01168,0.007445,0.02406,0.001769,12.98,25.72,82.98,516.5,0.1085,0.08615,0.05523,0.03715,0.2433,0.06563)

# change the input_data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the numpy array as we are predicting for one data point
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

# standardizing the input data
input_data_std = scaler.transform(input_data_reshaped)

prediction = model.predict(input_data_std)
print(prediction)

prediction_label = [np.argmax(prediction)]
print(prediction_label)

if(prediction_label[0] == 0):
  print('The tumor is Malignant')

else:
  print('The tumor is Benign')
```
<ins>**📊 Results**</ins><br />

| !(![WhatsApp Image 2024-10-04 at 09 41 59_6574649b](https://github.com/user-attachments/assets/82a8e4d5-cfee-459f-aa74-a34f75bbcd0d)
) | !(![WhatsApp Image 2024-10-04 at 09 41 59_6120567b](https://github.com/user-attachments/assets/b98c453f-5707-4f8b-b91d-b96a5878f0d5)
) 


<h1>Skin Cancer Detection Using Deep Learning</h1>

<ins>**🛠️ Technologies Used**</ins><br />

<li>Python: The primary programming language used for implementation.</li>
<li>Libraries:</li>
<li>NumPy: For numerical computations.</li>
<li>Matplotlib: For data visualization.</li>
<li>Scikit-learn: For dataset handling, preprocessing, and model evaluation.</li>
<li>TensorFlow/Keras: For building and training the neural network.</li>
<li>OpenCV - For image processing</li>
<p>
 <br>
</p>
<ins>**🏗️Model Architecture**</ins><br />


We utilized the MobileNet architecture as our base model. Here’s a brief overview of the architecture used:
```python
from keras.applications.mobilenet import MobileNet
from keras.layers import Flatten, Dense
from keras.models import Model

base_model = MobileNet(input_shape=(224, 224, 3), include_top=False)
X = Flatten()(base_model.output)
X = Dense(units=9, activation='softmax')(X)  # 9 classes for multi-class classification
model = Model(inputs=base_model.input, outputs=X)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
```

<ins>**🛠️ Installation**</ins><br />
<li></li>
To run this project, please ensure you have the following prerequisites installed:

<li>Python</li> 
<li>TensorFlow/Keras</li>
<li>NumPy</li>
<li>OpenCV</li>
<li>Matplotlib</li>
<li>scikit-learn</li>
<p>
 <br>
</p>

<ins>**🛠️ Code Snippet**</ins><br />

```python
from keras.callbacks import ModelCheckpoint

# Define ModelCheckpoint to save the best model
checkpoint = ModelCheckpoint('best.keras', monitor='val_accuracy', mode='max', save_best_only=True, verbose=1)

# Train the model
history = model.fit(
    train_data,
    steps_per_epoch=len(train_data),
    epochs=30,
    callbacks=[checkpoint]
)
```
<p>
 <br>
</p>
<ins>**📊 Results**</ins><br />

| !(![WhatsApp Image 2024-10-04 at 09 57 26_41b0662f](https://github.com/user-attachments/assets/e0391cba-9fd2-41da-af40-ad26ea242243)
) | !(![WhatsApp Image 2024-10-04 at 09 57 26_280558db](https://github.com/user-attachments/assets/38960efd-ffc3-41d5-9f75-b63d0b7175f2)
)
