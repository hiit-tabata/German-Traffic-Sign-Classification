# **Traffic Sign Recognition**

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[newImgs]: ./img/newImgs.png "newImgs"
[grayScale]: ./img/grayScale.png "grayScale"
[ImageAugumentation]: ./img/ImageAugumentation.png "ImageAugumentation"
[labelDistri]: ./img/class_dustribution.png "labelDistri"
[allClass]: ./img/allClass.png "allClass"
[newImgResult]: ./img/newImgResult.png "newImgResult"
[accResult]: ./img/accResult.png "accResult"
[failGraph]: ./img/failGraph.png "failGraph"
[failResult]: ./img/failResult.png "failResult"
[finalGraph]: ./img/finalGraph.png "finalGraph"

### Data Set Summary & Exploration
I use pickle to import the data into ipython notebook. Then I use numpy to calculate the statistics of the data set.

```python
  import numpy as np

  # Number of training examples
  n_train = len(X_train)

  # Number of testing examples.
  n_test = len(X_test)

  # the shape of an traffic sign image
  image_shape = X_train[0].shape

  # number of unique classes/labels there are in the dataset.
  n_classes = len(np.unique(y_train))

  print("Number of training examples =", n_train)
  print("Number of testing examples =", n_test)
  print("Image data shape =", image_shape)
  print("Number of classes =", n_classes)
```

* Image data shape = (32, 32, 3)
* Number of classes = 43


#### 2. Data set exploratoration and visualization
The chart below is the label distribution, which can show the label distribution are not even, some class have more data then others.
![label distribution][labelDistri]


The img below is the chart to show one iamge from each class. Some of them are really dark, which makes difficulties hgiher.
![allClass][allClass]
```python
import matplotlib.pyplot as plt
# Visualizations will be shown in the notebook.
%matplotlib inline
#edit matplot size in notebook
plt.rcParams["figure.figsize"] = [15,18]


fig = plt.figure(figsize=(5,5))
f, axarr = plt.subplots(9, 5)
# make it as a sigle dim. array
plts = np.reshape(axarr, -1)

#display one sample from all class
for classId in np.unique(y_train):
    thePicIndex = np.where(y_train == classId)[0]
    myplt = plts[classId]
    myplt.imshow(X_train[thePicIndex[25]])
    myplt.set_title("class " + str(classId))

plt.tight_layout()
```

### Design and Test a Model Architecture

#### Preprocessed data set
I have few ways to Preprocessed the data set:
1. Turn raw images into gray Scale
2. Normalize the image to range(0, 1)
3. Augmente the images

Gray scale image can reduce the image size, which helps to train the model easier. Image normalized in to range (0, 1) can help model to learn data set. Since the size of dataset not large enough, augmentation on image is necessary.

The Pre-processing params.
* ANGLE_ROTATE = 25
* TRANSLATION = 0.2
* NB_NEW_IMAGES = 10000

``` python
def toGrayscale(rgb):
    result = np.zeros((len(rgb), 32, 32,1))
    result[...,0] = np.dot(rgb[...,:3], [0.299, 0.587, 0.114])  
    return result

# normalize the images
def normalizeGrascale(grayScaleImages):
    return grayScaleImages/255

def processImages(rgbImages):
    return np.array(normalizeGrascale(toGrayscale(rgbImages)))

def transformOnHot(nbClass, listClass):
    oneHot = np.zeros((len(listClass), nbClass))
    oneHot[np.arange(len(listClass)), listClass] = 1
    return np.array(oneHot)

def augmenteImage(image, angle, translation):
    h, w, c = image.shape

    # random rotate
    angle_rotate = np.random.uniform(-angle, angle)
    rotation_mat = cv2.getRotationMatrix2D((w//2, h//2), angle_rotate, 1)

    img = cv2.warpAffine(image, rotation_mat, (IMG_SIZE, IMG_SIZE))

    # random translation
    x_offset = translation * w * np.random.uniform(-1, 1)
    y_offset = translation * h * np.random.uniform(-1, 1)
    mat = np.array([[1, 0, x_offset], [0, 1, y_offset]])

    # return warpped img
    return cv2.warpAffine(img, mat, (w, h))
```
Image below is the Image Augumentation result.
![ImageAugumentation][ImageAugumentation]

Image below is gray scale of above images
![grayScale][grayScale]

#### 2. Data set overview

since the data set provided a validation data set, thus i do not use data set spliting for validation.

* Number of training examples = 34799
* Number of validation examples = 4410
* Number of testing examples = 12630

Here is an example of an original image and an augmented image:

![alt text][image3]

The difference between the original data set and the augmented data set is the following ...


#### 3. Model
I use a model simliar to AlexNet with smaller number kernel used in conv. layer and less node in hidden layer. It's because the input size is smaller wiuth this problem.

![finalGraph][finalGraph]


```python
x = tf.placeholder('float', [None, IMG_SIZE, IMG_SIZE, NUM_CHANNELS])
y = tf.placeholder('float', [None, NUM_CLASSES])

# Placeholder for dropout keep probability
keep_prob = tf.placeholder(tf.float32)

# Use batch normalization for all convolution layers
with slim.arg_scope([slim.conv2d], normalizer_fn=slim.batch_norm):

    net = slim.conv2d(x, 16, [3, 3], scope='conv1')  
    net = slim.max_pool2d(net, [3, 3], 1, padding='SAME', scope='pool1')

    net = slim.conv2d(net, 64, [5, 5], 3, padding='VALID', scope='conv2')  
    net = slim.max_pool2d(net, [3, 3], 1, scope='pool2')  

    net = slim.conv2d(net, 128, [3, 3], scope='conv3')  

    net = slim.conv2d(net, 128, [3, 3], scope='conv4')  

    net = slim.conv2d(net, 64, [3, 3], scope='conv5')  
    net = slim.max_pool2d(net, [3, 3], 1, scope='pool3')  

    # Final fully-connected layers
    net = tf.contrib.layers.flatten(net)
    net = slim.fully_connected(net, 1024, scope='fc4')
    net = tf.nn.dropout(net, keep_prob)
    net = slim.fully_connected(net, 1024, scope='fc5')
    net = tf.nn.dropout(net, keep_prob)
    net = slim.fully_connected(net, NUM_CLASSES, scope='fc6')

# Final output (logits)
logits = net

# Loss (data loss and regularization loss) and optimizer
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = y))
optimizer = OPT.minimize(loss)

# Prediction (used during inference)
predictions = tf.argmax(logits, 1)

# Accuracy metric calculation
correct_prediction = tf.equal(predictions, tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
```


#### Training

I create a class batchGenerator to manage batch train, which perform batch mangement, it helps the train fucntion cleaner. It also have shuffle function, which randomize the datasets.

##### hyperparameters
* learning rate 0.005
* drop out rate 0.5
* optimizor
* total epochs
* batch size 128
* optimizer: gradient descent algorithm

```python
class batchGenerator:
    def __init__(self, x, y, shuffle= True):
        self.dataX = x
        self.dataY = y
        self.totalData = len(self.dataX)
        if shuffle:
            self.shuffle()

    def printLog(self):
        if len(self.dataX):
            print(str(totalData-len(self.dataX))+"/"+str(totalData) , end = '\r')
        else:
            print(str(totalData-len(self.dataX))+"/"+str(totalData))

    def shuffle(self):
        newOrder = np.arange(len(self.dataX))
        np.random.shuffle(newOrder)
        self.dataX = self.dataX[newOrder]
        self.dataY = self.dataY[newOrder]

    def hasNext(self):
        return len(self.dataX)>0

    def next_batch(self,size):
        if(len(self.dataX) < size):
            size = len(self.dataX)
        tempX = self.dataX[0: size]
        self.dataX = self.dataX[size:]
        tempY = self.dataY[0: size]
        self.dataY = self.dataY[size:]

        return np.array(tempX), np.array(tempY)
```

#### Training process

![accResult][accResult]
The chart above is the accuracy of training and validation during 40 epoch training.

The final model accuracy were:
* training set accuracy of 0.9993
* validation set accuracy of 0.9782
* test set accuracy of 0.9561

The code for calculating the accuracy of the model is located in the the [Ipython notebook](https://github.com/hmtai6/German-Traffic-Sign-Classification/blob/master/Traffic_Sign_Classifier-tfSlim%20.ipynb).

I have try different model.

* Conv 3x3x16 strides 1X1
* Conv 5x5x64  strides 2X2
* Conv 3x3x128  strides 1X1
* Fc 4096
* Fc 1024
* Fc 43
ps no normalization applied

![failGraph][failGraph]
![failResult][failResult]

The model need longer time to train to 0.8 acc., which can consider a inefficient model design. The major mistake in this model wasn't apply batch normalization in the training, which make it need longer to train and do not fully use the nonlinearity of relu.


### Test a Model on New Images

#### 1.  Six new German traffic signs

Here are six German traffic signs that I found on the web:
![newImgs][newImgs]

The fifth image much more difficult, it's because the image not only include single sign, thus it might make model harder to classify into a class.

For others, it have different lighting compare to dataset, they are  completly new images, it will be a chanllege to the model.

#### Predictions result

Here are the results of the prediction:

| Image			        |     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| Roundabout mandatory      		| Roundabout mandatory				|
| Ahead only      			| Ahead only 									|
| Yield					| Yield											|
| Speed limit (30km/h)  		| Speed limit (30km/h)		 				|
| Road work		| Road work      							|
|General caution	| General caution     							|


The model was able to correctly guess 6 of the 6 traffic signs, which gives an accuracy of 100%.

#### Vislualization softmax predictions
!["newImgResult"][newImgResult]
