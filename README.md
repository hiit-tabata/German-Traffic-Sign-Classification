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

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

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

* Number of training examples = 34799
* Number of testing examples = 12630
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

#### 2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

The code for splitting the data into training and validation sets is contained in the fifth code cell of the IPython notebook.  

To cross validate my model, I randomly split the training data into a training set and validation set. I did this by ...

My final training set had X number of images. My validation set and test set had Y and Z number of images.

The sixth code cell of the IPython notebook contains the code for augmenting the data set. I decided to generate additional data because ... To add more data to the the data set, I used the following techniques because ...

Here is an example of an original image and an augmented image:

![alt text][image3]

The difference between the original data set and the augmented data set is the following ...


#### 3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.



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


#### 4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

I create a class batchGenerator to manage batch train, which perform batch mangement, it helps the train fucntion cleaner. It also have shuffle function, which randomize the datasets.

##### hyperparameters
* learning rate 0.005
* drop out rate 0.5
* optimizor
* total epochs
* batch size 128

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

The code for training the model is located in the eigth cell of the ipython notebook.

To train the model, I used an ....




#### 5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the ninth cell of the Ipython notebook.

My final model results were:
* training set accuracy of ?
* validation set accuracy of ?
* test set accuracy of ?

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to over fitting or under fitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?


### Test a Model on New Images

#### 1.  Six new German traffic signs

Here are six German traffic signs that I found on the web:
![newImgs][newImgs]

The fifth image much more difficult, it's because the image not only include single sign, thus it might make model harder to classify into a class.

For others, it have different lighting compare to dataset, they are  completly new images, it will be a chanllege to the model.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the tenth cell of the Ipython notebook.

Here are the results of the prediction:

| Image			        |     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| Stop Sign      		| Stop sign   									|
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

#### vislualization softmax predictions
!["newImgResult"][newImgResult]
3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| .60         			| Stop sign   									|
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ...
