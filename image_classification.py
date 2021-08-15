"""
    Part 1    

    Objective: 
        Create a CNN / transfer learn  to accurately classify 10 types of imagaes in the CIFAR dataset.
        Then progress to building a YOLO detection algorithm

    Approach: Will use MobileNetV2 as the architecture to transfer learn so that we don't have to build the classifier from scratch.
        Source - https://keras.io/api/applications/mobilenet/#mobilenetv2-function

    Background: 1 hot Encoding of labels.
        Label	Description
        0	airplane
        1	automobile
        2	bird
        3	cat
        4	deer
        5	dog
        6	frog
        7	horse
        8	ship
        9	truck

    Research:
        Starting with MNIST - https://keras.io/examples/vision/mnist_convnet/
        CIFAR CNN from scatch - https://machinelearningmastery.com/how-to-develop-a-cnn-from-scratch-for-cifar-10-photo-classification/
        MobileNetV2 transfer learning - https://developer.ridgerun.com/wiki/index.php?title=Keras_with_MobilenetV2_for_Deep_Learning

"""

from keras.datasets import cifar10
from keras.applications import MobileNetV2
from keras.applications.mobilenet import preprocess_input
from keras import layers
from keras import Model
from keras import utils

import cv2 
import numpy as np
""" ---------------------------------------------------- DATASET ----------------------------------------------------- """
class dataset_generator():
    def __init__(self,dataset_settings):
        """Loads the CIFAR10 dataset"""
        self.settings = dataset_settings

        (self.train_x, self.train_y), (self.test_x, self.test_y) = cifar10.load_data()

        self.train_y = from_number_to_array__one_hot_encoding(self.train_y) 
        self.test_y = from_number_to_array__one_hot_encoding(self.test_y)

        # Normalise from 0-1 to represent probabilities
        self.train_x = preprocess_input(self.train_x)
        self.train_y = preprocess_input(self.train_y)

    def get_n_data_rows(self,n=50000):
        """Gets n number of rows amount of the data"""
        return self.train_x[:n], self.train_y[:n], self.test_x, self.test_y

""" ----------------------------------------------------- MODEL ----------------------------------------------------- """
class custom_model():
    def __init__(self,model_settings):
        self.settings = model_settings

        self.build_and_compile_model()

    def build_and_compile_model(self):
        backbone_classifier = MobileNetV2(
            weights = 'imagenet' #imports weights used in ImageNet competition
            ,include_top = False #removes last 1000 neuron layer so you can add your own layers for classifications
            ) #imports the MobileNetV2 model and discards the last 1000 neuron layer.

        x = backbone_classifier.output #get output of mobile
        x = layers.GlobalAveragePooling2D()(x) #join between convolutional layers and dense layers - https://paperswithcode.com/method/global-average-pooling
        x = layers.Dense(1024,activation='relu')(x) #additional dense layers to learn more complex features output by MobileNet
        x = layers.Dense(1024,activation='relu')(x) 
        x = layers.Dense(512,activation='relu')(x) 

        number_of_output_classes = len(self.settings["Encodings"]) #get number of classes that will be output

        output_prediction_layer = layers.Dense(number_of_output_classes,activation='softmax')(x) #final layer with softmax activation for N classes

        self.model = Model(inputs=backbone_classifier.input,outputs=output_prediction_layer) #specify the inputs and outputs

        for layer in self.model.layers[:self.settings["LastTrainableLayers"]]: #freeze everything but the last N layers
            layer.trainable = False
        for layer in self.model.layers[self.settings["LastTrainableLayers"]:]: #freeze everything but the last N layers
            layer.trainable = True

        self.model.compile(
            optimizer='Adam'
            ,loss='categorical_crossentropy' #use this for classification aka categories
            ,metrics=['accuracy'] #good metric to start with
            )

    def train(self,train_x,train_y,test_x,test_y):
        """Used to perform training"""
        self.history = self.model.fit(
            train_x
            ,train_y
            ,epochs=self.settings["Training"]["Epochs"]
            ,batch_size=self.settings["Training"]["BatchSize"]
            #,validation_data=(test_x, test_y)
            )

    def predict(self,img):
        """Predicts what the image is and returns text"""
        if len(img.shape) == 3:
            img = np.array([img])
        pre_processed_img = preprocess_input(img)
        predictions = self.model.predict(pre_processed_img)
        max_index_location = predictions.argmax() 

        decodings = {v: k for k, v in self.settings["Encodings"].items()}
        return decodings[max_index_location], predictions[max_index_location]

""" ------------------------------------------------- UTILITY FUNCIONS ----------------------------------------------------- """
def from_number_to_array__one_hot_encoding(input_element):
    """Converts output of [2] to an array of [0,0,1] for one hot encoding."""
    return utils.to_categorical(input_element)

""" ---------------------------------------------------- CONFIG ----------------------------------------------------- """
config = {
    "Dataset" : {
        "test" : "asd"
    }
    ,"Model" : {
        "Encodings" : {
            "airplane" : 0
            ,"automobile" : 1
            ,"bird" : 2
            ,"cat" : 3
            ,"deer" : 4
            ,"dog" : 5
            ,"frog" : 6
            ,"horse" : 7
            ,"ship" : 8
            ,"truck" : 9
        }
        ,"LastTrainableLayers" : 20
        ,"Training" : {
            "Epochs" : 50 #default = 100
            ,"BatchSize" : 8 #default - 64
        }
    }
}

if __name__ == "__main__":
    print("starting...")
    dataset = dataset_generator(config["Dataset"])
    cust_model = custom_model(config["Model"])

    train_x, train_y, test_x, test_y = dataset.get_n_data_rows(n=50)
    cust_model.train(train_x, train_y, test_x, test_y)

    cust_model.model.save("model.h5")

    single_test_img = cv2.imread("bird.jpg")
    print(cust_model.predict(single_test_img))

    print("finishing...")

