"""
Part 2
    Objective: 
        Find where an object is in an image assuming there is only one object in that image.
        Then progress to building a YOLO detection algorithm

    Approach: 
        1) Will use MobileNetV2 as the architecture to transfer learn so that we don't have to build the classifier from scratch.
            Link to first - https://keras.io/api/applications/mobilenet/#mobilenetv2-function
        2) Augment the CIFAR dataset to move the image elswhere

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
        Object Localisation - https://medium.com/analytics-vidhya/object-localization-using-keras-d78d6810d0be
"""

from keras.datasets import cifar10
from keras.applications import VGG16
from keras.applications.mobilenet import preprocess_input
from keras import layers
from keras import Model
from keras import utils
from keras import metrics
from keras import optimizers
from keras import losses

import cv2 
import numpy as np
import random

""" ---------------------------------------------------- DATASET ----------------------------------------------------- """
class dataset_generator():
    def __init__(self,dataset_settings,model_settings):
        """Loads the CIFAR10 dataset"""
        self.settings = dataset_settings
        self.model_settings = model_settings
        self.generate_dataset()

    def generate_dataset(self):
        """Creates a dataset by placing CIFAR10 images onto a large image to try and locate objects"""
        (train_x,train_y), (test_x,test_y) = cifar10.load_data() # Get data from dataset

        self.input_x = np.concatenate( (train_x,test_x) ) #append test to data
        self.output_y = np.concatenate( (train_y,test_y) )

    def place_img_and_coordinates(self,row):
        """Place image and get dataset"""
        big_img = np.zeros(self.settings["ImageDimensions"],dtype=np.uint8)
        small_img_to_position = row["x"]
        small_img_height = random.randint(16,32) 
        small_img_width = random.randint(16,32)
        small_img_to_position = cv2.resize(small_img_to_position,(small_img_width,small_img_height))
        big_img_height, big_img_width, _ = big_img.shape
        y_offset = random.randint(0,big_img_height-small_img_height)
        x_offset = random.randint(0,big_img_width-small_img_width)
        # Set input image 
        big_img[y_offset:y_offset+small_img_height, x_offset:x_offset+small_img_width] = small_img_to_position

        human_readable_row = {
            "HumanReadable": {
                "x" : big_img
                ,"y" : {
                    "x_centre" :  round(x_offset + small_img_width/2)
                    ,"y_centre" :  round(y_offset + small_img_height/2)
                    ,"width" : small_img_width
                    ,"height" : small_img_height
                    ,"Object" : row["y"]
                }
            }
        }
        return human_readable_row

    def generate_model_compatible_dataset(self, model_settings, n=50000):
        """Convert to format which model can interpret aka normalised array of 4 dimensions [row #, height, width, channels]"""
        self.dataset = [self.convert_row_to_be_compatable_with_model(row,model_settings) for row in self.dataset[:n]]

    def convert_row_to_be_compatable_with_model(self,row,model_settings):
        """For a single row normalise the image, and create encodings for data"""
        normalised_input_image_x = row["HumanReadable"]["x"] / 255.0 # Normalise from 0-1 to represent probabilities
        big_img_height, big_img_width, _ = normalised_input_image_x.shape

        location_encoding = np.array([ # Set output encodings
            row["HumanReadable"]["y"]["x_centre"] / big_img_width
            ,row["HumanReadable"]["y"]["y_centre"] / big_img_height
            ,row["HumanReadable"]["y"]["width"] / big_img_width
            ,row["HumanReadable"]["y"]["height"] / big_img_height
        ])
        class_encodings = [float(v==row["HumanReadable"]["y"]["Object"]) for v in range(len(model_settings["ClassEncodings"]))]
        output_encodings_y = location_encoding #np.concatenate((location_encoding,class_encodings))

        row.update({
            "ModelCompatible" : {
                "x" : normalised_input_image_x
                ,"y" : output_encodings_y
            }
        })
        return row

    def get_n_rows_model_compatible(self,n=-1):
        """Get n rows of training data"""
        if n == -1:
            n = len(self.dataset)
        input_x = np.array([row["ModelCompatible"]["x"] for row in self.dataset])
        output_y = np.array([row["ModelCompatible"]["y"] for row in self.dataset])
        return input_x, output_y

    def get_random_model_compatible_row(self,max_n=None):
        """Gets a random row from the dataset and returns input image and the machine correct coordinates"""
        if max_n == None:
            random_row = random.choice(self.dataset)
        else:
            random_row = random.choice(self.dataset[:max_n])
        return random_row["HumanReadable"]["x"], random_row["ModelCompatible"]["y"] 

    def batch_generator(self,number_of_rows_to_return=64,return_human_readable=False):
        """Generates an infinite number of batches for training and prevents storing it all in memory which has previously lead to overload"""
        while True:
            list_of_selected_indicies = [random.randint(0,self.settings["NumberOfRows"]) for _ in range(number_of_rows_to_return)]
            selected_data = [{
                "x" : self.input_x[ind]
                ,"y" : self.output_y[ind]
            } for ind in list_of_selected_indicies]

            human_readable_batch = [self.place_img_and_coordinates(row) for row in selected_data]
            human_and_model_compatible_batch = [self.convert_row_to_be_compatable_with_model(row,self.model_settings) for row in human_readable_batch]
            input_x = np.array([row["ModelCompatible"]["x"] for row in human_and_model_compatible_batch])
            output_y = np.array([row["ModelCompatible"]["y"] for row in human_and_model_compatible_batch])

            if return_human_readable:
                yield human_and_model_compatible_batch
            yield input_x, output_y # Yield inidcates this could be a generator

""" ----------------------------------------------------- MODEL ----------------------------------------------------- """
class custom_model():
    def __init__(self,model_settings,dataset_settings):
        self.settings = model_settings
        self.dataset_settings = dataset_settings
        self.build_and_compile_model()

    def build_and_compile_model(self):
        vgg_backbone = VGG16(
            input_shape=self.dataset_settings["ImageDimensions"]
            ,include_top=False
            ,weights="imagenet"
        )

        number_of_output_nodes = get_number_output_nodes_given_class_encodings_and_object_localisation(self.settings["ClassEncodings"])

        x = layers.Flatten()(vgg_backbone.output)
        output_layer = layers.Dense(number_of_output_nodes, activation='sigmoid')(x)

        self.model = Model(inputs=[vgg_backbone.input], outputs=[output_layer])

        for layer in self.model.layers[:-2]: # Only train the final layers of the model, no need to retrain
            layer.trainable = False

        self.model.compile(
            optimizer=optimizers.Adam(lr=0.001)
            ,loss='mean_squared_error' #use this for classification aka categories
            ,metrics=[metrics.BinaryCrossentropy()] #good metric to start with
            )
    
    def train(self,data_generator_function):
        """Used to perform training"""
        self.history = self.model.fit(
            data_generator_function(self.settings["Training"]["BatchSize"])
            ,steps_per_epoch = self.settings["Training"]["StepsPerEpoch"]
            ,epochs=1
            )

    def predict(self,img):
        """Predicts what the image is and returns text"""
        if len(img.shape) == 3:
            height, width, _ = self.dataset_settings["ImageDimensions"]
            img = cv2.resize(img,(height,width))
            img = np.array([img])
        pre_processed_img = preprocess_input(img)
        predictions = self.model.predict(pre_processed_img)

        # Decode locations
        location = {
            "p1p2" : {
                "x1" : round( predictions[0][self.settings["Encodings"]["x centre"]]*width - (predictions[0][self.settings["Encodings"]["width"]] * width)/2 )
                ,"y1" : round( predictions[0][self.settings["Encodings"]["y centre"]]*height - (predictions[0][self.settings["Encodings"]["height"]] * height)/2 )
                ,"x2" : round( predictions[0][self.settings["Encodings"]["x centre"]]*width + (predictions[0][self.settings["Encodings"]["width"]] * width)/2  )
                ,"y2" : round( predictions[0][self.settings["Encodings"]["y centre"]]*height + (predictions[0][self.settings["Encodings"]["height"]] * height)/2 )
            }
            ,"xywh" : {
                "x_c" : round( predictions[0][self.settings["Encodings"]["x centre"]] * width )
                ,"y_c" : round( predictions[0][self.settings["Encodings"]["y centre"]] * height )
                ,"w" : round( predictions[0][self.settings["Encodings"]["width"]] * width )
                ,"h" : round( predictions[0][self.settings["Encodings"]["height"]] * height )
            }
        }

        # Decode classification
        #max_index_location = predictions[0][4:].argmax() 
        #decodings = inverse_dictionary(self.settings["ClassEncodings"])
        classification = None #decodings[max_index_location]
        
        return classification, location, predictions 

""" ------------------------------------------------- UTILITY FUNCIONS ----------------------------------------------------- """
def from_number_to_array__one_hot_encoding(input_element):
    """Converts output of [2] to an array of [0,0,1] for one hot encoding."""
    return utils.to_categorical(input_element)

def inverse_dictionary(dictionary):
    """Inverse dictionary to make keys the values and values the keys""" 
    return {value: key for key, value in dictionary.items()}

def get_number_output_nodes_given_class_encodings_and_object_localisation(class_encodings_dicts):
    """Gets the number of output nodes given this model is a  class encodings"""
    x_c__y_c_width_height = 4
    num_class_encodings = 0 #len(class_encodings_dicts)
    return x_c__y_c_width_height + num_class_encodings

""" ---------------------------------------------------- CONFIG ----------------------------------------------------- """
config = {
    "Dataset" : {
        "ImageDimensions" : (64,64,3) # Height, width, channels (3 for colour)
        ,"NumberOfRows" : 1
    }
    ,"Model" : {
        "Encodings" : {
            "x centre" : 0
            ,"y centre" : 1
            ,"width" : 2
            ,"height" : 3
        }
        ,"ClassEncodings" : {
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
        ,"Training" : {
            "Epochs" : 5 #default = 100
            ,"BatchSize" : 16 #default - 64
            ,"StepsPerEpoch" : 64 #default - 64
        }
    }
}

if __name__ == "__main__":
    print("starting...")
    # TODO clean up code
    dataset = dataset_generator(config["Dataset"],config["Model"])
    cust_model = custom_model(config["Model"],config["Dataset"])
    
    #row = next(dataset.batch_generator(number_of_rows_to_return=1,return_human_readable=True))

    #cust_model.model.load_weights("model.h5")
    for i in range(config["Model"]["Training"]["Epochs"]):
        cust_model.train(dataset.batch_generator)
        cust_model.model.save("model.h5")
        #cust_model.model.load_weights("model.h5")
        for row in next(dataset.batch_generator(number_of_rows_to_return=4,return_human_readable=True)):
            [single_test_img,ground_truth_outputs] = [row["ModelCompatible"]["x"],row["ModelCompatible"]["y"]]
            classification, location, raw = cust_model.predict(single_test_img)
            #print(f"Pred={classification} and Truth={inverse_dictionary(config['Model']['ClassEncodings'])[ground_truth_outputs[4:].argmax()]}")
            print(f"[{i}]Loss={round(float(metrics.categorical_crossentropy(ground_truth_outputs,raw[0])),5)} Pred={raw} and Truth={ground_truth_outputs}")   
            single_test_img = cv2.rectangle(single_test_img,(location["p1p2"]["x1"],location["p1p2"]["y1"]),(location["p1p2"]["x2"],location["p1p2"]["y2"]),(255,255,255),2)
            cv2.imshow("test",single_test_img)
            cv2.waitKey(1000)

    print("finishing...")
