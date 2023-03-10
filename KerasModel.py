import cv2
import numpy as np
import os
import sys
import tensorflow as tf

from sklearn.model_selection import train_test_split

EPOCHS = 10
IMG_WIDTH = 32
IMG_HEIGHT = 32
NUM_CATEGORIES = 10
TEST_SIZE = 0.4


d = {'airplane' : 0, 'automobile' : 1, 'bird': 2, 'cat' : 3, 'deer' : 4, 'dog' : 5, 'frog' : 6,
    'horse': 7, 'ship' : 8, 'truck' : 9}


def main():

    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python KerasModel.py data_directory [model.h5]")

    # Get image arrays and labels for all image files
    images, labels = load_data(sys.argv[1])
    
    # Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )
    # Get a compiled neural network
    model = get_model()
    

    # Fit model on training data
    model.fit(x_train, y_train, epochs=EPOCHS)

    # Evaluate neural network performance
    model.evaluate(x_test,  y_test, verbose=2)

    # Save model to file
    if len(sys.argv) == 3:
        filename = sys.argv[2]
        model.save(filename)
        print(f"Model saved to {filename}.")



def load_data(data_dir):
    
    images = []
    labels = []
    
    # loop for categories
    for foldername in os.listdir(data_dir):
        folder = os.path.join(data_dir,foldername)
        
        # check that category is dir not file 
        if os.path.isdir(folder):
            
            # get label for image
            l = foldername

            # loop for images in each category
            for filename in os.listdir(folder):
                file = os.path.join(folder,filename)
                
                # check that image is file
                if os.path.isfile(file):
                    
                    # reading image
                    img = cv2.imread(file)

                    # resize image
                    resized = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
                    
                    # adding label in list  
                    labels.append(d[l])

                    # adding image in numpy two dimensional array 
                    images.append(np.ndarray(shape=resized.shape, dtype=resized.dtype, buffer=resized))
    
    return (images, labels)


def get_model():
    
    model = tf.keras.models.Sequential([

        # Convolutional layer. Learn 32 filters using a 3x3 kernel
        tf.keras.layers.Conv2D(
            32, (3, 3), activation="relu", input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)
        ),

        # Max-pooling layer, using 3x3 pool size
        tf.keras.layers.MaxPooling2D(pool_size=(3, 3)),

        # Flatten units
        tf.keras.layers.Flatten(),

        # Add a hidden layers with dropout
        tf.keras.layers.Dense(255, activation="relu"),
        tf.keras.layers.Dense(255, activation="relu"),
        tf.keras.layers.Dense(255, activation="relu"),
        
        tf.keras.layers.Dropout(0.5),

        # Add an output layer with output units for all 10 categories
        tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")
    ])

    # Train neural network
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    return model




if __name__ == "__main__":
    main()
