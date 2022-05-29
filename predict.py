import argparse
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.models import load_model
import tensorflow_hub as hub 
import numpy as np
import matplotlib.pyplot as plt
import json
import glob
from PIL import Image




#define image size
image_size = 224

#define process image function that takes in an image (in the form of a NumPy array) 
#and return an image in the form of a NumPy array with shape (image_size, image_size, 3)
def process_image(image_np_array):
    # Convert the image into a TensorFlow Tensor
    image_tensor = tf.convert_to_tensor(image_np_array)
    # Resize it to the appropriate size using tf.image.resize
    image = tf.image.resize(image_tensor, (image_size, image_size))
    # In order to normalize the images we are going to divide the pixel values by 255. 
    image /= 255
    # convert the image back to a NumPy array using the .numpy() method
    image = image.numpy()
    return image
    
#define predict function that uses the trained network selected by the user for inference
def predict(image_path, model, top_k):
    im = Image.open(image_path)
    im_arr = np.asarray(im)
    processed_im = process_image(im_arr)
    processed_im_batch = np.expand_dims(processed_im, axis=0)
    prediction = model.predict(processed_im_batch)
    # Use tf.math.top_k function to finds values and indices of the k largest entries for the last dimension
    probs, classes = tf.math.top_k(prediction,top_k)
    probs = probs.numpy().squeeze()
    classes_label = classes.numpy().squeeze()
    classes=[class_names[str(value+1)] for value in classes_label]
    return probs, classes


if __name__ == '__main__':
    print('predict.py, running')
    
    parser = argparse.ArgumentParser()
    parser.add_argument('image_path')
    parser.add_argument('pretrained_model')
    parser.add_argument('--top_k',type=int,default=5)
    parser.add_argument('--category_names',default='label_map.json')  
    
    args = parser.parse_args()
    print(args)
    print('arg1:', args.image_path)
    print('arg2:', args.pretrained_model)
    print('top_k:', args.top_k)
    print('category_names:', args.category_names)

    with open(args.category_names, 'r') as f:
        class_names = json.load(f)
    
    image_path = args.image_path 
    

    model = load_model('my_model_udacity.h5', compile = False,
    custom_objects={'KerasLayer':hub.KerasLayer})
    
    top_k = args.top_k

    probs, classes = predict(image_path, model, top_k)
    
    print('Predicted Flower Name: \n',classes)
    print('Probabilities: \n ', probs)