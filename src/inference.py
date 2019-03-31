mport pandas as pd
import numpy as np
import cv2

import keras
import h5py
import resnet
from keras.models import Model
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten
from keras.layers import Dense, Dropout, Input, SeparableConv2D



def set_model(network, num_classes, include_top=False, input_shape=(224, 224, 3)):
    
    if network == 'resnet50':
        base_model = resnet.ResNet50(include_top=include_top, input_shape=input_shape)
    elif network == 'resnet101':
        base_model = resnet.ResNet101(include_top=include_top, input_shape=input_shape)
    else:
        raise ValueError("Oops, wrong network, {} doesn't exist".format(network))
        
    x = base_model.output
    x = Flatten()(x)
    x = Dense(1024, activation='relu', name='fc1')(x)
    x = Dropout(0.7, name='dropout1')(x)
    x = Dense(128, activation='relu', name='fc2')(x)
    x = Dropout(0.5, name='dropout2')(x)
    x = Dense(num_classes, activation='sigmoid')(x)
    
    model = Model(inputs=base_model.input, outputs=x, name=network)
    
    return model


def preprocess(image):
    
    image = cv2.resize(image, (224, 224))
    image = np.float32((image - np.min(image))/(np.max(image) - np.min(image)))
    
    return image


def _read_images(csv_file):
    
    df = pd.read_csv(csv_file)
    paths = df['Path'].tolist()
    labels = df[df.columns[5:]].values
    
    return paths, labels


def _image_batch(image_paths, batch_size=234):
    
    image_batch = np.zeros((batch_size, 224, 224, 3))
    
    for n,path in enumerate(image_paths):
        
        path = 'data/' + path
        temp_img = cv2.imread(path)
        temp_img = preprocess(temp_img)
        image_batch[n] = temp_img
        
    return image_batch




def main():
	
    #test_df = pd.read_csv('data/CheXpert-v1.0-small/valid.csv')
    paths, labels = _read_images('data/CheXpert-v1.0-small/valid.csv')
    image_batch = _image_batch(paths)
    

    true_dict = dict()

    for i in range(14):
    	true_dict[i] = labels[:,i]




    model_dict = dict()
    model_list = ['main', 'nothing', 'lungs', 'cardio', 'pleural']

    for model in model_list:
    	if model == 'main':
            network = 'resnet101'
            num_classes = 4
            model_dict[model] = set_model(network, num_classes, include_top=False)
        
    	elif model == 'nothing':
            network = 'resnet101'
            num_classes = 4
            model_dict[model] = set_model(network, num_classes, include_top=False)
        
    	elif model == 'lungs':
            network = 'resnet101'
            num_classes = 7
            model_dict[model] = set_model(network, num_classes, include_top=False)
        
    	elif model == 'cardio':
            network = 'resnet50'
            num_classes = 3
            model_dict[model] = set_model(network, num_classes, include_top=False)
    
    	elif model == 'pleural':
            network = 'resnet101'
            num_classes = 4
            model_dict[model] = set_model(network, num_classes, include_top=False)
        
    	else:
            raise ValueError("Inncorrect value encountered - {}".format(model))   
	

       
    model_paths = ['data/snapshots/resnet101_main_05.h5', 
               'data/snapshots/resnet101_nothing_03.h5',
               'data/snapshots/resnet101_lungs_01.h5',
                'data/snapshots/resnet50_cardio_03.h5',
               'data/snapshots/resnet101_pleural_03.h5']

    for n, (key, val) in enumerate(model_dict.items()):
    	model_dict[key] = model_dict[key].load(model_paths[n])

if __name__ == '__main__':
	main()
