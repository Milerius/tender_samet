import os
import keras
import resnet
import tensorflow as tf
from keras.models import Model
from generator import Generator
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.layers import Input, Flatten, Dense, Dropout
from optparse import OptionParser



os.environ['CUDA_VISIBLE_DEVICES'] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"

def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)

keras.backend.tensorflow_backend.set_session(get_session())

def set_model(network, num_classes, include_top=False, weights='imagenet', input_shape=(224, 224, 3)):
    """
    The function constructs the base model with 
    
    """
    
    if network == 'resnet50':
        base_model = resnet.ResNet50(include_top=include_top, weights=weights, input_shape=input_shape)
    elif network == 'resnet101':
        base_model = resnet.ResNet50(include_top=include_top, weights=weights, input_shape=input_shape)
    else:
        raise ValueError("Oops, wrong network, {} doesn't exist".format(network))
        
    x = base_model.output
    x = Flatten()(x)
    x = Dense(1024, activation='relu', name='fc1')(x)
    x = Dropout(0.7, name='dropout1')(x)
    x = Dense(128, activation='relu', name='fc2')(x)
    x = Dropout(0.5, name='dropout2')(x)
    x = Dense(num_classes, activation='sigmoid')(x)
    
    model = Model(inputs=base_model.input, outputs=x)
    
    return model


def training(network, num_classes, epochs, batch_size, part):
    """Function that trains selected network"""

    train_data = pd.read_pickle('data/dataframe/{}_train_df.pkl'.format(part))
    nb_epoch_steps = train_data.shape[0]//batch_size

    model = set_model(network, num_classes, include_top=False, weights='imagenet')

    # Freeze 15% of the early layers
    no_train = int(len(model.layers)*0.15)
    for layer in model.layers[:no_train]:
        layer.trainable=False
    for layer in model.layers[no_train:]:
        layer.trainable=True


    gen = Generator()
    train_gen = gen.generator('data/dataframe/{}_train.csv'.format(part), num_classes=3, augmentation=True)
    val_gen = gen.generator('data/dataframe/{}_val.csv'.format(part), num_classes=3, augmentation=False)

    opt = Adam(lr=0.0001, decay=1e-5)
    es = EarlyStopping(patience=5)
    chkpt = ModelCheckpoint(filepath='data/snapshots/{}_{}_{{epoch:02d}}.h5'.format(model.name, part), save_best_only=True)
    model.compile(loss='binary_crossentropy', metrics=['accuracy'],optimizer=opt)

    # Fit model
    tensorboard = TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=True)
    history = model.fit_generator(train_gen, epochs=epochs, steps_per_epoch=nb_epoch_steps,
                                validation_data=val_gen, validation_steps=8, callbacks=[es,chkpt, tensorboard])


def parse_command_line():
    version = "%prog 1.0"
    usage = "usage %prog [options]"
    parser = OptionParser(usage=usage, version=version)
   
    parser.add_option("-c", "--classes", dest="classes",
            help="Number of classes", type=int)
    parser.add_option("-e", "--epochs", dest="epochs",
            help="Number of epochs", type=int, default=10)
    parser.add_option("-b", "--batch_size", dest="batch_size",
            help="Batch size", type=int, default=96)
    parser.add_option("-p","--part", dest="part",
            help="Lung part to use ['lungs, cardio, nothing, pleural", type=str)
    parser.add_option("-n","--network", dest="network", 
            help="Type of network to use", type=str)
   
    
    return parser.parse_args() 


if __name__ == '__main__':

    (options, args) = parse_command_line()
    training(options.classes,
             options.epochs,
             options.batch_size,
             options.part,
             options.network)


