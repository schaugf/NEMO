import os
import argparse
from collections import Counter
from PIL import ImageFile
import pandas as pd
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.utils import plot_model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import TerminateOnNaN, CSVLogger, ModelCheckpoint, EarlyStopping

import inception_v4
from clr_callback import CyclicLR


ImageFile.LOAD_TRUNCATED_IMAGES = True
os.environ['HDF5_USE_FILE_LOCKING']='FALSE'


def Conv2Dmodel(inputShape, nClasses):
    ''' initiate keras convolutional network
    '''
    model = Sequential()
    model.add(Conv2D(256, (3, 3), input_shape=inputShape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(nClasses))
    model.add(Activation('softmax'))
    return model


def CreateModel(inputShape, args):
    print(args.model)
    if args.model == 'inception_v4':
        print('generating inception model')
        return inception_v4.inception_v4(args.nClasses, dropout_keep_prob=args.dropout, 
                                         weights='imagenet', include_top=True)
    elif args.model == 'Conv2Dmodel':
        print('generating convolutional model')
        return Conv2Dmodel(inputShape, args.nClasses)
    

def TrainModel(args):
    ''' train keras convolutional network
    '''
    os.makedirs(args.saveDir, exist_ok=True)
    
    # assemble model
    inputShape = (args.tileSize, args.tileSize, args.nChannel)
    model = CreateModel(inputShape, args)
    
    print('compiling model...')
    model.compile(loss = keras.losses.categorical_crossentropy,
                  optimizer = keras.optimizers.adam(lr=0.0001, decay=1e-6),
                  metrics = ['accuracy'])
                  
    model.summary()
    plot_model(model, to_file = os.path.join(args.saveDir, 'model.png'), show_shapes=True)

    # instantiate data flow
    train_datagen = ImageDataGenerator(
        rescale         = 1. / 255,
        shear_range     = 0.2,
        zoom_range      = 0.2,
        horizontal_flip = True,
        vertical_flip   = True,
        width_shift_range = 0.2,
        height_shift_range = 0.2,
        rotation_range = 360,
        brightness_range = (0.8, 1.2)
        )

    train_generator = train_datagen.flow_from_directory(
        args.trainDir,
        target_size = (args.tileSize, args.tileSize),
        batch_size  = args.batchSize,
        shuffle     = True,
        class_mode  = 'categorical'
        )
    
    validation_generator = train_datagen.flow_from_directory(
        args.valDir,
        target_size = (args.tileSize, args.tileSize),
        batch_size = args.batchSize,
        shuffle = True,
        class_mode = 'categorical'
        )
    
    # save class IDs for downstream analysis
    print('saving class IDs')
    classIDs = pd.DataFrame.from_dict(data=train_generator.class_indices, 
                                      orient='index')
    classIDs.to_csv(os.path.join(args.saveDir, 'classIDs.csv'), header=False)

    # generate class weight matrix
    counter = Counter(train_generator.classes)                          
    max_val = float(max(counter.values()))       
    class_weights = {class_id : max_val/num_images for class_id, num_images in counter.items()}
    
    print('class weights matrix computed as', class_weights)

    # instantiate callbacks
    csv_logger = CSVLogger(os.path.join(args.saveDir, 'training.log'))
    
    term_nan = TerminateOnNaN()
    
    earlystop = EarlyStopping(monitor='val_loss', patience=5, 
                              restore_best_weights=True)
    
    checkpointer = ModelCheckpoint(os.path.join(args.saveDir, args.modelName), 
                                   monitor='val_loss')
    
    clr = CyclicLR(base_lr=0.0001, max_lr=0.001,
                   step_size=5*train_generator.samples // args.batchSize,
                   mode='triangular')
     
    # fit model
    print('fitting model')
    model.fit_generator(train_generator,
                        validation_data = validation_generator,
                        epochs = args.epochs,
                        callbacks = [csv_logger, 
                                     term_nan, 
                                     earlystop, 
                                     checkpointer,
                                     clr],
                        class_weight = class_weights,
                        steps_per_epoch = train_generator.samples // args.batchSize,
                        validation_steps = validation_generator.samples // args.batchSize)
        
    # save model
    model.save(os.path.join(args.saveDir, args.modelName))
    

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='train NEMO model')
    parser.add_argument('--model', type=str, default='inception_v4',
        help='Conv2Dmodel or inception_v4')
    parser.add_argument('--trainDir', type=str, default='data/model/train',
        help='train directory')
    parser.add_argument('--valDir', type=str, default='data/model/validation',
        help='validation directory')
    parser.add_argument('--saveDir', type=str, default='results',
        help='save directory')
    parser.add_argument('--modelName', type=str, default='model.h5',
        help='name for saved model')
    parser.add_argument('--tileSize', type=int, default=256,
        help='image size (x=y)')
    parser.add_argument('--nChannel', type=int, default=3,
        help='number of channels')
    parser.add_argument('--nClasses', type=int, default=3,
        help='number of classes')
    parser.add_argument('--epochs', type=int, default=1, 
        help='number of epochs')
    parser.add_argument('--batchSize', type=int, default=16, 
        help='batch size')
    parser.add_argument('--dropout', type=float, default=0.5,
        help='inception dropout rate')
    parser.add_argument('--verbose', type=int, default=1,
        help='0=quiet,1=bar,2=line')
    parser.add_argument('--weights', type=float, default=None,
        help='imagenet for pretrained model load')
    args = parser.parse_args()
    
    TrainModel(args)
