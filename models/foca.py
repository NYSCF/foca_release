'''
  ______                  
 |  ____|           /\    
 | |__ ___   ___   /  \   
 |  __/ _ \ / __| / /\ \  
 | | | (_) | (__ / ____ \ 
 |_|  \___/ \___/_/    \_\
                              
Focus Analysis Deep Learning Network 
'''
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam

@tf.function
def ranked_probability_score(predictions,targets,dim=1):
    '''
    Ranked Probability Score: measuring how well predictions match targets (used in Yang, et al.)
    
    Used as a loss function for Google's Microscope Image Quality analysis 
    deep learning architecture

    Args:
        predictions: np.ndarray
            probabilistic predictions per class per sample
            size (samples x classes)

        targets: np.ndarray
            probabilistic predictions per class per sample
            size (samples x classes)

        dim: int
            dimension along which to cumulatively sum predictions and targets
            should always be 1 if predictions and targets are of size (samples x classes)
    
    Returns:
        rps: float
            ranked probability score 
    '''
    predictions = tf.convert_to_tensor(predictions,name='predictions')
    targets = tf.convert_to_tensor(targets,name='targets')

    if predictions.dtype.is_floating and targets.dtype.is_integer:
        targets = tf.cast(targets,dtype=predictions.dtype)

    # cumulative distribution functions
    cdf_pred = tf.cumsum(predictions,dim)
    cdf_target = tf.cumsum(targets,dim)
    values = (cdf_pred-cdf_target) ** 2

    rps = tf.reduce_sum(values,dim)
    return rps

@tf.function
def soft_acc(y_true,y_pred):
    '''
    Soft accuracy: patch-level accuracy of rounded predictions

    Args:
        y_true: tf.Tensor
            true classes

        y_pred: tf.Tensor
            predicted classes

    Returns:
        acc: float
            mean of absolute difference between y_true and y_pred
    '''
    y_true =tf.math.round(y_true)
    acc = 1-tf.math.reduce_mean(tf.math.abs(y_true-y_pred))
    return acc


class FocA():
    '''
    Class for accessing focus analysis deep learning network
    
    Attributes:
        model: Keras sequential model instance
            model instance to be used for generating focal class predictions

        num_classes: int
            number of classes data is sorted into
            default is 1 (in-focus, out-of-focus, in-between)
        
        input_shape: tuple
            dimensions of input into Keras model, (rows,columns,channels)

        early_stop: tf.keras.callbacks.EarlyStopping
            used for monitoring statistics for validation set performance
        
        strides: int
            stride of convolutional layers in pixels

        filter_size: tuple of ints
            size of convolutional filters on first and second convolutional layers in pixels
        
        dense_size: int
            number of units in dense layer
        
        num_filters: tuple of ints
            number of convolutional filters on first and second convolutional layers

        seed: int
            random seed

    
    Methods:
        __init__:
            initialize early stopping and instantiate keras model
        

        instantiate_model:
            returns focal analysis model taking input_shape-sized inputs
            and outputting num_classes predictions associated with each class
    '''

    model = None
    num_classes = 3
    input_shape = (150,150,1)
    early_stop = None
    strides = 2
    filter_size = (3,5)
    dense_size = 512
    num_filters = (32,64)
    seed = 10

    def __init__(self,input_shape=input_shape,num_classes=num_classes,
                    lrn_rate=5e-5,num_filters=(32,64),
                    filter_size=(3,5),strides=2,dense_size=512,weights=None):

        self.num_filters = num_filters
        self.filter_size = filter_size
        self.strides = strides
        self.dense_size= dense_size
        self.early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss',min_delta=0.01,mode='min',patience=3,verbose=1,restore_best_weights=True)
        if weights is not None:
            self.model = tf.keras.models.load_model(weights,compile=False,custom_objects={'soft_acc':soft_acc})
        else:
            self.model = self.instantiate_model(num_classes,input_shape)
            self.model.compile(loss='binary_crossentropy',optimizer=Adam(learning_rate=lrn_rate),
                                metrics=[soft_acc])

        

    def instantiate_model(self,num_classes,input_shape):
    
        model = tf.keras.models.Sequential(name='foca')
        #     model.add()
        model.add(layers.InputLayer(input_shape=input_shape))
        for i in range(len(self.num_filters)):
            model.add(layers.Conv2D(self.num_filters[i],self.filter_size[i],activation='relu',padding='same',name='conv%d'%(i+1)))
            model.add(layers.MaxPool2D(pool_size=(2,2),strides=self.strides,name='pool%d'%(i+1)))
        # model.add(layers.Conv2D(self.conv2_filters,self.filter_size,activation='relu',padding='same',dilation_rate=1,name='conv2'))
        # model.add(layers.MaxPool2D(pool_size=(2,2),strides=2,name='pool2'))
        model.add(layers.Flatten())
        model.add(layers.Dense(self.dense_size,activation='relu',name='fc3',kernel_regularizer=tf.keras.regularizers.l2(l=0.001)))
        model.add(layers.Dropout(0.4,name='dropout3'))
        if num_classes <2:
            activation = 'sigmoid'
        else:
            activation = 'softmax'
        model.add(layers.Dense(num_classes,activation=activation,name='fc4',kernel_regularizer=tf.keras.regularizers.l2(l=0.01)))
        return model

