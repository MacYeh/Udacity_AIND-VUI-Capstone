from keras import backend as K
from keras.models import Model
from keras.layers import (BatchNormalization, Conv1D, Dense, Input, TimeDistributed, Activation, Bidirectional, SimpleRNN, GRU, LSTM, MaxPooling1D, AveragePooling1D, Dropout)

def simple_rnn_model(input_dim, output_dim=29):
    """ Build a recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add recurrent layer
    simp_rnn = GRU(output_dim, return_sequences=True, 
                 implementation=2, name='rnn')(input_data)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(simp_rnn)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def rnn_model(input_dim, units, activation, output_dim=29):
    """ Build a recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add recurrent layer
    simp_rnn = GRU(units, activation=activation,
        return_sequences=True, implementation=2, name='rnn')(input_data)
    # TODO: Add batch normalization 
    bn_rnn = BatchNormalization(axis=-1)(simp_rnn)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bn_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model


def cnn_rnn_model(input_dim, filters, kernel_size, conv_stride,
    conv_border_mode, units, output_dim=29):
    """ Build a recurrent + convolutional network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add convolutional layer
    conv_1d = Conv1D(filters, kernel_size, 
                     strides=conv_stride, 
                     padding=conv_border_mode,
                     activation='relu',
                     name='conv1d')(input_data)
    # Add batch normalization
    bn_cnn = BatchNormalization(name='bn_conv_1d')(conv_1d)
    # Add a recurrent layer
    simp_rnn = SimpleRNN(units, activation='relu',
        return_sequences=True, implementation=2, name='rnn')(bn_cnn)
    # TODO: Add batch normalization
    bn_rnn = BatchNormalization(axis=-1)(simp_rnn)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bn_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: cnn_output_length(
        x, kernel_size, conv_border_mode, conv_stride)
    print(model.summary())
    return model

def cnn_output_length(input_length, filter_size, border_mode, stride,
                       dilation=1):
    """ Compute the length of the output sequence after 1D convolution along
        time. Note that this function is in line with the function used in
        Convolution1D class from Keras.
    Params:
        input_length (int): Length of the input sequence.
        filter_size (int): Width of the convolution kernel.
        border_mode (str): Only support `same` or `valid`.
        stride (int): Stride size used in 1D convolution.
        dilation (int)
    """
    if input_length is None:
        return None
    assert border_mode in {'same', 'valid'}
    dilated_filter_size = filter_size + (filter_size - 1) * (dilation - 1)
    if border_mode == 'same':
        output_length = input_length
    elif border_mode == 'valid':
        output_length = input_length - dilated_filter_size + 1
    return (output_length + stride - 1) // stride

def cnn_output_length_2(input_length, filter_size, border_mode, stride,
                       dilation=1):
    """ Compute the length of the output sequence after 1D convolution along
        time. Note that this function is in line with the function used in
        Convolution1D class from Keras.
    Params:
        input_length (int): Length of the input sequence.
        filter_size (int): Width of the convolution kernel.
        border_mode (str): Only support `same` or `valid`.
        stride (int): Stride size used in 1D convolution.
        dilation (int)
    """
    if input_length is None:
        return None
    assert border_mode in {'same', 'valid'}
    dilated_filter_size = filter_size + (filter_size - 1) * (dilation - 1)
    if border_mode == 'same':
        output_length = input_length
    elif border_mode == 'valid':
        output_length = input_length - dilated_filter_size + 1
    return ((output_length + stride - 1) // stride) // 2

def deep_rnn_model(input_dim, units, recur_layers, output_dim=29):
    """ Build a deep recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    input = input_data
    # TODO: Add recurrent layers, each with batch normalization
    # Loop the layer
    for i in range(recur_layers):
        rnn = GRU(units, activation='tanh', return_sequences=True, implementation=2)(input)
        batchnorm = BatchNormalization(axis=-1)(rnn)
        input = batchnorm
 
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(batchnorm)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def bidirectional_rnn_model(input_dim, units, output_dim=29):
    """ Build a bidirectional recurrent network for speech
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # TODO: Add bidirectional recurrent layer
    bidir_rnn = Bidirectional(GRU(units, return_sequences=True), merge_mode='concat', name='bi_rnn')(input_data)
    batchnorm = BatchNormalization(axis=-1)(bidir_rnn)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(batchnorm)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def cnn_rnn_bidirectional_model(input_dim, filters, kernel_size, conv_stride,
    conv_border_mode, units, output_dim=29):
    """ Build a CNN with Bidirectional RNN following for the time-series sequences
    """
    # Main accoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Use CNN 
    conv_1d = Conv1D(filters, kernel_size, 
                     strides=conv_stride, 
                     padding=conv_border_mode,
                     activation='relu',
                     name='conv1d')(input_data)
    # BatchNorm
    bn_cnn = BatchNormalization(axis=-1, name='bn_conv_1d')(conv_1d)
    # Dropout (Optional)
    # Add bidirectional recurrent layer
    bidir_rnn = Bidirectional(GRU(units, activation='relu', return_sequences=True), 
                              merge_mode='concat', name='bi_rnn')(bn_cnn)
    # BatchNorm
    bn_bidir_rnn = BatchNormalization(axis=-1, name='bn_rnn')(bidir_rnn)
    # Add additional Dropout (Optional)
    # Dense Layer and Softmax
    time_dense_1 = TimeDistributed(Dense(128))(bn_bidir_rnn)
    # Add additional Dropout (Optional)
    time_dense_2 = TimeDistributed(Dense(output_dim))(time_dense_1)
    y_pred = Activation('softmax', name='softmax')(time_dense_2)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: cnn_output_length(
        x, kernel_size, conv_border_mode, conv_stride)
    print(model.summary())
    return model

def final_model_1(input_dim, filters, kernel_size, conv_stride,
    conv_border_mode, units, output_dim=29):
    """ Build a deep network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # TODO: Specify the layers in your network
    # Use CNN 
    conv_1d = Conv1D(filters, kernel_size, 
                     strides=conv_stride, 
                     padding=conv_border_mode,
                     activation='relu',
                     name='conv1d')(input_data)
    # BatchNorm
    bn_cnn = BatchNormalization(axis=-1, name='bn_conv_1d')(conv_1d)
    # Pooling
    # Dropout (Optional)
    bn_cnn_drop1 = Dropout(0.25)(bn_cnn)
    # Add bidirectional recurrent layer
    bidir_rnn = Bidirectional(GRU(units, activation='relu', return_sequences=True, recurrent_dropout=0.25), merge_mode='concat', name='bi_rnn')(bn_cnn_drop1)
    # BatchNorm
    bn_bidir_rnn = BatchNormalization(axis=-1, name='bn_rnn')(bidir_rnn)
    # Add additional Dropout (Optional)
    bn_bidir_rnn_drop = TimeDistributed(Dropout(0.25))(bn_bidir_rnn)
    # Dense Layer and Softmax
    time_dense_1 = TimeDistributed(Dense(256))(bn_bidir_rnn_drop)
    # Add additional Dropout (Optional)
    time_dense_1_drop = TimeDistributed(Dropout(0.25))(time_dense_1)
    time_dense_2 = TimeDistributed(Dense(output_dim))(time_dense_1_drop)
    
    # TODO: Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense_2)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    # TODO: Specify model.output_length
    model.output_length = lambda x: cnn_output_length(x, kernel_size, conv_border_mode, conv_stride)
    print(model.summary())
    return model

def final_model_2(input_dim, filters, kernel_size, conv_stride, conv_border_mode, units, recur_layers, output_dim=29):
    """ Build a deep network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # TODO: Specify the layers in your network
    # 1st-layer CNN
    conv_1d = Conv1D(filters, kernel_size, strides=conv_stride, padding=conv_border_mode, activation='relu', name='conv1d_1')(input_data)
    # BatchNorm
    bn_cnn = BatchNormalization(axis=-1, name='batch_norm_1')(conv_1d)
    # Pooling
    #bn_cnn_maxpool = AveragePooling1D(pool_size=2, padding='valid', name='avg_pool_1')(bn_cnn)
    # Dropout (Optional)
    #bn_cnn_drop1 = Dropout(0.25, name='dropout1')(bn_cnn_avgpool)
    
    #2nd-layer CNN
    conv_1d_2 = Conv1D(filters*2, kernel_size, strides=conv_stride, padding=conv_border_mode, activation='relu', name='conv1d_2')(bn_cnn)
    # BatchNorm
    bn_cnn_2 = BatchNormalization(axis=-1, name='batch_norm_2')(conv_1d_2)
    # Pooling
    bn_cnn_maxpool_2 = MaxPooling1D(pool_size=2, padding='valid', name='max_pool_2')(bn_cnn_2)
    # Dropout (Optional)
    #bn_cnn_drop2 = Dropout(0.25, name='dropout2')(bn_cnn_2)
        
    # Add Deep RNN Layer
    input = bn_cnn_maxpool_2
    for i in range(recur_layers):
        rnn = GRU(units, activation='relu', return_sequences=True, implementation=2, dropout=0.25)(input)
        batchnorm_rnn = BatchNormalization(axis=-1)(rnn)
        input = batchnorm_rnn

    # Dense Layer and Softmax
    time_dense_1 = TimeDistributed(Dense(128))(batchnorm_rnn)
    # Add additional Dropout (Optional)
    #time_dense_1_drop = TimeDistributed(Dropout(0.25))(time_dense_1)
    time_dense_2 = TimeDistributed(Dense(output_dim))(time_dense_1)
    
    # TODO: Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense_2)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    # TODO: Specify model.output_length
    #model.output_length = lambda x: cnn_output_length_2(x, kernel_size, conv_border_mode, conv_stride)
    model.output_length = lambda x: cnn_output_length_2(cnn_output_length(x, kernel_size, conv_border_mode, conv_stride), kernel_size, conv_border_mode, conv_stride) 
    
    print(model.summary())
    return model

def final_model_3(input_dim, filters, kernel_size, conv_stride, conv_border_mode, units, recur_layers, dropout, output_dim=29):
    """ Build a deep network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # TODO: Specify the layers in your network
    # 1st-layer CNN
    conv_1d = Conv1D(filters, kernel_size, strides=conv_stride, padding=conv_border_mode, activation='relu', name='conv1d_1')(input_data)
    # BatchNorm
    bn_cnn = BatchNormalization(axis=-1, name='batch_norm_1')(conv_1d)
    # Pooling
    #bn_cnn_maxpool = MaxPooling1D(pool_size=2, padding='valid', name='max_pool_1')(bn_cnn)
    # Dropout (Optional)
    #bn_cnn_drop1 = Dropout(0.25, name='dropout1')(bn_cnn_avgpool)
    
    #2nd-layer CNN
    conv_1d_2 = Conv1D(filters*2, kernel_size, strides=conv_stride, padding=conv_border_mode, activation='relu', name='conv1d_2')(bn_cnn)
    # BatchNorm
    bn_cnn_2 = BatchNormalization(axis=-1, name='batch_norm_2')(conv_1d_2)
    # Pooling
    bn_cnn_maxpool_2 = MaxPooling1D(pool_size=2, padding='valid', name='max_pool_2')(bn_cnn_2)
    # Dropout (Optional)
    #bn_cnn_drop2 = Dropout(0.25, name='dropout2')(bn_cnn_2)
        
    # Add Deep RNN Layer
    input = bn_cnn_maxpool_2
    for i in range(recur_layers):
        bi_rnn = Bidirectional(GRU(units, activation='relu', return_sequences=True, dropout=dropout[i], recurrent_dropout=dropout[i]), 
                      merge_mode='concat', name='bi_rnn_'+str(i))(input)
        batchnorm_rnn = BatchNormalization(axis=-1, name='recurrent_batch_'+str(i))(bi_rnn)
        input = batchnorm_rnn

    # Dense Layer and Softmax
    time_dense_1 = TimeDistributed(Dense(128), name='dense_layer_1')(batchnorm_rnn)
    # Add additional Dropout (Optional)
    #time_dense_1_drop = TimeDistributed(Dropout(0.25))(time_dense_1)
    time_dense_1_batch = TimeDistributed(BatchNormalization(axis=-1), name='batch_norm_3')(time_dense_1)
    time_dense_2 = TimeDistributed(Dense(output_dim), name='dense_layer_2')(time_dense_1_batch)
    
    # TODO: Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense_2)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    # TODO: Specify model.output_length
    #model.output_length = lambda x: cnn_output_length_2(x, kernel_size, conv_border_mode, conv_stride)
    model.output_length = lambda x: cnn_output_length_2(cnn_output_length(x, kernel_size, conv_border_mode, conv_stride), kernel_size, conv_border_mode, conv_stride) 
    
    print(model.summary())
    return model

def final_model_4(input_dim, filters, kernel_size, conv_stride, conv_border_mode, units, recur_layers, dropout, output_dim=29):
    """ Build a deep network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # TODO: Specify the layers in your network
    # 1st-layer CNN
    conv_1d = Conv1D(filters, kernel_size, strides=conv_stride, padding=conv_border_mode, activation='relu', name='conv1d_1')(input_data)
    # BatchNorm
    bn_cnn = BatchNormalization(axis=-1, name='batch_norm_1')(conv_1d)
    # Pooling
    #bn_cnn_maxpool = MaxPooling1D(pool_size=2, padding='valid', name='max_pool_1')(bn_cnn)
    # Dropout (Optional)
    #bn_cnn_drop1 = Dropout(0.25, name='dropout1')(bn_cnn_avgpool)
    
    #2nd-layer CNN
    conv_1d_2 = Conv1D(filters*2, kernel_size, strides=conv_stride, padding=conv_border_mode, activation='relu', name='conv1d_2')(bn_cnn)
    # BatchNorm
    bn_cnn_2 = BatchNormalization(axis=-1, name='batch_norm_2')(conv_1d_2)
    # Pooling
    bn_cnn_maxpool_2 = MaxPooling1D(pool_size=2, padding='valid', name='max_pool_2')(bn_cnn_2)
    # Dropout (Optional)
    #bn_cnn_drop2 = Dropout(0.25, name='dropout2')(bn_cnn_2)
        
    # Add Deep RNN Layer
    input = bn_cnn_maxpool_2
    for i in range(recur_layers):
        bi_rnn = Bidirectional(LSTM(units, activation='tanh', return_sequences=True, dropout=dropout[i], recurrent_dropout=dropout[i]), 
                      merge_mode='concat', name='bi_rnn_'+str(i))(input)
        batchnorm_rnn = BatchNormalization(axis=-1, name='recurrent_batch_'+str(i))(bi_rnn)
        input = batchnorm_rnn

    # Dense Layer and Softmax
    time_dense_1 = TimeDistributed(Dense(256, activation='relu'), name='dense_layer_1')(batchnorm_rnn)
    # Add additional Dropout (Optional)
    #time_dense_1_drop = TimeDistributed(Dropout(0.25))(time_dense_1)
    time_dense_1_batch = TimeDistributed(BatchNormalization(axis=-1), name='batch_norm_3')(time_dense_1)
    time_dense_2 = TimeDistributed(Dense(output_dim), name='dense_layer_2')(time_dense_1_batch)
    
    # TODO: Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense_2)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    # TODO: Specify model.output_length
    #model.output_length = lambda x: cnn_output_length_2(x, kernel_size, conv_border_mode, conv_stride)
    model.output_length = lambda x: cnn_output_length_2(cnn_output_length(x, kernel_size, conv_border_mode, conv_stride), kernel_size, conv_border_mode, conv_stride) 
    
    print(model.summary())
    return model

def final_model_5(input_dim, filters, kernel_size, conv_stride, dilation, conv_border_mode, units, recur_layers, dropout, output_dim=29):
    """ Build a deep network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # TODO: Specify the layers in your network
    # 1st-layer CNN
    conv_1d = Conv1D(filters, kernel_size, strides=conv_stride, dilation_rate=dilation[0], padding=conv_border_mode, activation='relu', name='conv1d_1')(input_data)
    # BatchNorm
    bn_cnn = BatchNormalization(axis=-1, name='batch_norm_1')(conv_1d)
    # Pooling
    #bn_cnn_maxpool = MaxPooling1D(pool_size=2, padding='valid', name='max_pool_1')(bn_cnn)
    # Dropout (Optional)
    #bn_cnn_drop1 = Dropout(0.25, name='dropout1')(bn_cnn_avgpool)
    
    #2nd-layer CNN
    conv_1d_2 = Conv1D(filters*2, kernel_size, strides=conv_stride, dilation_rate=dilation[1], padding=conv_border_mode, activation='relu', name='conv1d_2')(bn_cnn)
    # BatchNorm
    bn_cnn_2 = BatchNormalization(axis=-1, name='batch_norm_2')(conv_1d_2)
    # Pooling
    # bn_cnn_maxpool_2 = MaxPooling1D(pool_size=2, padding='valid', name='max_pool_2')(bn_cnn_2)
    # Dropout (Optional)
    #bn_cnn_drop2 = Dropout(0.25, name='dropout2')(bn_cnn_2)
        
    # Add Deep RNN Layer
    input = bn_cnn_2
    for i in range(recur_layers):
        bi_rnn = Bidirectional(GRU(units, activation='tanh', return_sequences=True, dropout=dropout[i], recurrent_dropout=dropout[i]), 
                      merge_mode='concat', name='bi_rnn_'+str(i))(input)
        batchnorm_rnn = BatchNormalization(axis=-1, name='recurrent_batch_'+str(i))(bi_rnn)
        input = batchnorm_rnn

    # Dense Layer and Softmax
    time_dense_1 = TimeDistributed(Dense(256, activation='relu'), name='dense_layer_1')(batchnorm_rnn)
    # Add additional Dropout (Optional)
    #time_dense_1_drop = TimeDistributed(Dropout(0.25))(time_dense_1)
    time_dense_1_batch = TimeDistributed(BatchNormalization(axis=-1), name='batch_norm_3')(time_dense_1)
    time_dense_2 = TimeDistributed(Dense(output_dim), name='dense_layer_2')(time_dense_1_batch)
    
    # TODO: Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense_2)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    # TODO: Specify model.output_length
    #model.output_length = lambda x: cnn_output_length_2(x, kernel_size, conv_border_mode, conv_stride)
    model.output_length = lambda x: cnn_output_length_2(cnn_output_length(x, kernel_size, conv_border_mode, conv_stride, dilation=dilation[0]), kernel_size, conv_border_mode, conv_stride, dilation=dilation[1]) 
    
    print(model.summary())
    return model
