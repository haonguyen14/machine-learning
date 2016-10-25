import tensorflow as tf


def pooling():

    kernel = [1, 2, 2, 1]
    strides = [1, 2, 2, 1]
    padding = "SAME"

    x = tf.placeholder(shape=[1, None, None, None], dtype=tf.float32)

    return (x, tf.nn.max_pool_with_argmax(x, ksize=kernel, strides=strides, padding=padding))


def unravel_pooling_argmax(argmax, shape):
    
    """
    max_index = (b * height * width * channels) + 
                (y * width * channels) + 
                (x * channels) +
                (c)

    flattened_index = [
           b,
           argmax // (width * channels),
           argmax %  (width * channels) // channels,
           c
    ]
    """
    
    WIDTH = shape[2]
    CHANNELS = shape[3]
    WC = WIDTH * CHANNELS
    
    x = argmax // WC
    y = argmax %  WC // CHANNELS
    
    return tf.pack([x, y])


def unpooling(pooled, argmax, pooled_shape, unpooled_shape):
    
    pooled_height = pooled_shape[1]
    pooled_width = pooled_shape[2]
    
    batch_size = unpooled_shape[0]
    height = unpooled_shape[1]
    width = unpooled_shape[2]
    channels = unpooled_shape[3]

    argmax = unravel_pooling_argmax(argmax, [batch_size, height, width, channels])

    t1 = tf.to_int64(tf.range(channels))
    t1 = tf.tile(t1, [batch_size*pooled_width*pooled_height])
    t1 = tf.reshape(t1, [-1, channels])
    t1 = tf.transpose(t1, perm=[1, 0])
    t1 = tf.reshape(t1, [channels, batch_size, pooled_height, pooled_width, 1])
    t1 = tf.transpose(t1, perm=[1, 0, 2, 3, 4])

    t2 = tf.to_int64(tf.range(batch_size))
    t2 = tf.tile(t2, [channels*pooled_width*pooled_height])
    t2 = tf.reshape(t2, [-1, batch_size])
    t2 = tf.transpose(t2, perm=[1, 0])
    t2 = tf.reshape(t2, [batch_size, channels, pooled_height, pooled_width, 1])

    t3 = tf.transpose(argmax, perm=[1, 4, 2, 3, 0])

    t = tf.concat(4, [t2, t3, t1])
    indices = tf.reshape(t, [pooled_width*pooled_height*channels*batch_size, 4])

    x1 = tf.transpose(pooled, perm=[0, 3, 1, 2])
    values = tf.reshape(x1, [-1])

    delta = tf.SparseTensor(indices, values, unpooled_shape)
    return tf.sparse_tensor_to_dense(tf.sparse_reorder(delta))