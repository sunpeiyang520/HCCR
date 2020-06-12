# -*- coding=utf-8 -*-
import tensorflow as tf

NUM_LABELS = 3755
stddev = 0.01
prob = 0.5  # dropout


def parametric_relu(_x):
    alphas = tf.get_variable('alpha', _x.get_shape()[-1],
                             initializer=tf.constant_initializer(0.0),
                             dtype=tf.float32)
    pos = tf.nn.relu(_x)
    neg = alphas * (_x - abs(_x)) * 0.5
    return pos + neg


def hccr_cnnnet(input_tensor, train, regularizer, channels):
    conv1_deep = 64
    conv2_deep = 64
    conv3_deep = 128
    conv4_deep = 128
    conv5_deep = 256
    conv6_deep = 256
    conv7_deep = 256

    conv8_deep = 512
    conv9_deep = 512
    conv10_deep = 512

    conv11_deep = 512
    conv12_deep = 512
    conv13_deep = 512

    fc1_num = 3755
    fc2_num = 3755


    with tf.variable_scope('layer0-bn'):
        bn0 = tf.layers.batch_normalization(input_tensor, training=train, name='bn0')

    with tf.variable_scope('layer1-conv1'):
        conv1_weights = tf.get_variable("weight", [3, 3, channels, conv1_deep],
                                        initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv1_biases = tf.get_variable("bias", [conv1_deep], initializer=tf.constant_initializer(0.0))
        conv1 = tf.nn.conv2d(bn0, conv1_weights, strides=[1, 1, 1, 1], padding='SAME')
        bn_conv1 = tf.layers.batch_normalization(tf.nn.bias_add(conv1, conv1_biases), training=train, name='bn_conv1')
        prelu1 = parametric_relu(bn_conv1)

    with tf.variable_scope("layer2-conv2"):
        conv2_weights = tf.get_variable("weight", [3, 3, conv1_deep, conv2_deep],
                                        initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv2_biases = tf.get_variable("bias", [conv2_deep], initializer=tf.constant_initializer(0.0))
        conv2 = tf.nn.conv2d(prelu1, conv2_weights, strides=[1, 1, 1, 1], padding='SAME')
        bn_conv2 = tf.layers.batch_normalization(tf.nn.bias_add(conv2, conv2_biases), training=train, name='bn_conv2')
        prelu2 = parametric_relu(bn_conv2)

    with tf.name_scope("layer3-pool1"):
        pool1 = tf.nn.max_pool(prelu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    with tf.variable_scope("layer4-conv3"):
        conv3_weights = tf.get_variable("weight", [3, 3, conv2_deep, conv3_deep],
                                        initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv3_biases = tf.get_variable("bias", [conv3_deep], initializer=tf.constant_initializer(0.0))
        conv3 = tf.nn.conv2d(pool1, conv3_weights, strides=[1, 1, 1, 1], padding='SAME')
        bn_conv3 = tf.layers.batch_normalization(tf.nn.bias_add(conv3, conv3_biases), training=train, name='bn_conv3')
        prelu3 = parametric_relu(bn_conv3)

    with tf.variable_scope("layer5-conv4"):
        conv4_weights = tf.get_variable("weight", [3, 3, conv3_deep, conv4_deep],
                                        initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv4_biases = tf.get_variable("bias", [conv4_deep], initializer=tf.constant_initializer(0.0))
        conv4 = tf.nn.conv2d(prelu3, conv4_weights, strides=[1, 1, 1, 1], padding='SAME')
        bn_conv4 = tf.layers.batch_normalization(tf.nn.bias_add(conv4, conv4_biases), training=train, name='bn_conv4')
        prelu4 = parametric_relu(bn_conv4)

    with tf.name_scope("layer6-pool2"):
        pool2 = tf.nn.max_pool(prelu4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    with tf.variable_scope("layer7-conv5"):
        conv5_weights = tf.get_variable("weight", [3, 3, conv4_deep, conv5_deep],
                                        initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv5_biases = tf.get_variable("bias", [conv5_deep], initializer=tf.constant_initializer(0.0))
        conv5 = tf.nn.conv2d(pool2, conv5_weights, strides=[1, 1, 1, 1], padding='SAME')
        bn_conv5 = tf.layers.batch_normalization(tf.nn.bias_add(conv5, conv5_biases), training=train, name='bn_conv5')
        prelu5 = parametric_relu(bn_conv5)

    with tf.variable_scope("layer8-conv6"):
        conv6_weights = tf.get_variable("weight", [3, 3, conv5_deep, conv6_deep],
                                        initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv6_biases = tf.get_variable("bias", [conv6_deep], initializer=tf.constant_initializer(0.0))
        conv6 = tf.nn.conv2d(prelu5, conv6_weights, strides=[1, 1, 1, 1], padding='SAME')
        bn_conv6 = tf.layers.batch_normalization(tf.nn.bias_add(conv6, conv6_biases), training=train, name='bn_conv6')
        prelu6 = parametric_relu(bn_conv6)

    with tf.variable_scope("layer9-conv8"):
        conv7_weights = tf.get_variable("weight", [3, 3, conv6_deep, conv7_deep],
                                        initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv7_biases = tf.get_variable("bias", [conv7_deep], initializer=tf.constant_initializer(0.0))
        conv7 = tf.nn.conv2d(prelu6, conv7_weights, strides=[1, 1, 1, 1], padding='SAME')
        bn_conv7 = tf.layers.batch_normalization(tf.nn.bias_add(conv7, conv7_biases), training=train, name='bn_conv7')
        prelu7 = parametric_relu(bn_conv7)

    with tf.name_scope("layer10-pool3"):
        pool3 = tf.nn.max_pool(prelu7, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    with tf.variable_scope("layer11-conv8"):
        conv8_weights = tf.get_variable("weight", [3, 3, conv7_deep, conv8_deep],
                                        initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv8_biases = tf.get_variable("bias", [conv8_deep], initializer=tf.constant_initializer(0.0))
        conv8 = tf.nn.conv2d(pool3, conv8_weights, strides=[1, 1, 1, 1], padding='SAME')
        bn_conv8 = tf.layers.batch_normalization(tf.nn.bias_add(conv8, conv8_biases), training=train,
                                                 name='bn_conv8')
        prelu8 = parametric_relu(bn_conv8)

    with tf.variable_scope("layer12-conv9"):
        conv9_weights = tf.get_variable("weight", [3, 3, conv8_deep, conv9_deep],
                                        initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv9_biases = tf.get_variable("bias", [conv9_deep], initializer=tf.constant_initializer(0.0))
        conv9 = tf.nn.conv2d(prelu8, conv9_weights, strides=[1, 1, 1, 1], padding='SAME')
        bn_conv9 = tf.layers.batch_normalization(tf.nn.bias_add(conv9, conv9_biases), training=train,
                                                 name='bn_conv9')
        prelu9 = parametric_relu(bn_conv9)

    with tf.variable_scope("layer13-conv10"):
        conv10_weights = tf.get_variable("weight", [3, 3, conv9_deep, conv10_deep],
                                         initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv10_biases = tf.get_variable("bias", [conv10_deep], initializer=tf.constant_initializer(0.0))
        conv10 = tf.nn.conv2d(prelu9, conv10_weights, strides=[1, 1, 1, 1], padding='SAME')
        bn_conv10 = tf.layers.batch_normalization(tf.nn.bias_add(conv10, conv10_biases), training=train,
                                                  name='bn_conv10')
        prelu10 = parametric_relu(bn_conv10)

    with tf.name_scope("layer14-pool4"):
        pool4 = tf.nn.max_pool(prelu10, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    with tf.variable_scope("layer15-conv11"):
        conv11_weights = tf.get_variable("weight", [3, 3, conv10_deep, conv11_deep],
                                        initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv11_biases = tf.get_variable("bias", [conv11_deep], initializer=tf.constant_initializer(0.0))
        conv11 = tf.nn.conv2d(pool4, conv11_weights, strides=[1, 1, 1, 1], padding='SAME')
        bn_conv11 = tf.layers.batch_normalization(tf.nn.bias_add(conv11, conv11_biases), training=train,
                                                 name='bn_conv11')
        prelu11 = parametric_relu(bn_conv11)

    with tf.variable_scope("layer16-conv12"):
        conv12_weights = tf.get_variable("weight", [3, 3, conv11_deep, conv12_deep],
                                        initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv12_biases = tf.get_variable("bias", [conv12_deep], initializer=tf.constant_initializer(0.0))
        conv12 = tf.nn.conv2d(prelu11, conv12_weights, strides=[1, 1, 1, 1], padding='SAME')
        bn_conv12 = tf.layers.batch_normalization(tf.nn.bias_add(conv12, conv12_biases), training=train,
                                                 name='bn_conv12')
        prelu12 = parametric_relu(bn_conv12)

    with tf.variable_scope("layer17-conv13"):
        conv13_weights = tf.get_variable("weight", [3, 3, conv12_deep, conv13_deep],
                                         initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv13_biases = tf.get_variable("bias", [conv13_deep], initializer=tf.constant_initializer(0.0))
        conv13 = tf.nn.conv2d(prelu12, conv13_weights, strides=[1, 1, 1, 1], padding='SAME')
        bn_conv13 = tf.layers.batch_normalization(tf.nn.bias_add(conv13, conv13_biases), training=train,
                                                  name='bn_conv13')
        prelu13 = parametric_relu(bn_conv13)

    with tf.name_scope("layer18-pool5"):
        pool5 = tf.nn.max_pool(prelu13, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    pool_shape = pool5.get_shape().as_list()
    nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
    reshaped = tf.reshape(pool5, [-1, nodes])

    with tf.variable_scope('layer19-fc1'):
        fc1_weights = tf.get_variable("weight", [nodes, fc1_num],
                                      initializer=tf.truncated_normal_initializer(stddev=stddev))
        if regularizer != None:
            tf.add_to_collection('losses', regularizer(fc1_weights))
        fc1_biases = tf.get_variable("bias", [fc1_num], initializer=tf.constant_initializer(0))
        bn_fc1 = tf.layers.batch_normalization(tf.matmul(reshaped, fc1_weights) + fc1_biases, training=train,
                                               name='bn_fc1')
        fc1 = parametric_relu(bn_fc1)
        if train:
            fc1 = tf.nn.dropout(fc1, prob)

    with tf.variable_scope('layer20-fc2'):
        fc2_weights = tf.get_variable("weight", [fc1_num, fc2_num],
                                      initializer=tf.truncated_normal_initializer(stddev=stddev))
        if regularizer != None:
            tf.add_to_collection('losses', regularizer(fc2_weights))
        fc2_biases = tf.get_variable("bias", [fc2_num], initializer=tf.constant_initializer(0))
        bn_fc2 = tf.layers.batch_normalization(tf.matmul(fc1, fc2_weights) + fc2_biases, training=train,
                                               name='bn_fc2')
        fc2 = parametric_relu(bn_fc2)
        if train:
            fc2 = tf.nn.dropout(fc2, prob)

    with tf.variable_scope('layer21-output'):
        fc3_weights = tf.get_variable("weight", [fc2_num, NUM_LABELS],
                                      initializer=tf.truncated_normal_initializer(stddev=stddev))
        if regularizer != None:
            tf.add_to_collection('losses', regularizer(fc3_weights))
        fc3_biases = tf.get_variable("bias", [NUM_LABELS], initializer=tf.constant_initializer(0.1))
        logit = tf.matmul(fc2, fc3_weights) + fc3_biases

    return logit