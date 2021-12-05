import numpy as np
import tensorflow as tf


# 复制的，不要看 !!!!!!!!!!!!!!!!!!!!!!!


epsilon = 1e-9


class CapsLayer(tf.keras.Model):
    ''' Capsule layer.
    Args:
        input: A 4-D tensor.
        num_outputs: the number of capsule in this layer.
        vec_len: integer, the length of the output vector of a capsule.
        layer_type: string, one of 'FC' or "CONV", the type of this layer,
            fully connected or convolution, for the future expansion capability
        with_routing: boolean, this capsule is routing with the
                      lower-level layer capsule.
    Returns:
        A 4-D tensor.
    '''
    def __init__(self, num_outputs_secondCaps, vec_len_secondCaps, batch_size, iter_routing,
                 embedding_size, with_routing=True, layer_type='FC', sequence_length=3, useConstantInit=False,
                 filter_size=1, num_filters=50):
        super().__init__()
        self.num_outputs_secondCaps = num_outputs_secondCaps
        self.vec_len_secondCaps = vec_len_secondCaps
        self.with_routing = with_routing
        self.layer_type = layer_type
        # self.batch_size = batch_size
        self.iter_routing = iter_routing
        self.embedding_size = embedding_size
        self.sequence_length = sequence_length
        self.useConstantInit = useConstantInit
        self.filter_size = filter_size
        self.num_filters = num_filters

        self.W = tf.Variable(tf.random.truncated_normal(shape=(1, self.embedding_size, self.num_outputs_secondCaps,
                                                               self.num_filters, self.vec_len_secondCaps), stddev=0.01, seed=1234),
                             dtype=tf.float32, name="Weight")

        if self.useConstantInit == False:
            filter_shape = [self.sequence_length, self.filter_size, 1, self.num_filters]  # sequence_length=3,
            self.conv_W = tf.Variable(tf.random.truncated_normal(filter_shape, stddev=0.1, seed=1234), name="W")
        else:
            init1 = tf.constant([[[[0.1]]], [[[0.1]]], [[[-0.1]]]])
            weight_init = tf.tile(init1, [1, self.filter_size, 1, self.num_filters])
            self.conv_W = tf.Variable(weight_init, name="W3")

        self.b = tf.Variable(tf.constant(0.0, shape=[self.num_filters]), name="b")


    def call(self, input, training=None, mask=None):
        '''
        The parameters 'kernel_size' and 'stride' will be used while 'layer_type' equal 'CONV'
        '''
        batch_size = tf.shape(input)[0]
        if self.layer_type == 'CONV':

            if not self.with_routing:

                conv = tf.nn.conv2d(
                    input,
                    self.conv_W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                conv1 = tf.nn.relu(tf.nn.bias_add(conv, self.b), name="relu")
                conv1 = tf.squeeze(conv1, axis=1)

                capsules = tf.expand_dims(conv1, -1)
                capsules = squash(capsules)

                return capsules #[batch_size, k, num_filters, 1]

        if self.layer_type == 'FC':
            if self.with_routing:
                # Reshape the input into [batch_size, k, 1, num_filters, 1]
                input = tf.reshape(input, shape=(-1, input.shape[1],
                                                 1, input.shape[-2], 1))

                # with tf.variable_scope('routing'):
                    # b_IJ: [batch_size, num_caps_l, num_caps_l_plus_1, 1, 1],
                    # about the reason of using 'batch_size', see issue #21
                b_IJ = tf.constant(np.zeros([batch_size, input.shape[1], self.num_outputs_secondCaps, 1, 1], dtype=np.float32))
                capsules = self.routing(input, b_IJ, batch_size=batch_size, iter_routing=self.iter_routing,
                                   num_caps_i=self.embedding_size, num_caps_j=self.num_outputs_secondCaps,
                                   len_u_i=self.num_filters, len_v_j=self.vec_len_secondCaps)
                capsules = tf.squeeze(capsules, axis=1)

                return capsules


    def routing(self, input, b_IJ, batch_size, iter_routing, num_caps_i, num_caps_j, len_u_i, len_v_j):

        ''' The routing algorithm.
        Args:
            input: A Tensor with [batch_size, num_caps_l, 1, length(u_i), 1]
                   shape, num_caps_l meaning the number of capsule in the layer l.
        Returns:
            A Tensor of shape [batch_size, num_caps_l_plus_1, length(v_j), 1]
            representing the vector output `v_j` in the layer l+1
        Notes:
            u_i represents the vector output of capsule i in the layer l, and
            v_j the vector output of capsule j in the layer l+1.
         '''

        # W: [num_caps_j, num_caps_i, len_u_i, len_v_j]
        W = self.W
        # W = tf.get_variable('Weight', shape=(1, num_caps_i, num_caps_j, len_u_i, len_v_j), dtype=tf.float32,
        #                     initializer=tf.random_normal_initializer(stddev=0.01, seed=1234))

        # Eq.2, calc u_hat
        # do tiling for input and W before matmul
        input = tf.tile(input, [1, 1, num_caps_j, 1, 1])
        W = tf.tile(W, [batch_size, 1, 1, 1, 1])

        u_hat = tf.matmul(W, input, transpose_a=True)
        # In forward, u_hat_stopped = u_hat; in backward, no gradient passed back from u_hat_stopped to u_hat
        u_hat_stopped = tf.stop_gradient(u_hat, name='stop_gradient')
        # line 3,for r iterations do
        for r_iter in range(iter_routing):
            # with tf.variable_scope('iter_' + str(r_iter)):
                # line 4:
            c_IJ = tf.nn.softmax(b_IJ, axis=1) * num_caps_i
            if r_iter == iter_routing - 1:
                # line 5:
                # weighting u_hat with c_IJ, element-wise in the last two dims
                s_J = tf.multiply(c_IJ, u_hat)
                # then sum in the second dim
                s_J = tf.reduce_sum(s_J, axis=1, keepdims=True)
                # line 6:
                # squash using Eq.1,
                v_J = squash(s_J)

            elif r_iter < iter_routing - 1:  # Inner iterations, do not apply backpropagation
                s_J = tf.multiply(c_IJ, u_hat_stopped)
                s_J = tf.reduce_sum(s_J, axis=1, keepdims=True)
                v_J = squash(s_J)
                # line 7:
                v_J_tiled = tf.tile(v_J, [1, num_caps_i, 1, 1, 1])
                u_produce_v = tf.matmul(u_hat_stopped, v_J_tiled, transpose_a=True)

                b_IJ += u_produce_v

        return v_J


def squash(vector):
    '''Squashing function corresponding to Eq. 1
    Args:
        vector: A tensor with shape [batch_size, 1, num_caps, vec_len, 1] or [batch_size, num_caps, vec_len, 1].
    Returns:
        A tensor with the same shape as vector but squashed in 'vec_len' dimension.
    '''
    vec_squared_norm = tf.reduce_sum(tf.square(vector), -2, keepdims=True)
    scalar_factor = vec_squared_norm / (1 + vec_squared_norm) / tf.sqrt(vec_squared_norm + epsilon)
    vec_squashed = scalar_factor * vector  # element-wise
    return(vec_squashed)


class CapsE(tf.keras.Model):
    def __init__(self, E_e, E_r, sequence_length, embedding_size, num_filters, vocab_size, iter_routing, batch_size=256,
                 num_outputs_secondCaps=1, vec_len_secondCaps=10, initialization=[], filter_size=1, useConstantInit=False):
        super(CapsE, self).__init__()
        # Placeholders for input, output
        # self.input_x = tf.placeholder(tf.int32, [batch_size, sequence_length], name="input_x")
        # self.input_y = tf.placeholder(tf.float32, [batch_size, 1], name="input_y")
        self.filter_size = filter_size
        self.num_filters = num_filters
        self.sequence_length = sequence_length
        self.embedding_size = embedding_size
        self.iter_routing = iter_routing
        self.num_outputs_secondCaps = num_outputs_secondCaps
        self.vec_len_secondCaps = vec_len_secondCaps
        self.batch_size = batch_size
        self.useConstantInit = useConstantInit

        self.E_e = E_e
        self.E_r = E_r


        self.firstCaps = CapsLayer(num_outputs_secondCaps=self.num_outputs_secondCaps, vec_len_secondCaps=self.vec_len_secondCaps,
                                with_routing=False, layer_type='CONV', embedding_size=self.embedding_size,
                                batch_size=self.batch_size, iter_routing=self.iter_routing,
                                useConstantInit=self.useConstantInit, filter_size=self.filter_size,
                                num_filters=self.num_filters, sequence_length=self.sequence_length)
        self.secondCaps = CapsLayer(num_outputs_secondCaps=self.num_outputs_secondCaps, vec_len_secondCaps=self.vec_len_secondCaps,
                                with_routing=True, layer_type='FC',
                                batch_size=self.batch_size, iter_routing=self.iter_routing,
                                embedding_size=self.embedding_size, useConstantInit=self.useConstantInit, filter_size=self.filter_size,
                                num_filters=self.num_filters, sequence_length=self.sequence_length)

        # Embedding layer
        # with tf.name_scope("embedding"):
        #     if initialization == []:
        #         self.W = tf.Variable(
        #             tf.random.truncated_normal([vocab_size, embedding_size], -math.sqrt(1.0 / embedding_size),
        #                               math.sqrt(1.0 / embedding_size), seed=1234), name="W")
        #     else:
        #         self.W = tf.Variable(initialization, name="W2")
        #
        # self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
        # self.X = tf.expand_dims(self.embedded_chars, -1)
        #
        # self.build_arch()
        # self.loss()
        # self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=500)
        #
        # tf.logging.info('Seting up the main structure')
    def call(self, inputs, training=None, mask=None):
        #The first capsule layer
        # with tf.variable_scope('FirstCaps_layer'):
        h_index, r_index, t_index = inputs[0], inputs[1], inputs[2]
        h = tf.nn.embedding_lookup(self.E_e, h_index)
        r = tf.nn.embedding_lookup(self.E_r, r_index)
        t = tf.nn.embedding_lookup(self.E_e, t_index)
        embedded_chars = tf.stack([h, r, t], axis=1)
        X = tf.expand_dims(embedded_chars, -1)


        caps1 = self.firstCaps(X)
        #The second capsule layer
        # with tf.variable_scope('SecondCaps_layer'):
        caps2 = self.secondCaps(caps1)

        v_length = tf.sqrt(tf.reduce_sum(tf.square(caps2), axis=2, keepdims=True) + epsilon)
        scores = tf.reshape(v_length, [tf.shape(h_index)[0], 1])

        return scores

    def compute_loss(self, scores, labels):
        losses = tf.square(tf.nn.softplus(scores * labels))
        total_loss = tf.reduce_mean(losses)
        return total_loss
