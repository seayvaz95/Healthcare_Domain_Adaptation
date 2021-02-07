import tensorflow as tf
from flip_gradient import flip_gradient

class LSTM_DA(object):
    def init_weights(self, input_dim, output_dim, name, std=0.1, reg=None):
        return tf.get_variable(name, shape=[input_dim, output_dim], initializer=tf.random_normal_initializer(0.0, std),
                               regularizer=reg)

    def init_bias(self, output_dim, name):
        return tf.get_variable(name, shape=[output_dim], initializer=tf.random_normal_initializer(0.0, 0.1))

    def init_forget_bias(self, output_dim, name):
        return tf.get_variable(name, shape=[output_dim], initializer=tf.constant_initializer(1.0))

    def no_init_weights(self, input_dim, output_dim, name):
        return tf.get_variable(name, shape=[input_dim, output_dim])

    def no_init_bias(self, output_dim, name):
        return tf.get_variable(name, shape=[output_dim])

    def __init__(self, input_dim, output_dim, hidden_dim, train, dann):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.input = tf.placeholder('float', shape=[None, None, self.input_dim])
        self.target = tf.placeholder('float', shape=[None, self.output_dim])
        self.target_domain = tf.placeholder('float', shape=[None, self.output_dim])
        self.l_coef = tf.placeholder('float', shape=[])
  
        self.dann = dann


        #self.time = tf.placeholder('float', shape=[None, None])
        #self.keep_prob = tf.placeholder(tf.float32)

        if train==1:
            # self.wt = self.init_weights(1, 1, name='Time_weight', reg=None)

            self.Wiy = self.init_weights(self.input_dim, self.hidden_dim, name='Input_Hidden_weight_1', reg=None)
            self.Uiy = self.init_weights(self.hidden_dim, self.hidden_dim, name='Input_State_weight_1', reg=None)
            self.biy = self.init_bias(self.hidden_dim, name='Input_Hidden_bias_1')

            self.Wfy = self.init_weights(self.input_dim, self.hidden_dim, name='Forget_Hidden_weight_1', reg=None)
            self.Ufy = self.init_weights(self.hidden_dim, self.hidden_dim, name='Forget_State_weight_1', reg=None)
            self.bfy = self.init_forget_bias(self.hidden_dim, name='Forget_Hidden_bias_1')

            self.Wogy = self.init_weights(self.input_dim, self.hidden_dim, name='Output_Hidden_weight_1', reg=None)
            self.Uogy = self.init_weights(self.hidden_dim, self.hidden_dim, name='Output_State_weight_1', reg=None)
            self.bogy = self.init_bias(self.hidden_dim, name='Output_Hidden_bias_1')

            self.Wcy = self.init_weights(self.input_dim, self.hidden_dim, name='Cell_Hidden_weight_1', reg=None)
            self.Ucy = self.init_weights(self.hidden_dim, self.hidden_dim, name='Cell_State_weight_1', reg=None)
            self.bcy = self.init_bias(self.hidden_dim, name='Cell_Hidden_bias_1')


            self.Woy = self.init_weights(self.hidden_dim, output_dim, name='Fc_Layer_weight_1',
                                        reg=None)  # tf.contrib.layers.l2_regularizer(scale=0.001)
            self.boy = self.init_bias(output_dim, name='Fc_Layer_bias_1')

            #weights for the second LSTM
            self.Wid = self.init_weights(self.hidden_dim, self.hidden_dim, name='Input_Hidden_weight_2', reg=None)
            self.Uid = self.init_weights(self.hidden_dim, self.hidden_dim, name='Input_State_weight_2', reg=None)
            self.bid = self.init_bias(self.hidden_dim, name='Input_Hidden_bias_2')

            self.Wfd = self.init_weights(self.hidden_dim, self.hidden_dim, name='Forget_Hidden_weight_2', reg=None)
            self.Ufd = self.init_weights(self.hidden_dim, self.hidden_dim, name='Forget_State_weight_2', reg=None)
            self.bfd = self.init_forget_bias(self.hidden_dim, name='Forget_Hidden_bias_2')

            self.Wogd = self.init_weights(self.hidden_dim, self.hidden_dim, name='Output_Hidden_weight_2', reg=None)
            self.Uogd = self.init_weights(self.hidden_dim, self.hidden_dim, name='Output_State_weight_2', reg=None)
            self.bogd = self.init_bias(self.hidden_dim, name='Output_Hidden_bias_2')

            self.Wcd = self.init_weights(self.hidden_dim, self.hidden_dim, name='Cell_Hidden_weight_2', reg=None)
            self.Ucd = self.init_weights(self.hidden_dim, self.hidden_dim, name='Cell_State_weight_2', reg=None)
            self.bcd = self.init_bias(self.hidden_dim, name='Cell_Hidden_bias_2')

            #self.W_decomp = self.init_weights(self.hidden_dim, self.hidden_dim, name='Decomposition_Hidden_weight',
                                              #reg=None)
            #self.b_decomp = self.init_bias(self.hidden_dim, name='Decomposition_Hidden_bias_enc')

            self.Wod = self.init_weights(self.hidden_dim, output_dim, name='Fc_Layer_weight_2',
                                         reg=None)  # tf.contrib.layers.l2_regularizer(scale=0.001)
            self.bod = self.init_bias(output_dim, name='Fc_Layer_bias_2')

        else:


            self.Wiy = self.no_init_weights(self.input_dim, self.hidden_dim, name='Input_Hidden_weight_1')
            self.Uiy = self.no_init_weights(self.hidden_dim, self.hidden_dim, name='Input_State_weight_1')
            self.biy = self.no_init_bias(self.hidden_dim, name='Input_Hidden_bias_1')

            self.Wfy = self.no_init_weights(self.input_dim, self.hidden_dim, name='Forget_Hidden_weight_1')
            self.Ufy = self.no_init_weights(self.hidden_dim, self.hidden_dim, name='Forget_State_weight_1')
            self.bfy = self.no_init_bias(self.hidden_dim, name='Forget_Hidden_bias_1')

            self.Wogy = self.no_init_weights(self.input_dim, self.hidden_dim, name='Output_Hidden_weight_1')
            self.Uogy = self.no_init_weights(self.hidden_dim, self.hidden_dim, name='Output_State_weight_1')
            self.bogy = self.no_init_bias(self.hidden_dim, name='Output_Hidden_bias_1')

            self.Wcy = self.no_init_weights(self.input_dim, self.hidden_dim, name='Cell_Hidden_weight_1')
            self.Ucy = self.no_init_weights(self.hidden_dim, self.hidden_dim, name='Cell_State_weight_1')
            self.bcy = self.no_init_bias(self.hidden_dim, name='Cell_Hidden_bias_1')

            self.Woy = self.no_init_weights(self.hidden_dim, output_dim,
                                           name='Fc_Layer_weight_1')  # tf.contrib.layers.l2_regularizer(scale=0.001)
            self.boy = self.no_init_bias(output_dim, name='Fc_Layer_bias_1')

            self.Wid = self.no_init_weights(self.hidden_dim, self.hidden_dim, name='Input_Hidden_weight_2')
            self.Uid = self.no_init_weights(self.hidden_dim, self.hidden_dim, name='Input_State_weight_2')
            self.bid = self.no_init_bias(self.hidden_dim, name='Input_Hidden_bias_2')

            self.Wfd = self.no_init_weights(self.hidden_dim, self.hidden_dim, name='Forget_Hidden_weight_2')
            self.Ufd = self.no_init_weights(self.hidden_dim, self.hidden_dim, name='Forget_State_weight_2')
            self.bfd = self.no_init_bias(self.hidden_dim, name='Forget_Hidden_bias_2')

            self.Wogd = self.no_init_weights(self.hidden_dim, self.hidden_dim, name='Output_Hidden_weight_2')
            self.Uogd = self.no_init_weights(self.hidden_dim, self.hidden_dim, name='Output_State_weight_2')
            self.bogd = self.no_init_bias(self.hidden_dim, name='Output_Hidden_bias_2')

            self.Wcd = self.no_init_weights(self.hidden_dim, self.hidden_dim, name='Cell_Hidden_weight_2')
            self.Ucd = self.no_init_weights(self.hidden_dim, self.hidden_dim, name='Cell_State_weight_2')
            self.bcd = self.no_init_bias(self.hidden_dim, name='Cell_Hidden_bias_2')

            #self.W_decomp = self.no_init_weights(self.hidden_dim, self.hidden_dim, name='Decomposition_Hidden_weight')
            #self.b_decomp = self.no_init_bias(self.hidden_dim, name='Decomposition_Hidden_bias_enc')

            self.Wod = self.no_init_weights(self.hidden_dim, output_dim,
                                           name='Fc_Layer_weight_2')  # tf.contrib.layers.l2_regularizer(scale=0.001)
            self.bod = self.no_init_bias(output_dim, name='Fc_Layer_bias_2')

    def LSTM_Unit(self, prev_hidden_memory, x):
        prev_hidden_state, prev_cell = tf.unstack(prev_hidden_memory)

        # Input gate
        i = tf.sigmoid(tf.matmul(x, self.Wiy) + tf.matmul(prev_hidden_state, self.Uiy) + self.biy)

        # Forget Gate
        f = tf.sigmoid(tf.matmul(x, self.Wfy) + tf.matmul(prev_hidden_state, self.Ufy) + self.bfy)

        # Output Gate
        o = tf.sigmoid(tf.matmul(x, self.Wogy) + tf.matmul(prev_hidden_state, self.Uogy) + self.bogy)

        # Candidate Memory Cell
        C = tf.nn.tanh(tf.matmul(x, self.Wcy) + tf.matmul(prev_hidden_state, self.Ucy) + self.bcy)

        # Current Memory cell
        Ct = f * prev_cell + i * C

        # Current Hidden state
        current_hidden_state = o * tf.nn.tanh(Ct)

        return tf.stack([current_hidden_state, Ct])

    def T_ALSTM_Unit(self, prev_hidden_memory, x):
        prev_hidden_state, prev_cell = tf.unstack(prev_hidden_memory)

        batch_size = tf.shape(x)[0]
        #x = tf.slice(concat_input, [0, 1], [batch_size, self.input_dim])
        #t = tf.slice(concat_input, [0, 0], [batch_size, 1])

        # Dealing with time irregularity

        # Map elapse time in days or months
        #T = self.map_elapse_time(t)

        # Decompose the previous cell if there is an elapse time
        #C_ST = tf.nn.tanh(tf.matmul(prev_cell, self.W_decomp) + self.b_decomp)
        #C_ST_dis = tf.multiply(T, C_ST)
        # if T is 0, then the weight is one
        #prev_cell = prev_cell - C_ST + C_ST_dis

        # Input gate
        i = tf.sigmoid(tf.matmul(x, self.Wid) + tf.matmul(prev_hidden_state, self.Uid) + self.bid)

        # Forget Gate
        f = tf.sigmoid(tf.matmul(x, self.Wfd) + tf.matmul(prev_hidden_state, self.Ufd) + self.bfd)

        # Output Gate
        o = tf.sigmoid(tf.matmul(x, self.Wogd) + tf.matmul(prev_hidden_state, self.Uogd) + self.bogd)

        # Candidate Memory Cell
        C = tf.nn.tanh(tf.matmul(x, self.Wcd) + tf.matmul(prev_hidden_state, self.Ucd) + self.bcd)

        # Current Memory cell
        Ct = f * prev_cell + i * C

        # Current Hidden state
        current_hidden_state = o * tf.nn.tanh(Ct)

        return tf.stack([current_hidden_state, Ct])

    def get_states_gy(self):  # Returns all hidden states for the samples in a batch
        batch_size = tf.shape(self.input)[0]
        scan_input_ = tf.transpose(self.input, perm=[2, 0, 1])
        scan_input = tf.transpose(scan_input_)  # scan input is [seq_length x batch_size x input_dim]
        initial_hidden = tf.zeros([batch_size, self.hidden_dim], tf.float32)
        ini_state_cell = tf.stack([initial_hidden, initial_hidden])
        packed_hidden_states = tf.scan(self.LSTM_Unit, scan_input, initializer=ini_state_cell, name='states_1')
        all_states = packed_hidden_states[:, 0, :, :]
        #all_cells = packed_hidden_states[:, 1, :, :]
        source_features = tf.slice(all_states, [0, 0, 0], [-1, batch_size // 2, -1])
        classify_feats = source_features if self.dann==1 else all_states
        return all_states, classify_feats


    def get_states_gd(self):  # Returns all hidden states for the samples in a batch
        batch_size = tf.shape(self.input)[0]
        Gd_input, _ = self.get_states_gy()
        #Gd_input = tf.map_fn(flip_gradient, [Gd_input, self.l_coef])
        #_, alphas = self.get_attention_coefs(inputs=Gd_input, attention_size=5)
        scan_input = Gd_input
        #scan_time = tf.transpose(self.time)  # scan_time [seq_length x batch_size]
        initial_hidden = tf.zeros([batch_size, self.hidden_dim], tf.float32)
        ini_state_cell = tf.stack([initial_hidden, initial_hidden])
        # make scan_time [seq_length x batch_size x 1]
        #scan_time = tf.reshape(scan_time, [tf.shape(scan_time)[0], tf.shape(scan_time)[1], 1])
        #concat_input = tf.concat([scan_time, scan_input], 2)  # [seq_length x batch_size x input_dim+1]
        packed_hidden_states = tf.scan(self.T_ALSTM_Unit, scan_input, initializer=ini_state_cell, name='states_2')
        all_states = packed_hidden_states[:, 0, :, :]
        return all_states


    def get_output_gy(self, state):
        output = tf.sigmoid(tf.matmul(state, self.Woy) + self.boy)
        return output

    def get_output_gd(self, state):
        output = tf.sigmoid(tf.matmul(state, self.Wod) + self.bod)
        return output

    def get_outputs_gy(self):
        _, classify_feats = self.get_states_gy()
        all_outputs = tf.map_fn(self.get_output_gy, classify_feats)
        #all_outputs_cell = tf.map_fn(self.get_output, all_cells)
        outputs = tf.reverse(all_outputs, [0])[0, :, :]
        #outputs_cell = tf.reverse(all_outputs_cell, [0])[0, :, :]
        return outputs#, all_outputs_cell

    def get_outputs_gd(self):
        all_states = self.get_states_gd()
        all_outputs = tf.map_fn(self.get_output_gd, all_states)
        #all_outputs_cell = tf.map_fn(self.get_output, all_cells)
        outputs = tf.reverse(all_outputs, [0])[0, :, :]
        #outputs_cell = tf.reverse(all_outputs_cell, [0])[0, :, :]
        return outputs#, all_outputs_cell

    def calculate_error(self, target, output):
        return tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=target)

    def get_loss_gy(self):
        self.label_pred = self.get_outputs_gy()
        batch_size = tf.shape(self.input)[0]
        source_labels = tf.slice(self.target, [0, 0], [batch_size // 2, -1])
        all_labels = self.target
        self.classify_labels = source_labels if self.dann==1 else all_labels
        errors = self.calculate_error(self.classify_labels, self.label_pred)
        return tf.reduce_mean(errors)

    def get_loss_gd(self):
        self.domain_pred = self.get_outputs_gd()
        errors = self.calculate_error(self.target_domain, self.domain_pred)
        return tf.reduce_mean(errors)

