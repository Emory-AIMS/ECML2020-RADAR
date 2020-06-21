import tensorflow as tf
from embedder.embedder import EMBEDDER

class VAE:
    def __init__(self,
                BATCH_SIZE = 64,
                MAX_TIMESTEP = 48,
                FEATURE_DIM = 19,
                ENCODER_NUM_UNITS = 64,
                ENCODER_NUM_UNITS_l1 = 512,
                ENCODER_NUM_UNITS_l2 = 256,
                ENCODER_FC_UNITS_l1 = 128,
                ENCODER_FC_UNITS_l2 = 64,
                ENCODER_FC_UNITS_l3 = 32,
                ATTENUNITS = 32,
                LEARNING_RATE = 0.005,
                REG_SCALE = 0.01):
        self.BATCH_SIZE = BATCH_SIZE
        self.MAX_TIMESTEP = MAX_TIMESTEP
        self.FEATURE_DIM = FEATURE_DIM
        self.ENCODER_NUM_UNITS_l1 = ENCODER_NUM_UNITS_l1
        self.ENCODER_NUM_UNITS_l2 = ENCODER_NUM_UNITS_l2
        self.ENCODER_FC_UNITS_l1 = ENCODER_FC_UNITS_l1
        self.ENCODER_FC_UNITS_l2 = ENCODER_FC_UNITS_l2
        self.ENCODER_FC_UNITS_l3 = ENCODER_FC_UNITS_l3
        self.ATTENUNITS = ATTENUNITS
        self.LEARNING_RATE = LEARNING_RATE
        self.REG_SCALE = REG_SCALE

        self.initializer = tf.random_normal_initializer(stddev=0.1)


    def encoder(self, x_pl):
        with tf.variable_scope('encoder', initializer = self.initializer):

            with tf.variable_scope('rnn', initializer = self.initializer):
                cell = tf.nn.rnn_cell.LSTMCell(self.ENCODER_NUM_UNITS)
                output, out_state = tf.nn.dynamic_rnn(cell = cell, inputs = x_pl, \
                                                    dtype=tf.float32)
                    
            with tf.variable_scope('rnn_reverse', initializer = self.initializer):
                cell = tf.nn.rnn_cell.LSTMCell(self.ENCODER_NUM_UNITS)
                x_pl_rev = tf.reverse(x_pl,[1])
                output, out_state_rev = tf.nn.dynamic_rnn(cell = cell, inputs = x_pl_rev, \
                                                    dtype=tf.float32)

                    
            with tf.variable_scope('fc', initializer = self.initializer):
                
                c_state = tf.concat([out_state[0],out_state_rev[0]],1)
                hidden_state = tf.concat([out_state[1],out_state_rev[1]],1)
                print (hidden_state, c_state)
                hz_1 = tf.layers.dense(hidden_state, units=self.ENCODER_FC_UNITS_l1, activation=tf.nn.relu)
                cz_1 = tf.layers.dense(c_state, units=self.ENCODER_FC_UNITS_l1, activation=tf.nn.relu)
                hz_2 = tf.layers.dense(hz_1, units=self.ENCODER_FC_UNITS_l2, activation=tf.nn.relu)
                cz_2 = tf.layers.dense(cz_1, units=self.ENCODER_FC_UNITS_l2, activation=tf.nn.relu)
                hz = tf.layers.dense(hz_2, units=self.ENCODER_FC_UNITS_l3, activation=tf.nn.relu)
                cz = tf.layers.dense(cz_2, units=self.ENCODER_FC_UNITS_l3, activation=tf.nn.relu)
        return tf.nn.rnn_cell.LSTMStateTuple(cz,hz)

    def decoder(self, z, decoder_input):
        with tf.variable_scope('decoder', initializer = self.initializer):

            with tf.variable_scope('rnn', initializer = self.initializer):
                cell = tf.nn.rnn_cell.LSTMCell(self.ENCODER_FC_UNITS_l3)
                output, out_state = tf.nn.dynamic_rnn(cell = cell, inputs = decoder_input, \
                                                    initial_state = z, dtype=tf.float32)###must be float

            with tf.variable_scope('fc', initializer = self.initializer):
                W_out = tf.get_variable('W_out', [self.ENCODER_FC_UNITS_l3, self.FEATURE_DIM])
                b_out = tf.get_variable('b_out', [self.FEATURE_DIM])
                output = tf.transpose(output, [1,0,2])
                #out = tf.matmul(output,W_out)+b_out
                out = tf.tensordot(output, W_out, axes=[[2],[0]])+b_out
                out = tf.transpose(out,[1,0,2])
        return out


    def encoder_stacked(self, x_pl):
        with tf.variable_scope('encoder_stack', initializer = self.initializer):

            with tf.variable_scope('rnn_stack', initializer = self.initializer):
                num_units = [self.ENCODER_NUM_UNITS_l1, self.ENCODER_NUM_UNITS_l2]
                cells = [tf.nn.rnn_cell.LSTMCell(num_units=n) for n in num_units]
                stacked_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(cells)
                #cell = tf.nn.rnn_cell.LSTMCell(ENCODER_NUM_UNITS)
                datum = tf.split(x_pl, self.MAX_TIMESTEP, axis = 1)
                output, out_state = tf.nn.dynamic_rnn(cell = stacked_rnn_cell, inputs = x_pl,\
                                                        dtype=tf.float32)###must be float
                    
            with tf.variable_scope('rnn_reverse_stack', initializer = self.initializer):
                num_units = [self.ENCODER_NUM_UNITS_l1, self.ENCODER_NUM_UNITS_l2]
                cells = [tf.nn.rnn_cell.LSTMCell(num_units=n) for n in num_units]    
                stacked_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(cells)
                x_pl_rev = tf.reverse(x_pl,[1])
                output, out_state_rev = tf.nn.dynamic_rnn(cell = stacked_rnn_cell, inputs = x_pl_rev, \
                                                        dtype=tf.float32)###must be float
                    
            with tf.variable_scope('fc_stack', initializer = self.initializer):
                
                c_state_s1 = tf.concat([out_state[0][0], out_state_rev[0][0]],1)
                c_state_s2 = tf.concat([out_state[1][0], out_state_rev[1][0]],1)
                hidden_state_s1 = tf.concat([out_state[0][1],out_state_rev[0][1]],1)
                hidden_state_s2 = tf.concat([out_state[1][1],out_state_rev[1][1]],1)
                print (c_state_s1, c_state_s2, hidden_state_s1, hidden_state_s2)
                
                hz_1 = tf.layers.dense(hidden_state_s1, units=self.ENCODER_FC_UNITS_l1, activation=tf.nn.relu)
                cz_1 = tf.layers.dense(c_state_s1, units=self.ENCODER_FC_UNITS_l1, activation=tf.nn.relu)
                hz_2 = tf.layers.dense(hidden_state_s2, units=self.ENCODER_FC_UNITS_l2, activation=tf.nn.relu)
                cz_2 = tf.layers.dense(c_state_s2, units=self.ENCODER_FC_UNITS_l2, activation=tf.nn.relu)

        return tf.nn.rnn_cell.LSTMStateTuple(cz_1,hz_1), tf.nn.rnn_cell.LSTMStateTuple(cz_2,hz_2)

    def decoder_stacked(self, z1,z2, batch_pl):
        decoder_input = tf.zeros([batch_pl, self.MAX_TIMESTEP, self.FEATURE_DIM])
        with tf.variable_scope('decoder_stacked', initializer = self.initializer):

            with tf.variable_scope('rnn_stacked', initializer = self.initializer):
                num_units = [self.ENCODER_FC_UNITS_l1, self.ENCODER_FC_UNITS_l2]
                cells = [tf.nn.rnn_cell.LSTMCell(num_units=n) for n in num_units] 
                stacked_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(cells, state_is_tuple = True)

                output, out_state = tf.nn.dynamic_rnn(cell = stacked_rnn_cell, inputs = decoder_input \
                                                    ,initial_state = (z1,z2), dtype=tf.float32)
            with tf.variable_scope('fc_stacked', initializer = self.initializer):
                W_out = tf.get_variable('W_out_stacked', [self.ENCODER_FC_UNITS_l2, self.FEATURE_DIM])
                b_out = tf.get_variable('b_out_stacked', [self.FEATURE_DIM])
                output = tf.transpose(output, [1,0,2])
                #out = tf.matmul(output,W_out)+b_out
                out = tf.tensordot(output, W_out, axes=[[2],[0]])+b_out
                out = tf.transpose(out,[1,0,2])
        return out

    def encoder_stacked_attention(self, x_pl, batch_pl, random = False):
        with tf.variable_scope('encoder_stack', initializer = self.initializer):

            with tf.variable_scope('rnn_stack', initializer = self.initializer):
                num_units = [self.ENCODER_NUM_UNITS_l1, self.ENCODER_NUM_UNITS_l2]
                cells = [tf.nn.rnn_cell.LSTMCell(num_units=n) for n in num_units]
                stacked_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(cells)
                #cell = tf.nn.rnn_cell.LSTMCell(ENCODER_NUM_UNITS)
                datum = tf.split(x_pl, self.MAX_TIMESTEP, axis = 1)
            
                state = stacked_rnn_cell.zero_state(batch_pl, tf.float32)
                out_state_l1_c = []
                out_state_l1_h = []
                out_state_l2_c = []
                out_state_l2_h = []
                for i in range(self.MAX_TIMESTEP):
                    inp = tf.reshape(datum[i], [-1, self.FEATURE_DIM])
                    out, state = stacked_rnn_cell(inp, state)
                    state_l1_c = state[0][0]
                    state_l1_h = state[0][1]
                    state_l2_c = state[1][0]
                    state_l2_h = state[1][1]
                    
                    out_state_l1_c.append(state_l1_c)
                    out_state_l2_c.append(state_l2_c)
                    out_state_l1_h.append(state_l1_h)
                    out_state_l2_h.append(state_l2_h)
                    
                out_state_l1_c = tf.stack(out_state_l1_c, axis = 1)
                out_state_l2_c = tf.stack(out_state_l2_c, axis = 1)
                out_state_l1_h = tf.stack(out_state_l1_h, axis = 1)
                out_state_l2_h = tf.stack(out_state_l2_h, axis = 1) 

            with tf.variable_scope('rnn_reverse_stack', initializer = self.initializer):
                num_units = [self.ENCODER_NUM_UNITS_l1, self.ENCODER_NUM_UNITS_l2]
                cells = [tf.nn.rnn_cell.LSTMCell(num_units=n) for n in num_units]    
                stacked_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(cells)
                x_pl_rev = tf.reverse(x_pl,[1])
                state = stacked_rnn_cell.zero_state(batch_pl, tf.float32)
                datum = tf.split(x_pl_rev, self.MAX_TIMESTEP, axis = 1)   
                out_state_l1_c_rev = []
                out_state_l1_h_rev = []
                out_state_l2_c_rev = []
                out_state_l2_h_rev = []
                for i in range(self.MAX_TIMESTEP):
                    inp = tf.reshape(datum[i], [-1, self.FEATURE_DIM])
                    out, state = stacked_rnn_cell(inp, state)
                    state_l1_c = state[0][0]
                    state_l1_h = state[0][1]
                    state_l2_c = state[1][0]
                    state_l2_h = state[1][1]
                    
                    out_state_l1_c_rev.append(state_l1_c)
                    out_state_l2_c_rev.append(state_l2_c)
                    out_state_l1_h_rev.append(state_l1_h)
                    out_state_l2_h_rev.append(state_l2_h)
                    
                out_state_l1_c_rev = tf.stack(out_state_l1_c_rev, axis = 1)
                out_state_l2_c_rev = tf.stack(out_state_l2_c_rev, axis = 1)
                out_state_l1_h_rev = tf.stack(out_state_l1_h_rev, axis = 1)
                out_state_l2_h_rev = tf.stack(out_state_l2_h_rev, axis = 1)
                    
            with tf.variable_scope('fc_stack', initializer = self.initializer):
                
                out_state_l1_c_rev = tf.reverse(out_state_l1_c_rev, [1])
                out_state_l2_c_rev = tf.reverse(out_state_l2_c_rev, [1])
                out_state_l1_h_rev = tf.reverse(out_state_l1_h_rev, [1])
                out_state_l2_h_rev = tf.reverse(out_state_l2_h_rev, [1])
                
                c_state_s1 = tf.concat([out_state_l1_c, out_state_l1_c_rev],2)
                c_state_s2 = tf.concat([out_state_l2_c, out_state_l2_c_rev],2)
                hidden_state_s1 = tf.concat([out_state_l1_h,out_state_l1_h_rev],2)
                hidden_state_s2 = tf.concat([out_state_l2_h,out_state_l2_h_rev],2)
                
                hz_1 = tf.layers.dense(hidden_state_s1, units=self.ENCODER_FC_UNITS_l1, activation=tf.nn.relu)
                cz_1 = tf.layers.dense(c_state_s1, units=self.ENCODER_FC_UNITS_l1, activation=tf.nn.relu)
                hz_2 = tf.layers.dense(hidden_state_s2, units=self.ENCODER_FC_UNITS_l2, activation=tf.nn.relu)
                cz_2 = tf.layers.dense(c_state_s2, units=self.ENCODER_FC_UNITS_l2, activation=tf.nn.relu)

                hz_1_last_state = tf.split(hz_1, self.MAX_TIMESTEP, axis = 1)[-1]
                hz_1_last_state = tf.reshape(hz_1_last_state, [batch_pl, self.ENCODER_FC_UNITS_l1])
                hz_2_last_state = tf.split(hz_2, self.MAX_TIMESTEP, axis = 1)[-1]
                hz_2_last_state = tf.reshape(hz_2_last_state, [batch_pl, self.ENCODER_FC_UNITS_l2])
                cz_1_last_state = tf.split(cz_1,self.MAX_TIMESTEP, axis = 1)[-1]
                cz_1_last_state = tf.reshape(cz_1_last_state, [batch_pl, self.ENCODER_FC_UNITS_l1])
                cz_2_last_state = tf.split(cz_2, self.MAX_TIMESTEP, axis = 1)[-1] 
                cz_2_last_state = tf.reshape(cz_2_last_state, [batch_pl, self.ENCODER_FC_UNITS_l2])

                if random == True:
                    cz_1_last_state = cz_1_last_state +  tf.random_normal([batch_pl, self.ENCODER_FC_UNITS_l1],0, 0.3)
                    hz_1_last_state = hz_1_last_state +  tf.random_normal([batch_pl, self.ENCODER_FC_UNITS_l1],0, 0.3)
                    cz_2_last_state = cz_2_last_state +  tf.random_normal([batch_pl, self.ENCODER_FC_UNITS_l2],0, 0.3)
                    hz_2_last_state = hz_2_last_state +  tf.random_normal([batch_pl, self.ENCODER_FC_UNITS_l2],0, 0.3)
            print (hz_2)

        return tf.nn.rnn_cell.LSTMStateTuple(cz_1_last_state,hz_1_last_state), \
               tf.nn.rnn_cell.LSTMStateTuple(cz_2_last_state,hz_2_last_state), hz_2

    def decoder_stacked_attention(self, memory, z1,z2, batch_pl):
        inp = tf.zeros([batch_pl, self.MAX_TIMESTEP, self.FEATURE_DIM])

        with tf.variable_scope('decoder_stacked', initializer = self.initializer):

            with tf.variable_scope('rnn_stacked', initializer = self.initializer):
                num_units = [self.ENCODER_FC_UNITS_l1, self.ENCODER_FC_UNITS_l2]
                cells = [tf.nn.rnn_cell.LSTMCell(num_units=n) for n in num_units] 
                stacked_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(cells, state_is_tuple = True)

                AttentionMech = tf.contrib.seq2seq.BahdanauAttention(self.ATTENUNITS, memory)
                att_wrapper = tf.contrib.seq2seq.AttentionWrapper(stacked_rnn_cell, AttentionMech)
                state = att_wrapper.zero_state(batch_pl, tf.float32)
                print (z1, z2)
                state = state.clone(cell_state=(z1, z2))
                print (state)
                datum = tf.split(inp, self.MAX_TIMESTEP, axis = 1)
                out_list = []
                for i in range(self.MAX_TIMESTEP):
                    inp = tf.reshape(datum[i], [batch_pl, self.FEATURE_DIM])
                    
                    out, state = att_wrapper(inp, state)
                    out_list.append(out)
                out_tensor = tf.stack(out_list, axis = 1)  
                print (out_tensor)
            with tf.variable_scope('fc_stacked', initializer = self.initializer):
                W_out = tf.get_variable('W_out_stacked', [self.ENCODER_FC_UNITS_l2, self.FEATURE_DIM])
                b_out = tf.get_variable('b_out_stacked', [self.FEATURE_DIM])
                output = tf.transpose(out_tensor, [1,0,2])
                #out = tf.matmul(output,W_out)+b_out
                out = tf.tensordot(output, W_out, axes=[[2],[0]])+b_out
                out = tf.transpose(out,[1,0,2])
        return out


    def encoder_stacked_attention_emb(self, x_pl, batch_pl):
        with tf.variable_scope('encoder_stack', initializer = self.initializer):

            with tf.variable_scope('rnn_stack', initializer = self.initializer):
                num_units = [self.ENCODER_NUM_UNITS_l1, self.ENCODER_NUM_UNITS_l2]
                cells = [tf.nn.rnn_cell.LSTMCell(num_units=n) for n in num_units]
                stacked_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(cells)
                #cell = tf.nn.rnn_cell.LSTMCell(ENCODER_NUM_UNITS)
                datum = tf.split(x_pl, self.MAX_TIMESTEP, axis = 1)
            
                state = stacked_rnn_cell.zero_state(batch_pl, tf.float32)
                out_state_l1_c = []
                out_state_l1_h = []
                out_state_l2_c = []
                out_state_l2_h = []
                for i in range(self.MAX_TIMESTEP):
                    inp = tf.reshape(datum[i], [-1, self.FEATURE_DIM])
                    out, state = stacked_rnn_cell(inp, state)
                    state_l1_c = state[0][0]
                    state_l1_h = state[0][1]
                    state_l2_c = state[1][0]
                    state_l2_h = state[1][1]
                    
                    out_state_l1_c.append(state_l1_c)
                    out_state_l2_c.append(state_l2_c)
                    out_state_l1_h.append(state_l1_h)
                    out_state_l2_h.append(state_l2_h)
                    
                out_state_l1_c = tf.stack(out_state_l1_c, axis = 1)
                out_state_l2_c = tf.stack(out_state_l2_c, axis = 1)
                out_state_l1_h = tf.stack(out_state_l1_h, axis = 1)
                out_state_l2_h = tf.stack(out_state_l2_h, axis = 1) 

            with tf.variable_scope('rnn_reverse_stack', initializer = self.initializer):
                num_units = [self.ENCODER_NUM_UNITS_l1, self.ENCODER_NUM_UNITS_l2]
                cells = [tf.nn.rnn_cell.LSTMCell(num_units=n) for n in num_units]    
                stacked_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(cells)
                x_pl_rev = tf.reverse(x_pl,[1])
                state = stacked_rnn_cell.zero_state(batch_pl, tf.float32)
                datum = tf.split(x_pl_rev, self.MAX_TIMESTEP, axis = 1)   
                out_state_l1_c_rev = []
                out_state_l1_h_rev = []
                out_state_l2_c_rev = []
                out_state_l2_h_rev = []
                for i in range(self.MAX_TIMESTEP):
                    inp = tf.reshape(datum[i], [-1, self.FEATURE_DIM])
                    out, state = stacked_rnn_cell(inp, state)
                    state_l1_c = state[0][0]
                    state_l1_h = state[0][1]
                    state_l2_c = state[1][0]
                    state_l2_h = state[1][1]
                    
                    out_state_l1_c_rev.append(state_l1_c)
                    out_state_l2_c_rev.append(state_l2_c)
                    out_state_l1_h_rev.append(state_l1_h)
                    out_state_l2_h_rev.append(state_l2_h)
                    
                out_state_l1_c_rev = tf.stack(out_state_l1_c_rev, axis = 1)
                out_state_l2_c_rev = tf.stack(out_state_l2_c_rev, axis = 1)
                out_state_l1_h_rev = tf.stack(out_state_l1_h_rev, axis = 1)
                out_state_l2_h_rev = tf.stack(out_state_l2_h_rev, axis = 1)
                    
            with tf.variable_scope('fc_stack', initializer = self.initializer):
                
                out_state_l1_c_rev = tf.reverse(out_state_l1_c_rev, [1])
                out_state_l2_c_rev = tf.reverse(out_state_l2_c_rev, [1])
                out_state_l1_h_rev = tf.reverse(out_state_l1_h_rev, [1])
                out_state_l2_h_rev = tf.reverse(out_state_l2_h_rev, [1])
                
                c_state_s1 = tf.concat([out_state_l1_c, out_state_l1_c_rev],2)
                c_state_s2 = tf.concat([out_state_l2_c, out_state_l2_c_rev],2)
                hidden_state_s1 = tf.concat([out_state_l1_h,out_state_l1_h_rev],2)
                hidden_state_s2 = tf.concat([out_state_l2_h,out_state_l2_h_rev],2)
                
                hz_1 = tf.layers.dense(hidden_state_s1, units=self.ENCODER_FC_UNITS_l1, activation=tf.nn.relu)
                cz_1 = tf.layers.dense(c_state_s1, units=self.ENCODER_FC_UNITS_l1, activation=tf.nn.relu)
                hz_2 = tf.layers.dense(hidden_state_s2, units=self.ENCODER_FC_UNITS_l2, activation=tf.nn.relu)
                cz_2 = tf.layers.dense(c_state_s2, units=self.ENCODER_FC_UNITS_l2, activation=tf.nn.relu)

            ##embedding
            with tf.variable_scope("hz_1_emb"):
                emb_hz_1 = EMBEDDER(
                        emb_shape = [self.MAX_TIMESTEP,self.ENCODER_FC_UNITS_l1],
                        num_category = [512],
                        emb_norm = "LAYER"
                        )
                hz_1_emb,_,_ = emb_hz_1.evaluate(hz_1, True)
            with tf.variable_scope("cz_1_emb"):
                emb_cz_1 = EMBEDDER(
                        emb_shape = [self.MAX_TIMESTEP,self.ENCODER_FC_UNITS_l1],
                        num_category = [512],
                        emb_norm = "LAYER"
                        )
                cz_1_emb,_,_ = emb_cz_1.evaluate(cz_1, True)
            with tf.variable_scope("hz_2_emb"):
                emb_hz_2 = EMBEDDER(
                        emb_shape = [self.MAX_TIMESTEP,self.ENCODER_FC_UNITS_l2],
                        num_category = [512],
                        emb_norm = "LAYER"
                        )
                hz_2_emb,_,_ = emb_hz_2.evaluate(hz_2, True)
            with tf.variable_scope("cz_2_emb"):
                emb_cz_2 = EMBEDDER(
                        emb_shape = [self.MAX_TIMESTEP,self.ENCODER_FC_UNITS_l2],
                        num_category = [512],
                        emb_norm = "LAYER"
                        )
                cz_2_emb,_,_ = emb_cz_2.evaluate(cz_2, True) 
            #import pdb; pdb.set_trace()
            hz_1_last_state = tf.split(hz_1_emb, self.MAX_TIMESTEP, axis = 1)[-1]
            hz_1_last_state = tf.reshape(hz_1_last_state, [batch_pl, self.ENCODER_FC_UNITS_l1])
            hz_2_last_state = tf.split(hz_2_emb, self.MAX_TIMESTEP, axis = 1)[-1]
            hz_2_last_state = tf.reshape(hz_2_last_state, [batch_pl, self.ENCODER_FC_UNITS_l2])
            cz_1_last_state = tf.split(cz_1_emb,self.MAX_TIMESTEP, axis = 1)[-1]
            cz_1_last_state = tf.reshape(cz_1_last_state, [batch_pl, self.ENCODER_FC_UNITS_l1])
            cz_2_last_state = tf.split(cz_2_emb, self.MAX_TIMESTEP, axis = 1)[-1] 
            cz_2_last_state = tf.reshape(cz_2_last_state, [batch_pl, self.ENCODER_FC_UNITS_l2])
            print (hz_2)

        return tf.nn.rnn_cell.LSTMStateTuple(cz_1_last_state,hz_1_last_state), \
               tf.nn.rnn_cell.LSTMStateTuple(cz_2_last_state,hz_2_last_state), hz_2

    def decoder_stacked_attention_emb(self, memory, z1,z2, batch_pl):
        inp = tf.zeros([batch_pl, self.MAX_TIMESTEP, self.FEATURE_DIM])

        with tf.variable_scope('decoder_stacked', initializer = self.initializer):

            with tf.variable_scope('rnn_stacked', initializer = self.initializer):
                num_units = [self.ENCODER_FC_UNITS_l1, self.ENCODER_FC_UNITS_l2]
                cells = [tf.nn.rnn_cell.LSTMCell(num_units=n) for n in num_units] 
                stacked_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(cells, state_is_tuple = True)

                AttentionMech = tf.contrib.seq2seq.BahdanauAttention(self.ATTENUNITS, memory)
                att_wrapper = tf.contrib.seq2seq.AttentionWrapper(stacked_rnn_cell, AttentionMech)
                state = att_wrapper.zero_state(batch_pl, tf.float32)
                print (z1, z2)
                state = state.clone(cell_state=(z1, z2))
                print (state)
                datum = tf.split(inp, self.MAX_TIMESTEP, axis = 1)
                out_list = []
                for i in range(self.MAX_TIMESTEP):
                    inp = tf.reshape(datum[i], [batch_pl, self.FEATURE_DIM])
                    
                    out, state = att_wrapper(inp, state)
                    out_list.append(out)
                out_tensor = tf.stack(out_list, axis = 1)  
                print (out_tensor)
            with tf.variable_scope('fc_stacked', initializer = self.initializer):
                W_out = tf.get_variable('W_out_stacked', [self.ENCODER_FC_UNITS_l2, self.FEATURE_DIM])
                b_out = tf.get_variable('b_out_stacked', [self.FEATURE_DIM])
                output = tf.transpose(out_tensor, [1,0,2])
                #out = tf.matmul(output,W_out)+b_out
                out = tf.tensordot(output, W_out, axes=[[2],[0]])+b_out
                out = tf.transpose(out,[1,0,2])
                out = tf.nn.leaky_relu(out)
        return out

    def reg_loss(self):

        opt_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        regularize = tf.contrib.layers.l2_regularizer(self.REG_SCALE)
        print (tf.GraphKeys.TRAINABLE_VARIABLES)
        reg_term = sum([regularize(param) for param in opt_vars])

        return reg_term 
    
    def l2_loss(self, x, x_gen):
        x_gen = tf.reverse(x_gen,[1])
        diff = x_gen-x
        norm = tf.sqrt(tf.reduce_sum(tf.square(diff),axis = [1,2]))
        norm = tf.reduce_mean(norm, axis = 0) 
        return norm

    def l1_loss(self, x, x_gen):
        x_gen = tf.reverse(x_gen,[1])
        diff = x_gen-x
        norm = tf.reduce_sum(tf.abs(diff),axis = [1,2])
        norm = tf.reduce_mean(norm, axis = 0) 
        return norm