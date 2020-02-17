import tensorflow as tf
import sys
from base import TransformerNet
from base import multihead_attention_original
import numpy as np


class OAR(object):
    def __init__(self, args, n_items, n_users):
        self.args = args
        self.n_items = n_items
        self.n_users = n_users
        self.n_time = 366 

        
        self.triple_bpr = tf.placeholder(dtype=tf.int32, shape=[None, 4])

        self.time_entity = np.array([i for i in range(1, self.n_time+1)])
        
        self.lr = tf.placeholder(tf.float32, shape=None, name='lr')

        self._build()

        self.saver = tf.train.Saver()


    def _build(self):
        self.u_list = tf.placeholder(tf.int32, shape=(None), name='u_list')
        self.inp = tf.placeholder(tf.int32, shape=(None, None), name='inp')
        self.inp_time = tf.placeholder(tf.int32, shape=(None, None), name='inp_time')
        self.pos = tf.placeholder(tf.int32, shape=(None, None), name='pos')
        self.pos_time = tf.placeholder(tf.int32, shape=(None, None), name='pos_time')
        self.neg = tf.placeholder(tf.int32, shape=(None, None, 1), name='neg')


        
        self.dropout = tf.placeholder_with_default(0., shape=())
        self.item_embedding = tf.get_variable('item_embedding', \
                                shape=(self.n_items + 1, self.args.emsize), \
                                dtype=tf.float32, \
                                regularizer=tf.contrib.layers.l2_regularizer(self.args.l2_reg), \
                                initializer=tf.contrib.layers.xavier_initializer())

        self.item_embedding_p = tf.get_variable('item_embedding_p', \
                                shape=(self.n_items + 1, self.args.emsize), \
                                dtype=tf.float32, \
                                regularizer=tf.contrib.layers.l2_regularizer(self.args.l2_reg), \
                                initializer=tf.contrib.layers.xavier_initializer())


        self.user_embedding =  tf.get_variable('user_embedding', \
                                shape=(self.n_users + 1, self.args.emsize), \
                                dtype=tf.float32, \
                                regularizer=tf.contrib.layers.l2_regularizer(self.args.l2_reg), \
                                initializer=tf.contrib.layers.xavier_initializer())

        self.time_embedding = tf.get_variable('time_embedding', \
                                shape=(self.n_time + 1, self.args.emsize), \
                                dtype=tf.float32, \
                                regularizer=tf.contrib.layers.l2_regularizer(self.args.l2_reg), \
                                initializer=tf.contrib.layers.xavier_initializer())

        self.global_m_table = tf.get_variable('global_m_table', \
                                shape=(self.n_time + 1, self.args.emsize), \
                                dtype=tf.float32, \
                                regularizer=tf.contrib.layers.l2_regularizer(self.args.l2_reg), \
                                initializer=tf.contrib.layers.xavier_initializer())

        ###zero_padding for item_embedding
        self.item_embedding = tf.concat((tf.zeros(shape=[1, self.args.emsize]),
                                  self.item_embedding[1:, :]), 0)

        self.item_embedding_p = tf.concat((tf.zeros(shape=[1, self.args.emsize]),
                                  self.item_embedding_p[1:, :]), 0)

        self.time_embedding = tf.concat((tf.zeros(shape=[1, self.args.emsize]),
                                  self.time_embedding[1:, :]), 0)

        self.global_m_table = tf.concat((tf.zeros(shape=[1, self.args.emsize]),
                                  self.global_m_table[1:, :]), 0)


        input_item = tf.nn.embedding_lookup(self.item_embedding, self.inp)
        input_item = input_item * (self.args.emsize ** 0.5)


        input_item_p = tf.nn.embedding_lookup(self.item_embedding_p, self.inp)
        input_item_p = input_item_p * (self.args.emsize ** 0.5)

        input_time = tf.nn.embedding_lookup(self.time_embedding, self.inp_time)
        positive_time = tf.nn.embedding_lookup(self.time_embedding, self.pos_time)

        
        copied_u_list = tf.tile(tf.expand_dims(self.u_list, [1]), [1, tf.shape(self.pos_time)[1]])
        user_vec = tf.nn.embedding_lookup(self.user_embedding, copied_u_list)  # pair with positive time, keys of attention
        
        # Personal Occasion Elicitation
        self.person_occasion, self.print_weights = multihead_attention_original(positive_time, input_time, input_item_p, num_units=self.args.emsize, \
            num_heads=self.args.num_heads, dropout_rate=self.dropout, causality=True, scope = 'multihead_attention_person_occasion', with_print_weights=True)

        # Global Occasion Memorization
        copied_time_embedding = tf.tile(tf.expand_dims(self.time_embedding, [0]), [tf.shape(self.inp)[0], 1, 1])
        copied_global_m_table = tf.tile(tf.expand_dims(self.global_m_table, [0]), [tf.shape(self.inp)[0], 1, 1])
        self.global_occasion = multihead_attention_original(positive_time, copied_time_embedding, copied_global_m_table, num_units=self.args.emsize, \
            num_heads=self.args.num_heads, dropout_rate=self.dropout, causality=False, scope = 'multihead_attention_global_occasion')

        mask = tf.expand_dims(tf.to_float(tf.not_equal(self.inp, 0)), -1)

        #intrinsic preference modeling
        self.net = TransformerNet(self.args.emsize, self.args.num_blocks, self.args.num_heads, self.args.seq_len, dropout_rate=self.dropout, pos_fixed=self.args.pos_fixed)        
        outputs = self.net(input_item, mask)
        outputs *= mask

        # query 
        user_time_query = tf.concat([user_vec, positive_time],-1) # for controlling the gate
        user_time_query_plain = tf.reshape(user_time_query, (-1, 2*self.args.emsize))        
        
        ############# for training #############
        ct_vec = tf.reshape(outputs, (-1, self.args.emsize)) #intrinsic
        person_occasion_vec = tf.reshape(self.person_occasion, (-1, self.args.emsize)) #person occasion
        global_occasion_vec = tf.reshape(self.global_occasion, (-1, self.args.emsize))  #global occasion
        
        # gated aggregation # 
        ############## for training #############            
        stacked_features = tf.stack([ct_vec,person_occasion_vec,global_occasion_vec])

        user_time_query_plain = tf.tile(tf.expand_dims(user_time_query_plain, [0]), [3, 1, 1])
        stacked_features_with_key = tf.concat([stacked_features, user_time_query_plain],-1)   #stacked_features#
        stacked_features_transformed = tf.layers.dense(stacked_features_with_key, self.args.emsize, activation=tf.nn.tanh, name='att1')
        stacked_features_score = tf.layers.dense(stacked_features_transformed, 1, name='att2')
        stacked_features_score = tf.nn.softmax(stacked_features_score, 0)
        stacked_features_score = tf.nn.dropout(stacked_features_score, keep_prob=1. - self.dropout)

        ct_vec_all = tf.reduce_sum(stacked_features_score*stacked_features, 0)
        ############## for test #############
        ct_vec_last = outputs[:,-1,:]
        ct_vec_last = tf.reshape(ct_vec_last, (-1, self.args.emsize))
        person_occasion_last = self.person_occasion[:,-1,:]
        person_occasion_last = tf.reshape(person_occasion_last, (-1, self.args.emsize))
        global_occasion_last = self.global_occasion[:,-1,:]
        global_occasion_last = tf.reshape(global_occasion_last, (-1, self.args.emsize))
        
        user_time_query_last = tf.concat([user_vec[:,-1,:], positive_time[:,-1,:]],-1)
        user_time_query_last = tf.reshape(user_time_query_last, (-1, 2*self.args.emsize))
               
        stacked_features_last = tf.stack([ct_vec_last,person_occasion_last,global_occasion_last])

        user_time_query_last = tf.tile(tf.expand_dims(user_time_query_last, [0]), [3, 1, 1])
        stacked_features_with_key_last = tf.concat([stacked_features_last, user_time_query_last],-1) #stacked_features_last#
        stacked_features_transformed_last = tf.layers.dense(stacked_features_with_key_last, self.args.emsize, activation=tf.nn.tanh, name='att1',reuse=True)
        stacked_features_score_last = tf.layers.dense(stacked_features_transformed_last, 1, name='att2',reuse=True)
        stacked_features_score_last = tf.nn.softmax(stacked_features_score_last, 0)
        stacked_features_score_last = tf.nn.dropout(stacked_features_score_last, keep_prob=1. - self.dropout)

        ct_vec_last_all = tf.reduce_sum(stacked_features_score_last*stacked_features_last, 0)
        #######################################

        self.total_loss_all = 0.

        self.istarget = istarget = tf.reshape(tf.to_float(tf.not_equal(self.pos, 0)), [-1])
    

        # calculate preference scores
        _pos_emb = tf.nn.embedding_lookup(self.item_embedding, self.pos)
        pos_emb = tf.reshape(_pos_emb, (-1, self.args.emsize))
        pos_emb_all = pos_emb
        _neg_emb = tf.nn.embedding_lookup(self.item_embedding, self.neg)
        neg_emb = tf.reshape(_neg_emb, (-1, 1, self.args.emsize))
        neg_emb_all = neg_emb

        temp_vec_neg_all = tf.tile(tf.expand_dims(ct_vec_all, [1]), [1, 1, 1]) #copy resulted vector based on the size of negative samples
        pos_logit_all = tf.reduce_sum(ct_vec_all * pos_emb_all, -1)
        neg_logit_all = tf.squeeze(tf.reduce_sum(temp_vec_neg_all * neg_emb_all, -1), 1)
        loss_all = tf.reduce_sum(
                    -tf.log(tf.sigmoid(pos_logit_all) + 1e-24) * istarget - \
                    tf.log(1 - tf.sigmoid(neg_logit_all) + 1e-24) * istarget \
                ) / tf.reduce_sum(istarget)


        self.test_item_batch = tf.placeholder(tf.int32, shape=(None, 101), name='test_item_batch')


        ###for 101 test cases
        ct_vec_batch_all = tf.tile(ct_vec_last_all, [101, 1])
        _test_item_emb_batch = tf.nn.embedding_lookup(self.item_embedding, self.test_item_batch)
        _test_item_emb_batch = tf.transpose(_test_item_emb_batch, perm=[1, 0, 2])
        test_item_emb_batch = tf.reshape(_test_item_emb_batch, (-1, self.args.emsize))
        test_item_emb_batch_all_i = test_item_emb_batch
        self.test_logits_batch_all = tf.reduce_sum(ct_vec_batch_all*test_item_emb_batch_all_i, -1)
        self.test_logits_batch_all = tf.transpose(tf.reshape(self.test_logits_batch_all, [101, tf.shape(self.inp)[0]]))
        ###

        
        ## loss 
        self.loss_all = loss_all
        self.total_loss_all += loss_all
        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        self.total_loss_all += sum(reg_losses)

        optimizer_all = tf.train.AdamOptimizer(self.lr)
        gvs_all = optimizer_all.compute_gradients(self.total_loss_all)
        capped_gvs_all = map(lambda gv: gv if gv[0] is None else [tf.clip_by_value(gv[0], -10., 10.), gv[1]], gvs_all)
        self.train_op_all = optimizer_all.apply_gradients(capped_gvs_all)
