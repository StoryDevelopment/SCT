
import itertools
import numpy as np

import tensorflow as tf
from tensorflow.contrib.rnn import BasicLSTMCell

from read_data import DataSet
from my.tensorflow import get_initializer
from my.tensorflow.nn import softsel, get_logits, highway_network, multi_conv1d,linear
from my.tensorflow.rnn import bidirectional_dynamic_rnn
from my.tensorflow.rnn_cell import SwitchableDropoutWrapper, AttentionCell



def get_multi_gpu_models(config):
    models = []
    for gpu_idx in range(config.num_gpus):
        with tf.name_scope("model_{}".format(gpu_idx)) as scope, tf.device("/{}:{}".format(config.device_type, gpu_idx)):
            if gpu_idx > 0:
                tf.get_variable_scope().reuse_variables()
            model = Model(config, scope, rep=gpu_idx == 0)
            models.append(model)
    return models


def dense_net_block(config, feature_map, growth_rate, layers, kernel_size, is_train, padding="SAME", act=tf.nn.relu,
                    scope=None):
    with tf.variable_scope(scope or "dense_net_block"):
        conv2d = tf.contrib.layers.convolution2d
        dim = feature_map.get_shape().as_list()[-1]

        list_of_features = [feature_map]
        features = feature_map
        for i in range(layers):
            ft = conv2d(features, growth_rate, (kernel_size, kernel_size), padding=padding, activation_fn=act)
            list_of_features.append(ft)
            features = tf.concat(list_of_features, axis=3)

        print("dense net block out shape")
        print(features.get_shape().as_list())
        return features


def dense_net_transition_layer(config, feature_map, transition_rate, scope=None):
    with tf.variable_scope(scope or "transition_layer"):
        out_dim = int(feature_map.get_shape().as_list()[-1] * transition_rate)
        feature_map = tf.contrib.layers.convolution2d(feature_map, out_dim, 1, padding="SAME", activation_fn=None)
        feature_map = tf.nn.max_pool(feature_map, [1, 2, 2, 1], [1, 2, 2, 1], "VALID")
        print("Transition Layer out shape")
        print(feature_map.get_shape().as_list())
        return feature_map

def dense_net(config, denseAttention, is_train):
    with tf.variable_scope("dense_net"):
        dim = denseAttention.get_shape().as_list()[-1]
        act = tf.nn.relu if True else None
        fm = tf.contrib.layers.convolution2d(denseAttention, int(dim * config.dense_net_first_scale_down_ratio),
                                             1, padding="SAME", activation_fn=act)

        fm = dense_net_block(config, fm, config.dense_net_growth_rate, config.dense_net_layers,
                             config.dense_net_kernel_size, is_train, scope="first_dense_net_block")
        fm = dense_net_transition_layer(config, fm, config.dense_net_transition_rate, scope='second_transition_layer')
        fm = dense_net_block(config, fm, config.dense_net_growth_rate, config.dense_net_layers,
                             config.dense_net_kernel_size, is_train, scope="second_dense_net_block")
        fm = dense_net_transition_layer(config, fm, config.dense_net_transition_rate, scope='third_transition_layer')
        fm = dense_net_block(config, fm, config.dense_net_growth_rate, config.dense_net_layers,
                             config.dense_net_kernel_size, is_train, scope="third_dense_net_block")

        fm = dense_net_transition_layer(config, fm, config.dense_net_transition_rate, scope='fourth_transition_layer')

        shape_list = fm.get_shape().as_list()
        print(shape_list)
        out_final = tf.reshape(fm, [config.batch_size, -1])
        return out_final

class Model(object):
    def __init__(self, config, scope, rep=True):
        self.scope = scope
        self.config = config
        self.global_step = tf.get_variable('global_step', shape=[], dtype='int32',
                                           initializer=tf.constant_initializer(0), trainable=False)

        # Define forward inputs here
        N, M, JX, JQ, VW, VC, W ,d= \
            config.batch_size, config.max_num_sents, config.max_sent_size, \
            config.max_ques_size, config.word_vocab_size, config.char_vocab_size, config.max_word_size, \
            config.hidden_size
        self.batch_mask = tf.placeholder('float', [N], name='batch_mask')
        self.x = tf.placeholder('int32', [N, None, None], name='x')
        self.cx = tf.placeholder('int32', [N, None, None, W], name='cx')
        self.x_sem = tf.placeholder('int32',[N,None,None],name='x_sem')
        self.x_pos= tf.placeholder('int32',[N,None,None],name='x_pos')
        self.x_neg = tf.placeholder('int32', [N, None, None], name='x_neg')
        self.x_mask = tf.placeholder('bool', [N, None, None], name='x_mask')

        self.q = tf.placeholder('int32', [N, None], name='q')
        self.cq = tf.placeholder('int32', [N, None, W], name='cq')
        self.q_mask = tf.placeholder('bool', [N, None], name='q_mask')
        self.q_sem = tf.placeholder('int32', [N, None], name='q_sem')
        self.q_pos = tf.placeholder('int32', [N, None], name='q_pos')
        self.q_neg = tf.placeholder('int32', [N, None], name='q_neg')
        self.is_train = tf.placeholder('bool', [], name='is_train')

        self.new_emb_mat = tf.placeholder('float', [None, config.word_emb_size], name='new_emb_mat')
        self.answers = tf.placeholder('float32', [N], name='answers')
        self.dense_input=tf.placeholder('float',[None,N, M * JQ * 2 * d])
        # Define misc
        self.tensor_dict = {}
        if self.config.mode=='train':
            self.num_candidate=config.train_num_can
        else:
            self.num_candidate = config.test_num_can
        # Forward outputs / loss inputs
        self.prediction = None
        self.var_list = None
        self.correct=None
        # Loss outputs
        self.loss = None
        self.acc=None
        self._build_forward()
        self._build_loss()
        self.var_ema = None
        if rep:
            self._build_var_ema()
        if config.mode == 'train':
            self._build_ema()

        self.summary = tf.summary.merge_all()
        self.summary = tf.summary.merge(tf.get_collection("summaries", scope=self.scope))

    def _build_forward(self):
        config = self.config
        N, M, JX, JQ, VW, VC, d, W = \
            config.batch_size,  config.max_num_sents, config.max_sent_size, \
            config.max_ques_size, config.word_vocab_size, config.char_vocab_size, config.hidden_size, \
            config.max_word_size

        print('VC:{}  NEW_EMB:{}'.format(VW,self.new_emb_mat.get_shape()))
        dc, dw, dco = config.char_emb_size, config.word_emb_size, config.char_out_size
        with tf.variable_scope("emb"):
            if config.use_char_emb:
                with tf.variable_scope("emb_var"), tf.device("/cpu:0"):
                    char_emb_mat = tf.get_variable("char_emb_mat", shape=[VC, dc], dtype='float')

                with tf.variable_scope("char"):
                    Acx = tf.nn.embedding_lookup(char_emb_mat, self.cx)  # [N, M, JX, W, dc]
                    Acq = tf.nn.embedding_lookup(char_emb_mat, self.cq)  # [N, JQ, W, dc]
                    Acx = tf.reshape(Acx, [-1, JX, W, dc])
                    Acq = tf.reshape(Acq, [-1, JQ, W, dc])


                    filter_sizes = list(map(int, config.out_channel_dims.split(',')))
                    heights = list(map(int, config.filter_heights.split(',')))
                    assert sum(filter_sizes) == dco, (filter_sizes, dco)
                    with tf.variable_scope("conv"):
                        xx = multi_conv1d(Acx, filter_sizes, heights, "VALID",  self.is_train, config.keep_prob, scope="xx")
                        if config.share_cnn_weights:
                            tf.get_variable_scope().reuse_variables()
                            qq = multi_conv1d(Acq, filter_sizes, heights, "VALID", self.is_train, config.keep_prob, scope="xx")

                        else:
                            qq = multi_conv1d(Acq, filter_sizes, heights, "VALID", self.is_train, config.keep_prob, scope="qq")
                        xx = tf.reshape(xx, [-1, M, JX, dco])
                        qq = tf.reshape(qq, [-1, JQ, dco])


            if config.use_word_emb:
                with tf.variable_scope("emb_var"), tf.device("/cpu:0"):
                    if config.mode == 'train':
                        word_emb_mat = tf.get_variable("word_emb_mat", dtype='float', shape=[VW, dw], initializer=get_initializer(config.emb_mat))
                    else:
                        word_emb_mat = tf.get_variable("word_emb_mat", shape=[VW, dw], dtype='float')
                    if config.use_glove_for_unk:
                        word_emb_mat = tf.concat(axis=0, values=[word_emb_mat, self.new_emb_mat])

                with tf.name_scope("word"):
                    Ax = tf.nn.embedding_lookup(word_emb_mat, self.x)  # [N, M, JX, d]
                    Aq = tf.nn.embedding_lookup(word_emb_mat, self.q)  # [N, JQ, d]

                    self.tensor_dict['x'] = Ax
                    self.tensor_dict['q'] = Aq
                if config.use_char_emb:
                    xx = tf.concat(axis=3, values=[xx, Ax])  # [N, M, JX, di]
                    qq = tf.concat(axis=2, values=[qq, Aq])  # [N, JQ, di]

                else:
                    xx = Ax
                    qq = Aq
            if config.use_pos_emb:
                with tf.variable_scope("pos_onehot"), tf.device("/cpu:0"):
                    pos_x = tf.one_hot(self.x_pos, depth=config.pos_tag_num)  # [N,M,JX,depth]
                    pos_q = tf.one_hot(self.q_pos, depth=config.pos_tag_num)  # [N,JQ,depth]
                    xx = tf.concat(axis=3, values=[xx, pos_x])  # [N, M, JX, di]
                    qq = tf.concat(axis=2, values=[qq, pos_q])
            if config.use_sem_emb:
                with tf.variable_scope("sem_onehot"), tf.device("/cpu:0"):
                    sem_x = tf.one_hot(self.x_sem, depth=3)  # [N,M,JX,3]
                    sem_q = tf.one_hot(self.q_sem, depth=3)  # [N,JQ,3]
                    xx = tf.concat(axis=3, values=[xx, sem_x])
                    qq = tf.concat(axis=2, values=[qq, sem_q])
            if config.use_neg_emb:
                with tf.variable_scope("neg_onehot"), tf.device("/cpu:0"):
                    neg_x = tf.one_hot(self.x_neg, depth=2)  # [N,M,JX,2]
                    neg_q = tf.one_hot(self.q_neg, depth=2)  # [N,JQ,2]

                    xx = tf.concat(axis=3, values=[xx, neg_x])
                    qq = tf.concat(axis=2, values=[qq, neg_q])


        # highway network
        if config.highway:
            with tf.variable_scope("highway"):
                xx = highway_network(xx, config.highway_num_layers, True, wd=config.wd, is_train=self.is_train)
                tf.get_variable_scope().reuse_variables()
                qq = highway_network(qq, config.highway_num_layers, True, wd=config.wd, is_train=self.is_train)
                tf.get_variable_scope().reuse_variables()


        self.tensor_dict['xx'] = xx
        self.tensor_dict['qq'] = qq


        cell_fw = BasicLSTMCell(d, state_is_tuple=True)
        cell_bw = BasicLSTMCell(d, state_is_tuple=True)
        d_cell_fw = SwitchableDropoutWrapper(cell_fw, self.is_train, input_keep_prob=config.input_keep_prob)
        d_cell_bw = SwitchableDropoutWrapper(cell_bw, self.is_train, input_keep_prob=config.input_keep_prob)

        x_len = tf.reduce_sum(tf.cast(self.x_mask, 'int32'), 2)  # [N, M]
        q_len = tf.reduce_sum(tf.cast(self.q_mask, 'int32'), 1)  # [N]

        with tf.variable_scope("prepro"):
            (fw_u, bw_u), ((_, fw_u_f), (_, bw_u_f)) = bidirectional_dynamic_rnn(d_cell_fw, d_cell_bw, qq, q_len, dtype='float', scope='u1')  # [N, J, d], [N, d]
            print('fw_u_f hsape:{}'.format(fw_u_f.get_shape()))
            u = tf.concat(axis=2, values=[fw_u, bw_u])
            if config.share_lstm_weights:
                tf.get_variable_scope().reuse_variables()
                (fw_h, bw_h), _ = bidirectional_dynamic_rnn(cell_fw, cell_bw, xx, x_len, dtype='float', scope='u1')  # [N, M, JX, 2d]
                h = tf.concat(axis=3, values=[fw_h, bw_h])  # [N, M, JX, 2d]

            else:
                (fw_h, bw_h), _ = bidirectional_dynamic_rnn(cell_fw, cell_bw, xx, x_len, dtype='float', scope='h1')  # [N, M, JX, 2d]
                h = tf.concat(axis=3, values=[fw_h, bw_h])  # [N, M, JX, 2d]
            self.tensor_dict['u'] = u
            self.tensor_dict['h'] = h
            h = tf.squeeze(h)
            x_mask = tf.expand_dims(tf.squeeze(self.x_mask),-1)
        with tf.variable_scope("self-attention"):
            for i in range(config.attention_layer_num):
                u = self_attention_layer(config, self.is_train, u, p_mask=tf.expand_dims(self.q_mask, -1),
                                         scope="{}_layer_self_att_enc".format(i+1))  # [N, len, dim]
                h = self_attention_layer(config, self.is_train, h, p_mask=x_mask,
                                         scope="{}_layer_self_att_enc_h".format(i+1))


        with tf.variable_scope("interact"):

            print('h___shape:{}'.format(h.get_shape()))
            print('u___shape:{}'.format(u.get_shape()))
            h = tf.expand_dims(h, 2)
            u = tf.expand_dims(u, 1)
            h = tf.tile(h, [1, 1, JQ, 1])
            u = tf.tile(u, [1, JX, 1, 1])
            attention = h * u  # N,JX,JQ,2d
            self.tensor_dict['atteention']=attention
        with tf.variable_scope('conv_dense'):
            out_final = dense_net(config, attention, self.is_train)
            self.tensor_dict['outfinal'] = out_final
            print('out_final_shape:{}'.format(out_final.get_shape()))

            self.prediction = linear(tf.concat([out_final],axis=-1), 1, True, bias_start=0.0, scope="logit", squeeze=False, wd=config.wd,
                                 input_keep_prob=config.output_keep_pro,
                                 is_train=self.is_train)


    def _build_loss(self):
        an=self.answers
        an=tf.squeeze(an)
        pr=tf.nn.sigmoid(tf.squeeze(self.prediction))

        loss2 = tf.reduce_mean(self.batch_mask * tf.square(tf.squeeze(tf.nn.sigmoid(self.prediction)) - an))

        tf.add_to_collection('losses', (loss2) * (self.config.batch_size) / tf.reduce_sum(self.batch_mask))

        self.acc = tf.reduce_mean(
            tf.cast(tf.equal(tf.arg_max(tf.reshape(pr,[-1,2]), dimension=1),
                             tf.arg_max(tf.cast(tf.reshape(an,[-1,2]), tf.int64), dimension=1)), tf.float32))

        tf.summary.scalar('model/acc', self.acc)
        # weights_added = tf.add_n([tf.nn.l2_loss(tensor) for tensor in tf.trainable_variables() if
        #                           tensor.name.endswith("weights:0") and not tensor.name.endswith(
        #                               "weighted_sum/weights:0") or tensor.name.endswith('kernel:0')])
        # full_l2_step = tf.constant(100000, dtype=tf.int32, shape=[], name='full_l2reg_step')
        # full_l2_ratio = tf.constant(9e-5, dtype=tf.float32, shape=[],
        #                             name='l2_regularization_ratio')
        # gs_flt = tf.cast(self.global_step, tf.float32)
        # half_l2_step_flt = tf.cast(full_l2_step / 2, tf.float32)
        #
        # # io = tf.sigmoid( tf.cast((self.global_step - full_l2_step / 2) * 8, tf.float32) / tf.cast(full_l2_step / 2 ,tf.float32)) * full_l2_ratio
        # l2loss_ratio = tf.sigmoid(((gs_flt - half_l2_step_flt) * 8) / half_l2_step_flt) * full_l2_ratio
        # tf.summary.scalar('l2loss_ratio', l2loss_ratio)
        # l2loss = weights_added * l2loss_ratio
        # tf.add_to_collection('losses', l2loss)


        self.loss = tf.add_n(tf.get_collection('losses', scope=self.scope), name='loss')
        tf.summary.scalar(self.loss.op.name, self.loss)
        tf.add_to_collection('ema/scalar', self.loss)

    def _build_ema(self):
        self.ema = tf.train.ExponentialMovingAverage(self.config.decay)
        ema = self.ema
        tensors = tf.get_collection("ema/scalar", scope=self.scope) + tf.get_collection("ema/vector", scope=self.scope)
        ema_op = ema.apply(tensors)
        for var in tf.get_collection("ema/scalar", scope=self.scope):
            ema_var = ema.average(var)
            tf.summary.scalar(ema_var.op.name, ema_var)
        for var in tf.get_collection("ema/vector", scope=self.scope):
            ema_var = ema.average(var)
            tf.summary.histogram(ema_var.op.name, ema_var)
        for var in tf.trainable_variables():
            ema_var = ema.average(var)
            if ema_var is not  None:
                tf.summary.histogram(ema_var.op.name, ema_var)
        with tf.control_dependencies([ema_op]):
            self.loss = tf.identity(self.loss)

    def _build_var_ema(self):
        self.var_ema = tf.train.ExponentialMovingAverage(self.config.var_decay)
        ema = self.var_ema
        ema_op = ema.apply(tf.trainable_variables())
        with tf.control_dependencies([ema_op]):
            self.loss = tf.identity(self.loss)

    def get_loss(self):
        return self.loss

    def get_global_step(self):
        return self.global_step

    def get_var_list(self):
        return self.var_list

    def get_feed_dict(self, batch, is_train, supervised=True):
        assert isinstance(batch, DataSet)
        config = self.config
        N, M, JX, JQ, VW, VC, d, W = \
            config.batch_size, config.max_num_sents, config.max_sent_size, \
            config.max_ques_size, config.word_vocab_size, config.char_vocab_size, config.hidden_size, config.max_word_size
        feed_dict = {}

        if config.len_opt:
            """
            Note that this optimization results in variable GPU RAM usage (i.e. can cause OOM in the middle of training.)
            First test without len_opt and make sure no OOM, and use len_opt
            """
            if sum(len(sent) for para in batch.data['x'] for sent in para) == 0:
                new_JX = 1
            else:
                new_JX = max(len(sent) for para in batch.data['x'] for sent in para)
            JX = min(JX, new_JX)

            if sum(len(ques) for ques in batch.data['q']) == 0:
                new_JQ = 1
            else:
                new_JQ = max(len(ques) for ques in batch.data['q'])
            JQ = min(JQ, new_JQ)

        if config.cpu_opt:
            if sum(len(para) for para in batch.data['x']) == 0:
                new_M = 1
            else:
                new_M = max(len(para) for para in batch.data['x'])
            M = min(M, new_M)

        x = np.zeros([N, M, JX], dtype='int32')
        cx = np.zeros([N, M, JX, W], dtype='int32')
        x_mask = np.zeros([N, M, JX], dtype='bool')
        x_sem = np.zeros([N, M, JX], dtype='int32')+3
        x_pos = np.zeros([N, M, JX], dtype='int32')+(config.pos_tag_num+1)
        x_neg = np.zeros([N, M, JX], dtype='int32')+3
        q = np.zeros([N, JQ], dtype='int32')
        cq = np.zeros([N, JQ, W], dtype='int32')
        q_mask = np.zeros([N, JQ], dtype='bool')
        q_sem = np.zeros([N, JQ], dtype='int32')+3
        q_pos = np.zeros([N, JQ], dtype='int32')+(config.pos_tag_num+1)
        q_neg = np.zeros([N, JQ], dtype='int32')+3
        answers = np.zeros([N],dtype = 'float32')
        batch_mask = np.zeros([N], dtype='float32')
        feed_dict[self.batch_mask] = batch_mask
        feed_dict[self.x] = x
        feed_dict[self.x_mask] = x_mask
        feed_dict[self.cx] = cx
        feed_dict[self.q] = q
        feed_dict[self.cq] = cq
        feed_dict[self.q_mask] = q_mask
        feed_dict[self.is_train] = is_train
        feed_dict[self.answers] = answers
        feed_dict[self.x_sem] = x_sem
        feed_dict[self.x_pos] = x_pos
        feed_dict[self.x_neg] = x_neg
        feed_dict[self.q_sem] = q_sem
        feed_dict[self.q_pos] = q_pos
        feed_dict[self.q_neg] = q_neg


        if config.use_glove_for_unk:
            feed_dict[self.new_emb_mat] = batch.shared['new_emb_mat']

        X = batch.data['x']
        CX = batch.data['cx']


        for i, answer in enumerate(batch.data['answerss']):
            answers[i]=answer
            batch_mask[i] = 1.0

        def _get_word(word):
            d = batch.shared['word2idx']
            for each in (word, word.lower(), word.capitalize(), word.upper()):
                if each in d:
                    return d[each]
            if config.use_glove_for_unk:
                d2 = batch.shared['new_word2idx']
                for each in (word, word.lower(), word.capitalize(), word.upper()):
                    if each in d2:
                        return d2[each] + len(d)
            return 1

        def _get_char(char):
            d = batch.shared['char2idx']
            if char in d:
                return d[char]
            return 1

        for i, xi in enumerate(zip(X,batch.data['x_neg'],batch.data['x_pos'],batch.data['x_sem'])):
            if self.config.squash:
                xi = [list(itertools.chain(*xi))]
            for j, xij in enumerate(zip(xi[0],xi[1],xi[2],xi[3])):
                if j == config.max_num_sents:
                    break
                for k, xijk in enumerate(zip(xij[0],xij[1],xij[2],xij[3])):
                    if k == config.max_sent_size:
                        break
                    each = _get_word(xijk[0])
                    assert isinstance(each, int), each
                    x[i, j, k] = each
                    x_mask[i, j, k] = True
                    x_sem[i, j, k]=xijk[3]
                    x_pos[i, j, k]=xijk[2]
                    x_neg[i, j, k]=xijk[1]

        for i, cxi in enumerate(CX):
            if self.config.squash:
                cxi = [list(itertools.chain(*cxi))]
            for j, cxij in enumerate(cxi):
                if j == config.max_num_sents:
                    break
                for k, cxijk in enumerate(cxij):
                    if k == config.max_sent_size:
                        break
                    for l, cxijkl in enumerate(cxijk):
                        if l == config.max_word_size:
                            break
                        cx[i, j, k, l] = _get_char(cxijkl)

        for i, qi in enumerate(zip(batch.data['q'],batch.data['q_neg'],batch.data['q_pos'],batch.data['q_sem'])):
            for j, qij in enumerate(zip(qi[0],qi[1],qi[2],qi[3])):
                q[i, j] = _get_word(qij[0])
                q_mask[i, j] = True
                q_sem[i,j]=qij[3]
                q_neg[i,j]=qij[1]
                q_pos[i,j]=qij[2]

        for i, cqi in enumerate(batch.data['cq']):
            for j, cqij in enumerate(cqi):
                for k, cqijk in enumerate(cqij):
                    cq[i, j, k] = _get_char(cqijk)
                    if k + 1 == config.max_word_size:
                        break

        return feed_dict



def self_attention_layer(config, is_train, p, p_mask=None, scope=None):
    with tf.variable_scope(scope or "self_attention_layer"):
        PL = tf.shape(p)[1]
        self_att = self_attention(config, is_train, p, p_mask=p_mask)
        print("self_att shape")
        print(self_att.get_shape())

        return self_att
def self_attention(config, is_train, p, p_mask=None, scope=None): #[N, L, 2d]
    with tf.variable_scope(scope or "self_attention"):

        JX = p.get_shape().as_list()[1]
        print(p.get_shape())

        p_aug_1 = tf.tile(tf.expand_dims(p, 2), [1,1,JX,1])
        p_aug_2 = tf.tile(tf.expand_dims(p, 1), [1,JX,1,1]) #[N, JX, JX, 2d]

        if p_mask is None:
            ph_mask = None
        else:
            p_mask_aug_1 = tf.reduce_any(tf.cast(tf.tile(tf.expand_dims(p_mask, 2), [1, 1, JX, 1]), tf.bool), axis=3)
            p_mask_aug_2 = tf.reduce_any(tf.cast(tf.tile(tf.expand_dims(p_mask, 1), [1, JX, 1, 1]), tf.bool), axis=3)
            self_mask = p_mask_aug_1 & p_mask_aug_2

        print(self_mask.get_shape().as_list())
        h_logits = get_logits([p_aug_1, p_aug_2], None, True, wd=config.wd, mask=self_mask,
                              is_train=is_train, func='tri_linear', scope='h_logits')  # [N, PL, HL]
        self_att = softsel(p_aug_2, h_logits)

        return self_att
