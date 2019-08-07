'''
Created on Oct 10, 2018
Tensorflow Implementation of Neural Graph Collaborative Filtering (NGCF) model in:
Wang Xiang et al. Neural Graph Collaborative Filtering. In SIGIR 2019.

@author: Xiang Wang (xiangwang@u.nus.edu)
'''
from __future__ import print_function

import tensorflow as tf
import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

from utility.helper import *
from utility.batch_test_dg import *


class DoubleGCF(object):
    def __init__(self, data_config, pretrain_data):
        # argument settings
        self.model_type = 'dgcf'
        self.adj_type = args.adj_type
        self.alg_type = args.alg_type

        self.pretrain_data = pretrain_data

        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']

        self.n_fold = 100

        # self.norm_adj = data_config['norm_adj']
        # self.n_nonzero_elems = self.norm_adj.count_nonzero()
        self.user_graph = data_config['user_graph']
        self.item_graph = data_config['item_graph']
        self.n_nonzero_elems_ug = self.user_graph.count_nonzero()
        self.n_nonzero_elems_ig = self.item_graph.count_nonzero()
        print(">>>", self.n_nonzero_elems_ug, self.n_nonzero_elems_ig)

        self.lr = args.lr

        # self.emb_dim = args.embed_size
        self.user_emb_dim = args.user_embed_size
        self.item_emb_dim = args.item_embed_size
        self.batch_size = args.batch_size

        # self.weight_size = eval(args.layer_size)
        # self.n_layers = len(self.weight_size)
        self.user_weight_size = eval(args.user_layer_size)
        self.n_user_layers = len(self.user_weight_size)
        self.item_weight_size = eval(args.item_layer_size)
        self.n_item_layers = len(self.item_weight_size)

        self.model_type += '_%s_%s_lu%d_li%d' % (self.adj_type, self.alg_type, self.n_user_layers, self.n_item_layers)

        self.regs = eval(args.regs)
        self.decay = self.regs[0]

        self.verbose = args.verbose

        '''
        *********************************************************
        Create Placeholder for Input Data & Dropout.
        '''
        # placeholder definition
        self.users = tf.placeholder(tf.int32, shape=(None,))
        self.pos_items = tf.placeholder(tf.int32, shape=(None,))
        self.neg_items = tf.placeholder(tf.int32, shape=(None,))

        # dropout: node dropout (adopted on the ego-networks);
        #          ... since the usage of node dropout have higher computational cost,
        #          ... please use the 'node_dropout_flag' to indicate whether use such technique.
        #          message dropout (adopted on the convolution operations).
        self.node_dropout_flag = args.node_dropout_flag
        self.node_dropout = tf.placeholder(tf.float32, shape=[None])
        # self.mess_dropout = tf.placeholder(tf.float32, shape=[None])

        """
        *********************************************************
        Create Model Parameters (i.e., Initialize Weights).
        """
        # initialization of model parameters
        self.weights = self._init_weights()

        """
        *********************************************************
        Compute Graph-based Representations of all users & items via Message-Passing Mechanism of Graph Neural Networks.
        Different Convolutional Layers:
            1. ngcf: defined in 'Neural Graph Collaborative Filtering', SIGIR2019;
            2. gcn:  defined in 'Semi-Supervised Classification with Graph Convolutional Networks', ICLR2018;
            3. gcmc: defined in 'Graph Convolutional Matrix Completion', KDD2018;
        """
        if self.alg_type in ['ngcf']:
            self.ua_embeddings, self.ia_embeddings = self._create_ngcf_embed()

        elif self.alg_type in ['gcn']:
            self.ua_embeddings, self.ia_embeddings = self._create_gcn_embed()

        elif self.alg_type in ['gcmc']:
            self.ua_embeddings, self.ia_embeddings = self._create_gcmc_embed()

        """
        *********************************************************
        Establish the final representations for user-item pairs in batch.
        """
        self.u_g_embeddings = tf.nn.embedding_lookup(self.ua_embeddings, self.users)
        # self.u_g_embeddings = tf.Print(self.u_g_embeddings, [tf.shape(self.u_g_embeddings)], "u_g_embeddings:")
        self.pos_i_g_embeddings = tf.nn.embedding_lookup(self.ia_embeddings, self.pos_items)
        # self.pos_i_g_embeddings = tf.Print(self.pos_i_g_embeddings, [tf.shape(self.u_g_embeddings)], "pos_i_g_embeddings:")
        self.neg_i_g_embeddings = tf.nn.embedding_lookup(self.ia_embeddings, self.neg_items)

        """
        *********************************************************
        Inference for the testing phase.
        """
        self.batch_ratings = tf.matmul(self.u_g_embeddings, self.pos_i_g_embeddings, transpose_a=False, transpose_b=True)

        """
        *********************************************************
        Generate Predictions & Optimize via BPR loss.
        """
        self.mf_loss, self.emb_loss, self.reg_loss = self.create_bpr_loss(self.u_g_embeddings,
                                                                          self.pos_i_g_embeddings,
                                                                          self.neg_i_g_embeddings)
        self.loss = self.mf_loss + self.emb_loss + self.reg_loss

        self.opt = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

    def _init_weights(self):
        all_weights = dict()

        initializer = tf.contrib.layers.xavier_initializer()

        if self.pretrain_data is None:
            all_weights['user_embedding'] = tf.Variable(initializer([self.n_users, self.user_emb_dim]), name='user_embedding')
            all_weights['item_embedding'] = tf.Variable(initializer([self.n_items, self.item_emb_dim]), name='item_embedding')
            print('using xavier initialization')
        else:
            all_weights['user_embedding'] = tf.Variable(initial_value=self.pretrain_data['user_embed'], trainable=True,
                                                        name='user_embedding', dtype=tf.float32)
            all_weights['item_embedding'] = tf.Variable(initial_value=self.pretrain_data['item_embed'], trainable=True,
                                                        name='item_embedding', dtype=tf.float32)
            print('using pretrained initialization')

        if args.double_graph_type != 3:
            self.user_weight_size_list = [self.user_emb_dim] + self.user_weight_size
            for k in range(self.n_user_layers):
                all_weights['W_ugc_%d' %k] = tf.Variable(
                    initializer([self.user_weight_size_list[k], self.user_weight_size_list[k+1]]), name='W_ugc_%d' % k)
                all_weights['b_ugc_%d' %k] = tf.Variable(
                    initializer([1, self.user_weight_size_list[k+1]]), name='b_ugc_%d' % k)

                all_weights['W_ubi_%d' % k] = tf.Variable(
                    initializer([self.user_weight_size_list[k], self.user_weight_size_list[k + 1]]), name='W_ubi_%d' % k)
                all_weights['b_ubi_%d' % k] = tf.Variable(
                    initializer([1, self.user_weight_size_list[k + 1]]), name='b_ubi_%d' % k)

                all_weights['W_umlp_%d' % k] = tf.Variable(
                    initializer([self.user_weight_size_list[k], self.user_weight_size_list[k+1]]), name='W_umlp_%d' % k)
                all_weights['b_umlp_%d' % k] = tf.Variable(
                    initializer([1, self.user_weight_size_list[k+1]]), name='b_umlp_%d' % k)

        if args.double_graph_type != 2:
            self.item_weight_size_list = [self.item_emb_dim] + self.item_weight_size
            for k in range(self.n_item_layers):
                all_weights['W_igc_%d' %k] = tf.Variable(
                    initializer([self.item_weight_size_list[k], self.item_weight_size_list[k+1]]), name='W_igc_%d' % k)
                all_weights['b_igc_%d' %k] = tf.Variable(
                    initializer([1, self.item_weight_size_list[k+1]]), name='b_igc_%d' % k)

                all_weights['W_ibi_%d' % k] = tf.Variable(
                    initializer([self.item_weight_size_list[k], self.item_weight_size_list[k + 1]]), name='W_ibi_%d' % k)
                all_weights['b_ibi_%d' % k] = tf.Variable(
                    initializer([1, self.item_weight_size_list[k + 1]]), name='b_ibi_%d' % k)

                all_weights['W_imlp_%d' % k] = tf.Variable(
                    initializer([self.item_weight_size_list[k], self.item_weight_size_list[k+1]]), name='W_imlp_%d' % k)
                all_weights['b_imlp_%d' % k] = tf.Variable(
                    initializer([1, self.item_weight_size_list[k+1]]), name='b_imlp_%d' % k)

        return all_weights

    def _split_A_hat(self, X, is_user=True):
        A_fold_hat = []

        fold_len = self.n_users // self.n_fold if is_user else self.n_items // self.n_fold
        for i_fold in range(self.n_fold):
            start = i_fold * fold_len
            if i_fold == self.n_fold -1:
                end = self.n_users if is_user else self.n_items
            else:
                end = (i_fold + 1) * fold_len

            A_fold_hat.append(self._convert_sp_mat_to_sp_tensor(X[start:end]))
        return A_fold_hat

    def _split_A_hat_node_dropout(self, X, is_user=True):
        A_fold_hat = []

        fold_len = self.n_users // self.n_fold if is_user else self.n_items // self.n_fold
        for i_fold in range(self.n_fold):
            start = i_fold * fold_len
            if i_fold == self.n_fold -1:
                end = self.n_users if is_user else self.n_items
            else:
                end = (i_fold + 1) * fold_len

            # A_fold_hat.append(self._convert_sp_mat_to_sp_tensor(X[start:end]))
            temp = self._convert_sp_mat_to_sp_tensor(X[start:end])
            n_nonzero_temp = X[start:end].count_nonzero()
            A_fold_hat.append(self._dropout_sparse(temp, 1 - self.node_dropout[0], n_nonzero_temp))

        return A_fold_hat

    def _create_ngcf_embed(self):
        # Generate a set of adjacency sub-matrix.
        if self.node_dropout_flag:
            # node dropout.
            if args.double_graph_type != 3:
                A_fold_hat_user = self._split_A_hat_node_dropout(self.user_graph)
            if args.double_graph_type != 2:
                A_fold_hat_item = self._split_A_hat_node_dropout(self.item_graph, is_user=False)
        else:
            if args.double_graph_type != 3:
                A_fold_hat_user = self._split_A_hat(self.user_graph)
            if args.double_graph_type != 2:
                A_fold_hat_item = self._split_A_hat(self.item_graph, is_user=False)

        # ego_embeddings = tf.concat([self.weights['user_embedding'], self.weights['item_embedding']], axis=0)

        u_g_embeddings = self._generate_ngcf_embed(A_fold_hat_user, self.weights['user_embedding']) if args.double_graph_type != 3 else self.weights['user_embedding']
        i_g_embeddings = self._generate_ngcf_embed(A_fold_hat_item, self.weights['item_embedding'], is_user=False) if args.double_graph_type != 2 else self.weights['item_embedding']
        return u_g_embeddings, i_g_embeddings

    def _generate_ngcf_embed(self, A_fold_hat, ego_embeddings, is_user=True):
        n_layers = self.n_user_layers if is_user else self.n_item_layers
        all_embeddings = [ego_embeddings]
        for k in range(0, n_layers):

            temp_embed = []
            for f in range(self.n_fold):
                temp_embed.append(tf.sparse_tensor_dense_matmul(A_fold_hat[f], ego_embeddings))

            # sum messages of neighbors.
            side_embeddings = tf.concat(temp_embed, 0)
            # transformed sum messages of neighbors.
            wgc = self.weights['W_ugc_%d' % k] if is_user else self.weights['W_igc_%d' % k]
            bgc = self.weights['b_ugc_%d' % k] if is_user else self.weights['b_igc_%d' % k]
            sum_embeddings = tf.nn.leaky_relu( tf.matmul(side_embeddings, wgc) + bgc)

            # bi messages of neighbors.
            bi_embeddings = tf.multiply(ego_embeddings, side_embeddings)
            # transformed bi messages of neighbors.
            wbi = self.weights['W_ubi_%d' % k] if is_user else self.weights['W_ibi_%d' % k]
            bbi = self.weights['b_ubi_%d' % k] if is_user else self.weights['b_ibi_%d' % k]
            bi_embeddings = tf.nn.leaky_relu(tf.matmul(bi_embeddings, wbi) + bbi)

            # non-linear activation.
            ego_embeddings = sum_embeddings + bi_embeddings

            # message dropout.
            # ego_embeddings = tf.nn.dropout(ego_embeddings, 1 - self.mess_dropout[k])
            ego_embeddings = tf.nn.dropout(ego_embeddings, 0.9) # wdckpt: fixed dropout

            # normalize the distribution of embeddings.
            norm_embeddings = tf.math.l2_normalize(ego_embeddings, axis=1)

            all_embeddings += [norm_embeddings]

        # all_embeddings = tf.concat(all_embeddings, 1)
        # u_g_embeddings, i_g_embeddings = tf.split(all_embeddings, [self.n_users, self.n_items], 0)
        return all_embeddings[-1]

    def _create_gcn_embed(self):
        if args.double_graph_type != 3:
            A_fold_hat_user = self._split_A_hat(self.user_graph)
            u_g_embeddings = self._generate_gcn_embed(A_fold_hat_user, self.weights['user_embedding'])
        else:
            u_g_embeddings = self.weights['user_embedding']

        if args.double_graph_type != 2:
            A_fold_hat_item = self._split_A_hat(self.item_graph, is_user=False)
            i_g_embeddings = self._generate_gcn_embed(A_fold_hat_item, self.weights['item_embedding'], is_user=False)
        else:
            i_g_embeddings = self.weights['item_embedding']

        return u_g_embeddings, i_g_embeddings


    def _generate_gcn_embed(self, A_fold_hat, embeddings, is_user=True):
        n_layers = self.n_user_layers if is_user else self.n_item_layers
        all_embeddings = [embeddings]

        for k in range(0, n_layers):
            temp_embed = []
            for f in range(self.n_fold):
                temp_embed.append(tf.sparse_tensor_dense_matmul(A_fold_hat[f], embeddings))

            embeddings = tf.concat(temp_embed, 0)
            wgc = self.weights['W_ugc_%d' %k] if is_user else self.weights['W_igc_%d' %k]
            bgc = self.weights['b_ugc_%d' %k] if is_user else self.weights['b_igc_%d' %k]
            embeddings = tf.nn.leaky_relu(tf.matmul(embeddings, wgc) + bgc)
            # embeddings = tf.nn.dropout(embeddings, 1 - self.mess_dropout[k])
            embeddings = tf.nn.dropout(embeddings, 0.1)  # wdckpt

            all_embeddings += [embeddings]

        all_embeddings = tf.concat(all_embeddings, 1)
        # u_g_embeddings, i_g_embeddings = tf.split(all_embeddings, [self.n_users, self.n_items], 0)
        return all_embeddings

    def _create_gcmc_embed(self):
        A_fold_hat = self._split_A_hat(self.norm_adj)

        embeddings = tf.concat([self.weights['user_embedding'], self.weights['item_embedding']], axis=0)

        all_embeddings = []

        for k in range(0, self.n_layers):
            temp_embed = []
            for f in range(self.n_fold):
                temp_embed.append(tf.sparse_tensor_dense_matmul(A_fold_hat[f], embeddings))
            embeddings = tf.concat(temp_embed, 0)
            # convolutional layer.
            embeddings = tf.nn.leaky_relu(tf.matmul(embeddings, self.weights['W_gc_%d' % k]) + self.weights['b_gc_%d' % k])
            # dense layer.
            mlp_embeddings = tf.matmul(embeddings, self.weights['W_mlp_%d' %k]) + self.weights['b_mlp_%d' %k]
            # mlp_embeddings = tf.nn.dropout(mlp_embeddings, 1 - self.mess_dropout[k])
            mlp_embeddings = tf.nn.dropout(mlp_embeddings, 0.9) # wdckpt

            all_embeddings += [mlp_embeddings]
        all_embeddings = tf.concat(all_embeddings, 1)

        u_g_embeddings, i_g_embeddings = tf.split(all_embeddings, [self.n_users, self.n_items], 0)
        return u_g_embeddings, i_g_embeddings


    def create_bpr_loss(self, users, pos_items, neg_items):
        pos_scores = tf.reduce_sum(tf.multiply(users, pos_items), axis=1)
        neg_scores = tf.reduce_sum(tf.multiply(users, neg_items), axis=1)

        regularizer = tf.nn.l2_loss(users) + tf.nn.l2_loss(pos_items) + tf.nn.l2_loss(neg_items)
        regularizer = regularizer/self.batch_size

        maxi = tf.log(tf.nn.sigmoid(pos_scores - neg_scores))
        mf_loss = tf.negative(tf.reduce_mean(maxi))

        emb_loss = self.decay * regularizer

        reg_loss = tf.constant(0.0, tf.float32, [1])

        return mf_loss, emb_loss, reg_loss

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        indices = np.mat([coo.row, coo.col]).transpose()
        return tf.SparseTensor(indices, coo.data, coo.shape)

    def _dropout_sparse(self, X, keep_prob, n_nonzero_elems):
        """
        Dropout for sparse tensors.
        """
        noise_shape = [n_nonzero_elems]
        random_tensor = keep_prob
        random_tensor += tf.random_uniform(noise_shape)
        dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
        pre_out = tf.sparse_retain(X, dropout_mask)

        return pre_out * tf.div(1., keep_prob)

def load_pretrained_data():
    pretrain_path = '%spretrain/%s/%s.npz' % (args.proj_path, args.dataset, 'embedding')
    try:
        pretrain_data = np.load(pretrain_path)
        print('load the pretrained embeddings.')
    except Exception:
        pretrain_data = None
    return pretrain_data

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    config = dict()
    config['n_users'] = data_generator.n_users
    config['n_items'] = data_generator.n_items

    """
    *********************************************************
    Generate the Laplacian matrix, where each entry defines the decay factor (e.g., p_ui) between two connected nodes.
    """
    user_graph, norm_user_graph, mean_user_graph, item_graph, norm_item_graph, mean_item_graph = data_generator.get_adj_mat()

    if args.adj_type == 'plain':
        config['user_graph'] = user_graph
        config['item_graph'] = item_graph
        print('use the plain adjacency matrix')

    elif args.adj_type == 'norm':
        config['user_graph'] = norm_user_graph
        config['item_graph'] = norm_item_graph
        print('use the normalized adjacency matrix')

    elif args.adj_type == 'gcmc':
        config['user_graph'] = mean_user_graph
        config['item_graph'] = mean_item_graph
        print('use the gcmc adjacency matrix')

    else:
        config['user_graph'] = mean_user_graph + sp.eye(mean_user_graph.shape[0])
        config['item_graph'] = mean_item_graph + sp.eye(mean_item_graph.shape[0])
        print('use the mean adjacency matrix')

    t0 = time()

    if args.pretrain == -1:
        pretrain_data = load_pretrained_data()
    else:
        pretrain_data = None

    model = DoubleGCF(data_config=config, pretrain_data=pretrain_data)

    """
    *********************************************************
    Save the model parameters.
    """
    saver = tf.train.Saver()

    if args.save_flag == 1:
        layer = '-'.join([str(l) for l in eval(args.layer_size)])
        weights_save_path = '%sweights/%s/%s/%s/l%s_r%s' % (args.weights_path, args.dataset, model.model_type, layer,
                                                            str(args.lr), '-'.join([str(r) for r in eval(args.regs)]))
        ensureDir(weights_save_path)
        save_saver = tf.train.Saver(max_to_keep=1)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    """
    *********************************************************
    Reload the pretrained model parameters.
    """
    if args.pretrain == 1:
        layer = '-'.join([str(l) for l in eval(args.layer_size)])

        pretrain_path = '%sweights/%s/%s/%s/l%s_r%s' % (args.weights_path, args.dataset, model.model_type, layer,
                                                        str(args.lr), '-'.join([str(r) for r in eval(args.regs)]))


        ckpt = tf.train.get_checkpoint_state(os.path.dirname(pretrain_path + '/checkpoint'))
        if ckpt and ckpt.model_checkpoint_path:
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, ckpt.model_checkpoint_path)
            print('load the pretrained model parameters from: ', pretrain_path)

            # *********************************************************
            # get the performance from pretrained model.
            if args.report != 1:
                users_to_test = list(data_generator.test_set.keys())
                ret = test(sess, model, users_to_test, drop_flag=True)
                cur_best_pre_0 = ret['recall'][0]

                pretrain_ret = 'pretrained model recall=[%.5f, %.5f], precision=[%.5f, %.5f], hit=[%.5f, %.5f],' \
                               'ndcg=[%.5f, %.5f]' % \
                               (ret['recall'][0], ret['recall'][-1],
                                ret['precision'][0], ret['precision'][-1],
                                ret['hit_ratio'][0], ret['hit_ratio'][-1],
                                ret['ndcg'][0], ret['ndcg'][-1])
                print(pretrain_ret)
        else:
            sess.run(tf.global_variables_initializer())
            cur_best_pre_0 = 0.
            print('without pretraining.')

    else:
        sess.run(tf.global_variables_initializer())
        cur_best_pre_0 = 0.
        print('without pretraining.')

    """
    *********************************************************
    Get the performance w.r.t. different sparsity levels.
    """
    if args.report == 1:
        assert args.test_flag == 'full'
        users_to_test_list, split_state = data_generator.get_sparsity_split()
        users_to_test_list.append(list(data_generator.test_set.keys()))
        split_state.append('all')

        report_path = '%sreport/%s/%s.result' % (args.proj_path, args.dataset, model.model_type)
        ensureDir(report_path)
        f = open(report_path, 'w')
        f.write(
            'user_embed_size=%d, item_embed_size=%d, lr=%.4f, layer_size=%s, keep_prob=%s, regs=%s, loss_type=%s, adj_type=%s\n'
            % (args.user_embed_size, args.item_embed_size, args.lr, args.layer_size, args.keep_prob, args.regs, args.loss_type, args.adj_type))

        for i, users_to_test in enumerate(users_to_test_list):
            ret = test(sess, model, users_to_test, drop_flag=True)

            final_perf = "recall=[%s], precision=[%s], hit=[%s], ndcg=[%s]" % \
                         ('\t'.join(['%.5f' % r for r in ret['recall']]),
                          '\t'.join(['%.5f' % r for r in ret['precision']]),
                          '\t'.join(['%.5f' % r for r in ret['hit_ratio']]),
                          '\t'.join(['%.5f' % r for r in ret['ndcg']]))
            print(final_perf)

            f.write('\t%s\n\t%s\n' % (split_state[i], final_perf))
        f.close()
        exit()

    """
    *********************************************************
    Train.
    """
    loss_loger, pre_loger, rec_loger, ndcg_loger, hit_loger = [], [], [], [], []
    stopping_step = 0
    should_stop = False

    for epoch in range(args.epoch):
        t1 = time()
        loss, mf_loss, emb_loss, reg_loss = 0., 0., 0., 0.
        n_batch = data_generator.n_train // args.batch_size + 1

        for idx in range(n_batch):
            users, pos_items, neg_items = data_generator.sample()
            _, batch_loss, batch_mf_loss, batch_emb_loss, batch_reg_loss = sess.run([model.opt, model.loss, model.mf_loss, model.emb_loss, model.reg_loss],
                               feed_dict={model.users: users, model.pos_items: pos_items,
                                          model.node_dropout: eval(args.node_dropout),
                                        #   model.mess_dropout: eval(args.mess_dropout),
                                          model.neg_items: neg_items})
            loss += batch_loss
            mf_loss += batch_mf_loss
            emb_loss += batch_emb_loss
            reg_loss += batch_reg_loss

        if np.isnan(loss) == True:
            print('ERROR: loss is nan.')
            sys.exit()

        # print the test evaluation metrics each 10 epochs; pos:neg = 1:10.
        if (epoch + 1) % 10 != 0:
            if args.verbose > 0 and epoch % args.verbose == 0:
                perf_str = 'Epoch %d [%.1fs]: train==[%.5f=%.5f + %.5f]' % (
                    epoch, time() - t1, loss, mf_loss, reg_loss)
                print(perf_str)
            continue

        t2 = time()
        users_to_test = list(data_generator.test_set.keys())
        ret = test(sess, model, users_to_test, drop_flag=True)

        t3 = time()

        loss_loger.append(loss)
        rec_loger.append(ret['recall'])
        pre_loger.append(ret['precision'])
        ndcg_loger.append(ret['ndcg'])
        hit_loger.append(ret['hit_ratio'])

        if args.verbose > 0:
            perf_str = 'Epoch %d [%.1fs + %.1fs]: train==[%.5f=%.5f + %.5f + %.5f], recall=[%.5f, %.5f], ' \
                       'precision=[%.5f, %.5f], hit=[%.5f, %.5f], ndcg=[%.5f, %.5f]' % \
                       (epoch, t2 - t1, t3 - t2, loss, mf_loss, emb_loss, reg_loss, ret['recall'][0], ret['recall'][-1],
                        ret['precision'][0], ret['precision'][-1], ret['hit_ratio'][0], ret['hit_ratio'][-1],
                        ret['ndcg'][0], ret['ndcg'][-1])
            print(perf_str)

        cur_best_pre_0, stopping_step, should_stop = early_stopping(ret['recall'][0], cur_best_pre_0,
                                                                    stopping_step, expected_order='acc', flag_step=5)

        # *********************************************************
        # early stopping when cur_best_pre_0 is decreasing for ten successive steps.
        if should_stop == True:
            break

        # *********************************************************
        # save the user & item embeddings for pretraining.
        if ret['recall'][0] == cur_best_pre_0 and args.save_flag == 1:
            save_saver.save(sess, weights_save_path + '/weights', global_step=epoch)
            print('save the weights in path: ', weights_save_path)

    recs = np.array(rec_loger)
    pres = np.array(pre_loger)
    ndcgs = np.array(ndcg_loger)
    hit = np.array(hit_loger)

    best_rec_0 = max(recs[:, 0])
    idx = list(recs[:, 0]).index(best_rec_0)

    final_perf = "Best Iter=[%d]@[%.1f]\trecall=[%s], precision=[%s], hit=[%s], ndcg=[%s]" % \
                 (idx, time() - t0, '\t'.join(['%.5f' % r for r in recs[idx]]),
                  '\t'.join(['%.5f' % r for r in pres[idx]]),
                  '\t'.join(['%.5f' % r for r in hit[idx]]),
                  '\t'.join(['%.5f' % r for r in ndcgs[idx]]))
    print(final_perf)

    save_path = '%soutput/%s/%s.result' % (args.proj_path, args.dataset, model.model_type)
    ensureDir(save_path)
    f = open(save_path, 'a')

    f.write(
        'user_embed_size=%d, item_embed_size=%d, lr=%.4f, layer_size=%s, node_dropout=%s, mess_dropout=%s, regs=%s, adj_type=%s\n\t%s\n'
        % (args.user_embed_size, args.item_embed_size, args.lr, args.layer_size, args.node_dropout, args.mess_dropout, args.regs,
           args.adj_type, final_perf))
    f.close()
