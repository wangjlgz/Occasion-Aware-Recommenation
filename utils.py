import pandas as pd
import numpy as np
import random
import os
import json
import copy
from tqdm import tqdm
from collections import defaultdict
from collections import Counter
from sklearn.metrics import roc_auc_score
import time
import datetime

data_path = 'data/'


####################################
#split the training, validation and testing
#we already sort the purchase record in chronological order per user
#the last two records of each user are after the cutting time (acting as validation and testing case)
#####################################

def data_partition_time(args): 
    usernum = 0
    itemnum = 0
    User = defaultdict(list)
    user_train = {}
    user_valid = {}
    user_test = {}
    neg_test = {}
    train_dic = {}
    # assume user/item index starting from 1
    path_to_data = data_path + args.data + '/' + args.data + '_all.txt'
    
    # assume user/item index starting from 1
    f = open(path_to_data, 'r')
    for line in f:
        u, i, t = line.rstrip().split(' ')
        u = int(u)
        i = int(i)
        t = int(datetime.datetime.fromtimestamp(int(t)).strftime("%j")) # Day of the year as a decimal number [001,366]
        usernum = max(u, usernum)
        itemnum = max(i, itemnum)
        User[str(u)].append((i,t))


    for user in User:
        nfeedback = len(User[user])
        neg_test[user] = []
        if nfeedback < 3:
            user_train[user] = User[user]
            user_valid[user] = []
            user_test[user] = []
        else:
            user_train[user] = User[user][:-2]
            user_valid[user] = []
            user_valid[user].append(User[user][-2])
            user_test[user] = []
            user_test[user].append(User[user][-1])


    # randomly select 100 negative items for testing 
    # https://github.com/kang205/SASRec/blob/master/util.py
    print('sampling 100 negative items for each sequence (for testing):')
    for user in tqdm(User, total=len(User), ncols=100, leave=False, unit='b'):
        rated = set(user_train[user])
        neg_test[user] = []
        for _ in range(100):
            t = np.random.randint(1, itemnum + 1)
            while t in rated or t==user_test[user][0]: t = np.random.randint(1, itemnum + 1)
            neg_test[user].append(t)
       

    return [user_train, user_valid, user_test, neg_test, itemnum, usernum]



def prepare_eval_test_time(dataset, args):
    [train, valid, test, neg_test, itemnum, usernum] = copy.deepcopy(dataset)
    batch_size = args.eval_batch_size
    if batch_size < 2:
        batch_size = 2
    uids = test.keys()
    all_u = []
    all_inp = []
    all_inp_time = []
    all_pos = []
    all_pos_time = []

    for u in uids:
        all_u.append(int(u))

        inp = np.zeros([args.seq_len], dtype=np.int32)
        inp_time = np.zeros([args.seq_len], dtype=np.int32)

        idx = args.seq_len - 1
        for i in reversed(train[u]):
            inp[idx] = i[0]
            inp_time[idx] = i[1]
            idx -= 1
            if idx == -1: break

        item_idx = np.array([test[u][0][0]] + neg_test[u])
        item_idx_time = np.array([test[u][0][1]]*(1+len(neg_test[u])))

        
        all_inp.append(inp)
        all_inp_time.append(inp_time)
        all_pos.append(item_idx)
        all_pos_time.append(item_idx_time)


    num_batches = int(len(all_u) / batch_size)
    batches = []
    for i in range(num_batches):
        batch_u = all_u[i*batch_size: (i+1)*batch_size]
        batch_inp = all_inp[i*batch_size: (i+1)*batch_size]
        batch_pos = all_pos[i*batch_size: (i+1)*batch_size]
        batch_inp_time = all_inp_time[i*batch_size: (i+1)*batch_size]
        batch_pos_time = all_pos_time[i*batch_size: (i+1)*batch_size]
        batches.append((batch_u, batch_inp, batch_pos, batch_inp_time, batch_pos_time))
    if num_batches * batch_size < len(all_u):
        batches.append((all_u[num_batches * batch_size:], all_inp[num_batches * batch_size:], all_pos[num_batches * batch_size:],\
            all_inp_time[num_batches * batch_size:], all_pos_time[num_batches * batch_size:]))
        
    return batches


def evaluate_batch_time(model, test_batch, args, sess, k_list): # predict for test/valid based on the training sequence
    
    valid_user = 0
    HT = [0.0000 for k in k_list]
    NDCG = [0.0000 for k in k_list]
    MRR = 0.0000
    AUC = 0.0000
    true_label = [0 for i in range(101)]
    true_label[0] = 1

    for step in tqdm(range(len(test_batch)), total=len(test_batch), ncols=70, leave=False, unit='b'):
        
        batch = test_batch[step]

        feed_dict = {model.u_list: batch[0], model.inp: batch[1], model.dropout: 0.}
        feed_dict[model.test_item_batch] = batch[2]
        feed_dict[model.inp_time] = batch[3]
        feed_dict[model.pos_time] = batch[4]
        
        predictions_batch = sess.run([model.test_logits_batch_all], feed_dict=feed_dict)

        predictions_batch = predictions_batch[0]

        for re in range(predictions_batch.shape[0]):
            
            predictions = -predictions_batch[re,:]
            rank = predictions.argsort().argsort()[0]

            valid_user += 1

            MRR += 1.0/(rank+1)
            
            AUC += roc_auc_score(y_true = true_label, y_score = -predictions)
            
            for k in range(len(k_list)):  
                if rank < k_list[k]:
                    NDCG[k] += 1.0 / np.log2(rank + 2)
                    HT[k] += 1

    return [NDCG[k]*1.0 / valid_user for k in range(len(k_list))], [HT[k]*1.0 / valid_user for k in range(len(k_list))], MRR*1.0 / valid_user, AUC*1.0 / valid_user