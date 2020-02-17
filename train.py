from tqdm import tqdm
import tensorflow as tf
import argparse
import numpy as np
import sys
import time
import math
import random
from collections import defaultdict
import os

from utils import *
from model import *
from sampler import *

parser = argparse.ArgumentParser(description='Occasion-aware Sequential Recommendation')
parser.add_argument('--batch_size', type=int, default=128, help='batch size (default: 128)')
parser.add_argument('--seq_len', type=int, default=50, help='max sequence length (default: 50)')
parser.add_argument('--dropout', type=float, default=0.5, help='dropout (default: 0.5)')
parser.add_argument('--l2_reg', type=float, default=0.0, help='regularization scale (default: 0.0)')
parser.add_argument('--epochs', type=int, default=20, help='upper epoch limit (default: 20)')
parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate for Adam (default: 0.001)')
parser.add_argument('--emsize', type=int, default=100, help='dimension of item embedding (default: 100)')
parser.add_argument('--worker', type=int, default=3, help='number of sampling workers (default: 10)')
parser.add_argument('--data', type=str, default='amazon', help='data set name (default: amazon)')
parser.add_argument('--eval_interval', type=int, default=5, help='eval/test interval (default: 1e3)')
parser.add_argument('--eval_batch_size', type=int, default=2048, help='eval/test batch size (default: 2048)')

# ****************************** unique arguments for transformer model. *************************************************
parser.add_argument('--num_blocks', type=int, default=2, help='num_blocks')
parser.add_argument('--num_heads', type=int, default=1, help='num_heads')
parser.add_argument('--pos_fixed', type=int, default=0, help='trainable positional embedding usually has better performance')


args = parser.parse_args()

train_data, val_data, test_data, neg_test, n_items, n_users = data_partition_time(args)

print(args)
print ('#Item: ', n_items)
print ('#User: ', n_users)


model = OAR(args, n_items, n_users)

lr = args.lr


def main():
    global lr
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    init = tf.global_variables_initializer()
    sess.run(init)
    
    k_list = [5,10,15,20] # Ks for Top-K evaluation
    test_batch = prepare_eval_test_time([train_data, val_data, test_data, neg_test, n_items, n_users], args)

    print('Start training...')
    T = 0.0
    t0 = time.time()
    try:
                
        train_sampler = Sampler_time(
                    data=train_data, 
                    n_items=n_items, 
                    n_users=n_users,
                    batch_size=args.batch_size, 
                    max_len=args.seq_len,
                    neg_size=1,
                    n_workers=args.worker,
                    neg_method='rand')

        num_batch = int( n_users / args.batch_size)

        for epoch in range(0, args.epochs + 1):
            loss_sum = 0
            for step in tqdm(range(num_batch), total=num_batch, ncols=70, leave=False, unit='b'):
                cur_batch = train_sampler.next_batch()
                inp = np.array(cur_batch[1])
                feed_dict = {model.u_list: np.array(cur_batch[0]), model.inp: inp, model.lr: lr, model.dropout: args.dropout}
                feed_dict[model.pos] = np.array(cur_batch[2])
                feed_dict[model.neg] = np.array(cur_batch[3])
                feed_dict[model.inp_time] = np.array(cur_batch[4])
                feed_dict[model.pos_time] = np.array(cur_batch[5])


                _, train_loss = sess.run([model.train_op_all, model.loss_all], feed_dict=feed_dict)
                loss_sum += train_loss

            print('epoch:%d, training loss (total): %f' % (epoch, loss_sum))

            if  epoch % args.eval_interval == 0:

                print ('Evaluating',)

                t_test = evaluate_batch_time(model, test_batch, args, sess, k_list)
                print ('')
                print ('epoch:%d, time: %f(s)' % (epoch, T))
                print('NDCG', t_test[0])
                print('HT', t_test[1]) 
                print('MRR', t_test[2])
                print('AUC', t_test[3])

               
        
    except Exception as e:
        print(str(e))
        train_sampler.close()
        exit(1)
    train_sampler.close()
    print('Done')

if __name__ == '__main__':
    main()
