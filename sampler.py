import numpy as np
import random
from multiprocessing import Process, Queue

def random_neg(pos, n, s, ts):
    '''
    p: positive one
    n: number of items
    s: size of samples.
    '''
    neg = set()
    for _ in range(s):
        t = np.random.randint(1, n+1)
        while t in pos or t in neg or t in ts:
            t = np.random.randint(1, n+1)
        neg.add(t)
    return list(neg)




def sample_function_time(data, n_items, n_users, batch_size, max_len, neg_size, result_queue, SEED, neg_method='rand'):
    def sample():

        u = np.random.randint(1, n_users + 1)        
        while len(data[str(u)]) <= 1: u = np.random.randint(1, n_users + 1)
        user = str(u)

        seq = np.zeros([max_len], dtype=np.int32)
        seq_time = np.zeros([max_len], dtype=np.int32)
        pos = np.zeros([max_len], dtype=np.int32)
        pos_time = np.zeros([max_len], dtype=np.int32)
        neg = np.zeros([max_len, neg_size], dtype=np.int32)
        nxt = data[user][-1][0]
        nxt_time = data[user][-1][1]
        idx = max_len - 1


        ts = set([i[0] for i in data[user]])
        for i in reversed(data[user][:-1]):
            seq[idx] = i[0]
            seq_time[idx] = i[1]
            pos[idx] = nxt
            pos_time[idx] = nxt_time
            if nxt != 0: neg[idx,:] = random_neg([pos[idx]], n_items, neg_size, ts)
            nxt = i[0]
            nxt_time = i[1]
            idx -= 1
            if idx == -1: break

        return (u, seq, pos, neg, seq_time, pos_time)

    np.random.seed(SEED)
    while True:
        one_batch = []
        for i in range(batch_size):
            one_batch.append(sample())

        result_queue.put(list(zip(*one_batch)))        

class Sampler_time(object):
    def __init__(self, data, n_items, n_users, batch_size=128, max_len=20, neg_size=10, n_workers=10, neg_method='rand'):
        self.result_queue = Queue(maxsize=int(512))
        self.processors = []
        for i in range(n_workers):
            self.processors.append(
                Process(target=sample_function_time, args=(data,
                                                    n_items, 
                                                    n_users,
                                                    batch_size, 
                                                    max_len, 
                                                    neg_size, 
                                                    self.result_queue, 
                                                    np.random.randint(2e9),
                                                    neg_method)))
            self.processors[-1].daemon = True
            self.processors[-1].start()
    
    def next_batch(self):
        return self.result_queue.get()

    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()


