import pickle
import tensorflow as tf
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.framework.ops import reset_default_graph
from sklearn.model_selection import train_test_split
from utilities import load_data, onehot, cal_l1, cal_l2, cal_l_inf
import VAE_RNN as VAE_RNN
from tpf_embedder.embedder import EMBEDDER
# This file is trying to train the detector AE
#save: train_plot, pos_test, neg_test

BATCH_SIZE = 64
MAX_TIMESTEP = 48
FEATURE_DIM = 19
TEST_SIZE = 7504
POS_SIZE = 4137
NEG_SIZE = 33376
def set_graph(args, embed_flag = False):
    reset_default_graph()
    vae = VAE_RNN.VAE(ENCODER_NUM_UNITS_l1 = 32,
                      ENCODER_NUM_UNITS_l2 = 64,
                      ENCODER_FC_UNITS_l1 = 16,
                      ENCODER_FC_UNITS_l2 = 32,
                      ATTENUNITS = 64)

    x_pl = tf.placeholder(tf.float32, shape=[None, MAX_TIMESTEP, FEATURE_DIM], name='X_input')
    batch_pl = tf.placeholder(tf.int32, shape=[])
    #decoder_input = tf.placeholder(tf.float32, shape=[batch_pl, MAX_TIMESTEP, FEATURE_DIM])
    
    with tf.variable_scope('vae'):
        if embed_flag:
            z1, z2, memory = vae.encoder_stacked_attention_emb(x_pl, batch_pl)   
            out = vae.decoder_stacked_attention_emb(memory, z1, z2, batch_pl)
        else:
            #z1, z2, memory = vae.encoder_stacked_attention(x_pl, batch_pl)   
            #out = vae.decoder_stacked_attention(memory, z1, z2, batch_pl)
            z = vae.encoder(x_pl)
            out = vae.decoder(z, None)
    with tf.variable_scope('loss'):
        loss1 = vae.l2_loss(x_pl, out)
        loss2 = vae.reg_loss()
        loss3 = vae.l1_loss(x_pl, out)
        loss = args.beta*loss1+loss2+args.alpha*loss3
    with tf.variable_scope('train'):
        optimizer = tf.train.AdamOptimizer(args.lr)
        train_op = optimizer.minimize(loss)
    return x_pl, loss1, loss2, loss3, loss, train_op, out, batch_pl

def get_batch(size, data):
    sample_len = data.shape[0]
    index = np.arange(sample_len)
    np.random.shuffle(index)
    train_idx = index[0:size]
    batch_data = data[train_idx]

    return batch_data

def train(args):
    x_pl, loss1, loss2, loss3, loss, train_op, out, batch_pl = set_graph(args, True)
    X_pos, Y_pos, X_neg, Y_neg = load_data()
    X_pos_train, X_pos_test= train_test_split(X_pos, test_size = 0.2)
    X_neg_train, X_neg_test= train_test_split(X_neg, test_size = 0.2)
    np.save("./vae_output/neg_clean_{}.npy".format(args.date), X_neg_test)
    np.save("./vae_output/pos_clean_{}.npy".format(args.date), X_pos_test)
    print ("Saved test data")
    sess = tf.Session()
    init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init)
    print ("model init")
    loss1_list_train = []
    loss2_list_train = []
    loss3_list_train = []
    loss_list_train = []
    loss1_list_test = []
    loss2_list_test = []
    loss3_list_test = []
    loss_list_test = []
    itr = 0
    #TRAIN_SAMPLE = X_neg_train.shape[0]
    min_loss1 = float("inf")
    min_loss2 = float("inf")
    min_loss3 = float("inf")
    min_loss = float("inf")
    count = 0
    while( itr<args.epochs ):
        sample_processed = 0
        while(sample_processed < NEG_SIZE):
            pos_batch = get_batch(BATCH_SIZE, X_pos_train)
            neg_batch = get_batch(BATCH_SIZE/2, X_neg_train)
                
            train_data = np.concatenate((pos_batch, neg_batch),0)
    
            feed_dict = {x_pl: train_data, batch_pl: train_data.shape[0]}
            fetch = [loss1, loss2, loss3, loss, train_op]
            loss1_out, loss2_out, loss3_out , loss_out, _ = sess.run(fetches = fetch, feed_dict = feed_dict)
            
            sample_processed += BATCH_SIZE
    
        if(itr%10==0):
            loss1_list_train.append(loss1_out)
            loss2_list_train.append(loss2_out)
            loss_list_train.append(loss_out) 
            test_data = np.concatenate((X_pos_test, X_neg_test),0)
            feed_dict_test = {x_pl: test_data, batch_pl: test_data.shape[0]}
        
            fetch_test = [loss1, loss2, loss3, loss]
            loss1_out_test, loss2_out_test, loss3_out_test, loss_out_test = sess.run(fetches = fetch_test, feed_dict = feed_dict_test)
            loss1_list_test.append(loss1_out_test)
            loss2_list_test.append(loss2_out_test)
            loss3_list_test.append(loss3_out_test)
            loss_list_test.append(loss_out_test)   
            if loss1_out<min_loss1 :
                saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='vae'))
    
                fn =  './vae_models' + '/model_{}'.format(args.date)+'/model'   
                saver.save(sess, fn)
                print('\n Saving model')

                min_loss1 = loss1_out
                min_loss2 = loss2_out
                min_loss3 = loss3_out
                min_loss = loss_out
                count = 0
            else:
                count += 1     
            if count == 10:
                print ("Early stopped")
                print ("train_loss:", min_loss1, min_loss2, min_loss3, min_loss)
                print("train_loss:", loss1_out, loss2_out, loss3_out, loss_out, "test_loss:", loss1_out_test, loss2_out_test, loss3_out_test, loss_out_test)
                break
            else:
                print("train_loss:", loss1_out, loss2_out, loss3_out, loss_out, "test_loss:", loss1_out_test, loss2_out_test, loss3_out_test, loss_out_test)
        itr+=1
      
    return loss1_list_train, loss2_list_train, loss3_list_train, loss_list_train, loss1_list_test, loss2_list_test, loss3_list_test, loss_list_test
  

def main(args):
    loss1_list_train, loss2_list_train, loss3_list_train, loss_list_train, \
    loss1_list_test, loss2_list_test, loss3_list_test, loss_list_test = train(args)
    return loss1_list_train, loss2_list_train, loss3_list_train, loss_list_train, loss1_list_test,loss2_list_test, loss3_list_test,  loss_list_test
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--date",
                        default=None,
                        type=str,
                        required=True)
    parser.add_argument("--epochs",
                        default=1000,
                        type=int)
    parser.add_argument('--alpha',
                        type=float,
                        default=1)
    parser.add_argument('--beta',
                        type=float,
                        default=1)
    parser.add_argument('--lr',
                        type=float,
                        default=1e-4)
    args = parser.parse_args()
    loss1_list_train, loss2_list_train, loss3_list_train, loss_list_train, loss1_list_test,loss2_list_test, loss3_list_test,  loss_list_test = main(args)
    np.savez("./vae_output/plotSources_{}_pos.npy".format(date), loss1_list_train, loss2_list_train, loss3_list_train, loss_list_train, loss1_list_test,loss2_list_test, loss3_list_test,  loss_list_test)



