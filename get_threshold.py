import pickle
import tensorflow as tf
import numpy as np
import scipy
import dit
import argparse
import os
from tensorflow.python.framework.ops import reset_default_graph
from sklearn.model_selection import train_test_split
from dit.divergences import jensen_shannon_divergence
from utilities import load_data, onehot, cal_l1, cal_l2, cal_l_inf
import VAE_RNN as VAE_RNN
from VAE_train import set_graph
from RNN_train import model

NUM_UNITS = 128
NUM_OUTPUTS = 2
MAX_TIMESTEP = 48
FEATURE_DIM = 19
FC_UNITS = 32
LEARNING_RATE = 0.05
BATCH_SIZE = 64
ITER = 10

def cal_threshold(diff, logits_clean, ae_logits_clean, tile):
    ###l_inf threshold
    result = []
    for i in range(diff.shape[0]):
        dd = diff[i:i+1]
        result.append(cal_l_inf(dd))
    result = np.sort(result)
    t_inf = result[int(len(result)*tile)]
    ###l_2 threshold
    result = []
    for i in range(diff.shape[0]):
        dd = diff[i:i+1]
        result.append(cal_l2(dd))
    result = np.sort(result)
    t2 = result[int(len(result)*tile)]
    ###l_1 threshold
    result = []
    for i in range(diff.shape[0]):
        dd = diff[i:i+1]
        result.append(cal_l1(dd))
    result = np.sort(result)
    t1 = result[int(len(result)*tile)]
    ### JS_divergence between f(x) and f(ae(x)) as threshold
    result = []
    for i in range(diff.shape[0]):
        js_d = js_divergence(softmax(logits_clean[i]),softmax(ae_logits_clean[i]))
        result.append(js_d)
    result = np.sort(result)
    t_js = result[int(len(result)*tile)]
    ### entropy (uncertainty level ) as threshold
    result = []
    for i in range(diff.shape[0]):
        uncertainty = scipy.stats.entropy(softmax(logits_clean[i]))
        result.append(uncertainty)
    result = np.sort(result)
    t_uncertainty = result[int(len(result)*0.9)]

    return t1, t2, t_inf, t_js, t_uncertainty


def pass_rate(diff_adv, logits_adv, ae_logits_adv, t1, t2, t_inf, t_js, t_uncertainty, pos_num):
    succ_count = 0.0
    fail_jsd = 0
    fail_unc = 0
    fail_rec = 0
    indx_list = []
    y_adv = []
    for i in range(diff_adv.shape[0]): 
        l1 = cal_l1(diff_adv[i:i+1])
        l2 = cal_l2(diff_adv[i:i+1])
        l_inf = cal_l_inf(diff_adv[i:i+1])
        js_d = js_divergence(softmax(logits_adv[i]), softmax(ae_logits_adv[i]))
        unc = scipy.stats.entropy(softmax(logits_adv[i]))
        #print l1, l2, l_inf, js_d, unc
        if js_d>t_js or js_d==t_js:
            fail_jsd += 1.0
        if unc>t_uncertainty or unc==t_uncertainty:
            fail_unc += 1.0
        if l1>t1 or l2>t2 or l_inf>t_inf :
            fail_rec +=1.0
        if l1<t1 and l2<t2 and l_inf<t_inf and js_d<t_js and unc<t_uncertainty:
            succ_count+=1.0
    #return succ_count/diff_adv.shape[0], fail_jsd/diff_adv.shape[0], fail_unc/diff_adv.shape[0]
            indx_list.append(i)
            if(i<pos_num):
                y_adv.append(1)
            else:
                y_adv.append(0)


    return fail_unc, fail_jsd, fail_rec, succ_count, diff_adv.shape[0], indx_list, y_adv


def ae_test(dd, tt, date, x_pl, batch_pl, out,sess):
    out_data_list = []
    for i in range(dd.shape[0]):
        x_data = dd[i:i+1]
        feed_dict = {x_pl: x_data, batch_pl: 1}
        fetch = out
        out_data = sess.run(fetches = fetch, feed_dict = feed_dict)
        out_data_list.append(out_data)
    out_data_np = np.concatenate(out_data_list, axis = 0)
    out_data_np = np.squeeze(out_data_np)
    out_data_neg = np.flip(out_data_np,1)
    np.save("./vae_output/{}_out_{}.npy".format(tt, date), out_data_neg)
    diff = dd - out_data_neg
    np.save("./vae_output/{}_diff_{}.npy".format(tt, date), diff)
    print ("saved")
    return diff, out_data_neg

def softmax(x):
    ex = np.exp(x)
    sum_ex = np.sum( np.exp(x))
    return ex/sum_ex

def js_divergence(logits,ae_logits):
    a = dit.ScalarDistribution([0,1],logits)
    b = dit.ScalarDistribution([0,1],ae_logits)
    return jensen_shannon_divergence([a,b])

def main(model_path, pos_clean_path, neg_clean_path, pos_adv_path, neg_adv_path, date, tile):
    #### load AE model
    x_pos_clean = np.load(pos_clean_path)
    x_neg_clean = np.load(neg_clean_path)
    x_pos_adv = np.load(pos_adv_path)
    x_neg_adv = np.load(neg_adv_path)
    pos_num = x_pos_adv.shape[0]
    pos_num_clean = x_pos_clean.shape[0]
    x_clean = np.concatenate([x_pos_clean, x_neg_clean], 0)
    x_adv = np.concatenate([x_pos_adv, x_neg_adv], 0)
    x_pl, loss1, loss2, loss3, loss, train_op, out, batch_pl = set_graph()
    sess = tf.Session()
    init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='vae'))
    saver.restore(sess, model_path)
    if not os.path.exists("./vae_output/clean_diff_0417.npy"):

        diff_clean, ae_clean = ae_test(x_clean, "clean", "0417", x_pl, batch_pl, out, sess)
        
    else:
        diff_clean = np.load("./vae_output/clean_diff_0417.npy")
        ae_clean = np.load("./vae_output/clean_out_0417.npy")

    if not os.path.exists("./vae_output/adv_diff_{}.npy"):
        diff_adv, ae_adv = ae_test(x_adv, "adv",date, x_pl, batch_pl, out, sess)
    else:
        diff_adv = np.load("./vae_output/adv_diff_{}.npy".format(date))
        ae_adv = np.load("./vae_output/adv_out_{}.npy".format(date))

    sess.close()
    #load RNN model to calculate threshold

    dict_model = model(MAX_TIMESTEP, FEATURE_DIM, NUM_OUTPUTS, NUM_UNITS, FC_UNITS)
    sess = tf.Session()
    init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init)
    saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='model'))
    saver.restore(sess, "./models/model_0208_fold2/model")

    logits_clean = []
    for i in range(x_clean.shape[0]):
        feed_dict = {dict_model['x_pl'] : x_clean[i:i+1], dict_model['y_pl']: np.tile([0,1],(1,1))}#np.tile([0,1],(x_clean.shape[0],1))
        fetch = dict_model['logits']
        logits = sess.run(fetches=fetch, feed_dict=feed_dict)
        logits_clean.append(logits)
    logits_clean = np.concatenate(logits_clean, axis = 0)
    logits_clean = np.squeeze(logits_clean)
    ##
    ae_logits_clean = []
    for i in range(ae_clean.shape[0]):
        feed_dict = {dict_model['x_pl'] : ae_clean[i:i+1], dict_model['y_pl']: np.tile([0,1],(1,1))}
        fetch = dict_model['logits']
        ae_logits = sess.run(fetches=fetch, feed_dict=feed_dict)
        ae_logits_clean.append(ae_logits)
    ae_logits_clean = np.concatenate(ae_logits_clean, axis = 0)
    ae_logits_clean = np.squeeze(ae_logits_clean)
    ##

    logits_adv = []
    for i in range(x_adv.shape[0]):
        feed_dict = {dict_model['x_pl'] : x_adv[i:i+1], dict_model['y_pl']: np.tile([0,1],(1,1))}
        fetch = dict_model['logits']
        logits = sess.run(fetches=fetch, feed_dict=feed_dict)
        logits_adv.append(logits)
    logits_adv = np.concatenate(logits_adv, axis = 0)
    logits_adv = np.squeeze(logits_adv)
    import pdb; pdb.set_trace()
    ##
    ae_logits_adv = []
    for i in range(ae_adv.shape[0]):
        feed_dict = {dict_model['x_pl'] : ae_adv[i:i+1], dict_model['y_pl']: np.tile([0,1],(1,1))}
        fetch = dict_model['logits']
        ae_logits = sess.run(fetches=fetch, feed_dict=feed_dict)
        ae_logits_adv.append(ae_logits)
    ae_logits_adv = np.concatenate(ae_logits_adv, axis = 0)
    ae_logits_adv = np.squeeze(ae_logits_adv)
    pdb.set_trace()

    #calculate threshold
    t1, t2, t_inf, t_js, t_uncertainty = cal_threshold(diff_clean, logits_clean, ae_logits_clean, tile)
 
    fail_unc, fail_jsd, fail_rec, succ_count, dim, indx_list, y_adv = pass_rate(diff_adv, logits_adv, ae_logits_adv, t1, t2, t_inf, t_js, t_uncertainty, pos_num)
    _,_,_,_,_, indx_list_clean, y_adv_clean = pass_rate(diff_clean, logits_clean, ae_logits_clean, t1, t2, t_inf, t_js, t_uncertainty, pos_num_clean)

    print (fail_unc, fail_jsd, fail_rec, succ_count, dim)
    return indx_list, y_adv, indx_list_clean, y_adv_clean

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--date",
                        default=None,
                        type=str,
                        required=True)
    parser.add_argument("--tile",
                        default=0.9,
                        type=float,
                        required=True)
    args = parser.parse_args()
    model_path = "./vae_models/model_0417/model"
    pos_clean_path = "./vae_output/pos_clean_0417.npy"
    neg_clean_path = "./vae_output/neg_clean_0417.npy"
    pos_adv_path = "./adv_diff/pos_x_adv_list_{}.npy".format(args.date)
    neg_adv_path = "./adv_diff/neg_x_adv_list_{}.npy".format(args.date)
    passed_sample, y_adv, indx_list_clean, y_adv_clean = main(model_path, pos_clean_path, neg_clean_path, pos_clean_path, neg_clean_path, args.date, args.tile)
    np.save("./vae_output/T90/adv_passed_index_{}".format(args.date), passed_sample)
    np.save("./vae_output/T90/adv_passed_label_{}".format(args.date), y_adv)
    print ("saved")


