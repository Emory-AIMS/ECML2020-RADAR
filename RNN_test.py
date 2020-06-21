import pickle
import tensorflow as tf
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import argparse
from tensorflow.python.framework.ops import reset_default_graph
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, \
                            f1_score, precision_score, recall_score
from sklearn.model_selection import KFold
from utilities import onehot, load_data, cal_auc_f1
from RNN_train import model


NUM_UNITS = 128
NUM_OUTPUTS = 2
MAX_TIMESTEP = 48
FEATURE_DIM = 19
FC_UNITS = 32
LEARNING_RATE = 0.05
BATCH_SIZE = 64

def test(X, Y, select_flag):
    dict_model = model(MAX_TIMESTEP, FEATURE_DIM, NUM_OUTPUTS, NUM_UNITS, FC_UNITS)
    sess = tf.Session()
    saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='model'))
    saver.restore(sess, "./models/model_0208_fold2/model")
    feed_dict = {dict_model['x_pl'] : X, dict_model['y_pl']: Y}
    y_true_test, y_pred_test, y_pre_test = sess.run([dict_model['y_true'], dict_model['y_pred'], dict_model['y_pre']], feed_dict=feed_dict)
    #acc = accuracy_score(y_pred_test, y_true_test)
    #auc, f1, precision, recall = cal_auc_f1(y_true_test_list, y_pred_test_list)

    #return acc, auc, f1, precision, recall
    return y_pred_test, y_true_test, y_pre_test

#X_pos, Y_pos, X_neg, Y_neg = load_data()
#Y_true = np.concatenate([Y_pos, Y_neg],0)
#X = np.concatenate([X_pos, X_neg],0)
#true_label = np.load("./vae_output/adv_passed_label_0424.npy")

parser = argparse.ArgumentParser()
parser.add_argument("--date",
                    default=None,
                    type=str,
                    required=True)
args = parser.parse_args()
true_label = onehot(np.concatenate([[1]*150,[0]*1000]),2)
ind = np.load("./vae_output/T90/adv_passed_index_{}.npy".format(args.date))
unpassed_ind = list(set(range(1150))-set(ind))
x_reformed =np.load("./vae_output/adv_out_{}.npy".format(args.date))
x_adv_neg = np.load("./adv_diff/neg_x_adv_list_{}.npy".format(args.date))
x_adv_pos = np.load("./adv_diff/pos_x_adv_list_{}.npy".format(args.date))
x_adv = np.concatenate([x_adv_pos, x_adv_neg], 0)
"""y_pos = np.load("./adv_diff/pos_y_adv_list_{}.npy".format(args.date))
y_neg = np.load("./adv_diff/neg_y_adv_list_{}.npy".format(args.date))
true_label = np.concatenate([y_pos, y_neg], 0)
true_label = np.argmax(true_label,1)"""



##no defense
y_pred, y_true, y_pre = test(x_adv, true_label, False)
acc = accuracy_score(y_pred, y_true)
import pdb; pdb.set_trace()
auc, f1, precision, recall = cal_auc_f1(true_label, y_pre)
##print(acc, auc, f1, precision, recall)

#detector
y_pred_passed, y_true_passed, y_pre_passed = test(x_adv[ind], true_label[ind], False)
y_pred, y_true, y_pre= test(x_adv[unpassed_ind], true_label[unpassed_ind], False)
y_pred = np.array([1-i for i in y_pred])
y_pre = np.array([1-i for i in y_pre])
acc_d = accuracy_score(np.concatenate([y_pred_passed, y_pred]), np.concatenate([y_true_passed, y_true]))
auc_d, f1_d, precision_d, recall_d = cal_auc_f1(np.concatenate([onehot(y_true_passed,2), onehot(y_true,2)]), np.concatenate([y_pre_passed, y_pre]))
#print(acc_d, auc_d, f1_d, precision_d, recall_d)

#reformer
y_pred,y_true,y_pre = test(x_reformed, true_label, False)
acc_r = accuracy_score(y_pred, y_true)
auc_r, f1_r, precision_r, recall_r = cal_auc_f1(true_label, y_pre)
#print(acc_r, auc_r, f1_r, precision_r, recall_r)


#reformer+detector
y_pred_passed, y_true_passed, y_pre_passed = test(x_reformed[ind], true_label[ind], False)
y_pred, y_true, y_pre = test(x_adv[unpassed_ind], true_label[unpassed_ind], False)
y_pred = np.array([1-i for i in y_pred])
y_pre = np.array([1-i for i in y_pre])
acc_rd = accuracy_score(np.concatenate([y_pred_passed, y_pred]), np.concatenate([y_true_passed, y_true]))
auc_rd, f1_rd, precision_rd, recall_rd = cal_auc_f1(np.concatenate([onehot(y_true_passed,2), onehot(y_true,2)]), np.concatenate([y_pre_passed, y_pre]))

print(acc, auc, f1, precision, recall)
print(acc_d, auc_d, f1_d, precision_d, recall_d)
print(acc_r, auc_r, f1_r, precision_r, recall_r)
print(acc_rd, auc_rd, f1_rd, precision_rd, recall_rd)
