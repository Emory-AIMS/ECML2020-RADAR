import pickle
import numpy as np
import scipy.stats
import tensorflow as tf
from numpy import linalg as LA
from sklearn.metrics import roc_auc_score, confusion_matrix, \
                            f1_score, precision_score, recall_score

                            
def load_data():
    path = "/Users/wendy/Documents/XiongLab/Data/processed-mimic3-mortality"
    fn = open(path+'/DATA_M_PAD_NORM.pkl', 'rb')
    D = pickle.load(fn)
    X = D[0]
    Y = D[1]
    pos_index = np.nonzero(Y)[0]
    neg_index = np.nonzero(Y==0)[0]
    X_pos, Y_pos = X[pos_index], Y[pos_index]
    X_neg, Y_neg = X[neg_index], Y[neg_index]

    return X_pos, Y_pos, X_neg, Y_neg
def onehot(t, num_classes):
    out = np.zeros((t.shape[0], num_classes))
    for row, col in enumerate(t):
        out[row, col] = 1
    return out

def get_batch(batch_size, data, labels):
    sample_len = data.shape[0]
    index = np.arange(sample_len)
    np.random.shuffle(index)
    train_idx = index[0:batch_size]
    train_data = data[train_idx]
    train_label = labels[train_idx]

    
    return train_data, onehot(train_label,2)

def cal_auc_f1(y_true, y_pred):
    auc = roc_auc_score(y_true, y_pred)
    y_true = np.argmax(y_true,1)
    y_pred = np.argmax(y_pred, 1)
    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred )
    recall = recall_score(y_true, y_pred)
    return auc, f1, precision, recall

def cal_normal_pdf(x_clean, x_adv):
    shape = x_clean.shape
    x_clean = x_clean.reshape([shape[0],shape[1]*shape[2]])
    mean = np.mean(x_clean, axis = 0)
    #std = np.cov(x_clean, rowvar=False)
    std = np.std(x_clean, axis = 0)
    x_adv_shape = x_adv.shape
    x_adv = x_adv.reshape([x_adv_shape[0], x_adv_shape[1]*x_adv_shape[2]])
    
    pdf = scipy.stats.norm.pdf(x_adv, mean, std)
    return np.sum(np.log(pdf+0.0001), axis = 1)

def cal_l1(diff):
    mean_sig = 0
    for i in range(diff.shape[0]):
        mean_sig += LA.norm(diff[i].flatten(), ord = 1)
    return mean_sig/diff.shape[0] 
def cal_l2(diff):
    mean_sig = 0
    for i in range(diff.shape[0]):
        mean_sig += LA.norm(diff[i].flatten(), ord = 2)
    return mean_sig/diff.shape[0] 
    
def cal_l_inf(diff):
    max_sig = 0
    for i in range(diff.shape[0]):
        max_sig += LA.norm(diff[i].flatten(), ord = 2)
    return max_sig/diff.shape[0] 

def heatmap_plot(diff_path):
    import seaborn as sns
    diff = np.load(diff_path)
    m = np.mean(diff, axis = 0)

    sns.heatmap(m, annot=False, vmin=0, vmax=0.01, cmap="YlGnBu",linewidths=.5)

def KL(X1, X2):
    m1 = tf.reduce_mean(X1, axis = 0)
    std1 = tf.sqrt(tf.reduce_mean(tf.square(X1 - m1), axis = 0))
    m2 = tf.reduce_mean(X2, axis = 0)
    std2 = tf.sqrt(tf.reduce_mean(tf.square(X2 - m2), axis = 0))

    dist1 = tf.distributions.Normal(m1,std1)
    dist2 = tf.distributions.Normal(m2,std2)
    div = dist1.kl_divergence(dist2)
    
    return tf.reduce_sum(div)

def Gaussian_Observation(X_adv, X_clean):
    diff = X_adv - X_clean
    log_prob = -0.5 * (tf.log(2 * np.pi) + tf.square(diff))
    return tf.reduce_sum(log_prob)