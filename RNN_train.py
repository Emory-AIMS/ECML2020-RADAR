import pickle
import tensorflow as tf
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.framework.ops import reset_default_graph
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix, \
                            f1_score, precision_score, recall_score
from sklearn.model_selection import KFold


TRAIN_SAMPLE = 26700
NUM_UNITS = 128
NUM_OUTPUTS = 2
MAX_TIMESTEP = 48
FEATURE_DIM = 19
FC_UNITS = 32
LEARNING_RATE = 0.05
BATCH_SIZE = 64


def onehot(t, num_classes):
    out = np.zeros((t.shape[0], num_classes))
    for row, col in enumerate(t):
        out[row, col] = 1
    return out

def make_data(X_pos, Y_pos, X_neg, Y_neg, n):
    x_train_pos = []
    y_train_pos = []
    x_test_pos = []
    y_test_pos = []

    k_fold = KFold(n_splits=n)
    for train_indices, test_indices in k_fold.split(X_pos):
        #print('Train: %s | test: %s' % (train_indices, test_indices))
        x_train_pos.append(X_pos[train_indices])
        y_train_pos.append(Y_pos[train_indices])

        x_test_pos.append(X_pos[test_indices])
        y_test_pos.append(Y_pos[test_indices])

    x_train_neg = []
    y_train_neg = []
    x_test_neg = []
    y_test_neg = []

    k_fold = KFold(n_splits=n)
    for train_indices, test_indices in k_fold.split(X_neg):
        #print('Train: %s | test: %s' % (train_indices, test_indices))
        x_train_neg.append(X_neg[train_indices])
        y_train_neg.append(Y_neg[train_indices])

        x_test_neg.append(X_neg[test_indices])
        y_test_neg.append(Y_neg[test_indices])   
    
    return x_train_pos, y_train_pos, x_test_pos, y_test_pos, x_train_neg, y_train_neg, x_test_neg, y_test_neg


def get_batch(batch_size, data, labels):
    sample_len = data.shape[0]
    index = np.arange(sample_len)
    np.random.shuffle(index)
    train_idx = index[0:batch_size]
    train_data = data[train_idx]
    train_label = labels[train_idx]
    
    return train_data, onehot(train_label,2)

def cal_auc_f1(y_true, y_pred):
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred )
    recall = recall_score(y_true, y_pred)
    return auc, f1, precision, recall

def model(MAX_TIMESTEP, FEATURE_DIM, NUM_OUTPUTS, NUM_UNITS, FC_UNITS):
    reset_default_graph()
    dict_model = {}
    with tf.variable_scope('model'):
        x_pl = tf.placeholder(tf.float32, shape=[None, MAX_TIMESTEP, FEATURE_DIM], name='X_input')
        #x_len = tf.placeholder(tf.float32, shape=[None], name='X_len')
        y_pl = tf.placeholder(tf.float32, shape=[None,NUM_OUTPUTS], name='Y_input')


        initializer = tf.random_normal_initializer(stddev=0.1)

        with tf.variable_scope('rnn', initializer = initializer):
            cell = tf.nn.rnn_cell.LSTMCell(NUM_UNITS)
            output, out_state = tf.nn.dynamic_rnn(cell = cell, inputs = x_pl, \
                                                   dtype=tf.float32)###must be float
            output = tf.transpose(output, [1,0,2])

        with tf.variable_scope('fc', initializer = initializer):
            _, output_last_time = tf.split(output, [MAX_TIMESTEP-1 ,1], 0)
            z = tf.layers.dense(output_last_time[-1], units=FC_UNITS, activation=tf.nn.relu)

        with tf.variable_scope('softmax', initializer = initializer):
            logits = tf.layers.dense(z, units=NUM_OUTPUTS, name='logits')

        y_pre = tf.nn.softmax(logits, name='y_pre')

    y_pred = tf.argmax(y_pre, 1)
    y_true = tf.argmax(y_pl, 1)
    with tf.variable_scope('loss'):
        reg_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, 'model')
        xent = tf.nn.softmax_cross_entropy_with_logits(labels=y_pl, logits=logits)
        loss = tf.reduce_mean(xent) + tf.reduce_sum(reg_loss)
    with tf.variable_scope('train'):
        optimizer = tf.train.AdamOptimizer(LEARNING_RATE)
        train_op = optimizer.minimize(loss)
    with tf.variable_scope('acc'):
        count = tf.equal(tf.argmax(y_pl, axis=1), tf.argmax(y_pre, axis=1))
        acc = tf.reduce_mean(tf.cast(count, tf.float32), name='acc')
    dict_model['x_pl'] = x_pl
    dict_model['y_pl'] = y_pl
    dict_model['y_pred'] = y_pred
    dict_model['y_pre'] = y_pre
    dict_model['y_true'] = y_true
    dict_model['loss'] = loss
    dict_model['acc'] = acc
    dict_model['train_op'] = train_op
    dict_model['logits'] = logits

    return dict_model

def train(batch_size, num_iter, x_train_pos, y_train_pos, x_train_neg, y_train_neg,\
          x_val_pos, y_val_pos, x_val_neg, y_val_neg, fn):

    ITER = num_iter
    itr = 0
    loss_train_plt = []
    acc_train_plt = []
    auc_train_plt = []
    f1_train_plt = []
    precision_train_plt = []
    recall_train_plt = []

    loss_val_plt = []
    acc_val_plt =[]
    auc_val_plt = []
    f1_val_plt = []
    precision_val_plt = []
    recall_val_plt = []

    dict_model = model(MAX_TIMESTEP, FEATURE_DIM, NUM_OUTPUTS, NUM_UNITS, FC_UNITS)

    sess = tf.Session()
    init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init)

    while(itr<ITER):
        sample_processed = 0
        ep = 1.0
        train_loss_iter = 0.0
        train_acc_iter= 0.0
        auc_score_iter= 0.0
        f1_score_iter= 0.0
        precresion_score_iter= 0.0
        recall_score_iter= 0.0

        while(sample_processed < TRAIN_SAMPLE):
            ep += 1
            
            train_data_neg, train_label_neg = get_batch(BATCH_SIZE, x_train_neg, y_train_neg)

            train_data_pos, train_label_pos = get_batch(BATCH_SIZE/2, x_train_pos, y_train_pos)
     
            train_data = np.concatenate((train_data_neg, train_data_pos),0)
            train_label = np.concatenate((train_label_neg, train_label_pos),0)
           
            
            feed_dict_train = {dict_model['x_pl'] : train_data,\
                               dict_model['y_pl']: train_label}
            fetch_train = [dict_model['train_op'], dict_model['loss'], \
                           dict_model['acc'], dict_model['y_true'], dict_model['y_pred']]
            _, train_loss, train_accuracy, y_true_train, y_pred_train = sess.run(fetches=fetch_train, feed_dict=feed_dict_train)
        
            train_auc, train_f1, train_precision, train_recall = cal_auc_f1(y_true_train, y_pred_train)
            train_loss_iter += train_loss
            train_acc_iter += train_accuracy
            auc_score_iter += train_auc
            f1_score_iter += train_f1
            precresion_score_iter += train_precision
            recall_score_iter += train_recall

            sample_processed += BATCH_SIZE
        loss_train_plt.append(round(train_loss_iter/ep,3))
        acc_train_plt.append(round(train_acc_iter/ep,3))
        auc_train_plt.append(round(auc_score_iter/ep,3))
        f1_train_plt.append(round(f1_score_iter/ep,3))   
        precision_train_plt.append(round(precresion_score_iter/ep,3))
        recall_train_plt.append(round(recall_score_iter/ep,3))

        val_data = np.concatenate((x_val_neg, x_val_pos),0)
        val_label = np.concatenate((y_val_neg, y_val_pos),0)
        val_label = onehot(val_label, 2)
        feed_dict_val = {dict_model['x_pl']  : val_data, dict_model['y_pl']: val_label}
        fetch_val = [dict_model['loss'], dict_model['acc'], dict_model['y_true'], dict_model['y_pred']]
        val_loss, val_accuracy, y_true_val, y_pred_val  = sess.run(fetches=fetch_val, feed_dict=feed_dict_val)
        val_auc, val_f1 , val_precision, val_recall = cal_auc_f1(y_true_val, y_pred_val)
        loss_val_plt.append(round(val_loss,3))
        acc_val_plt.append(round(val_accuracy,3))
        auc_val_plt.append(round(val_auc,3))
        f1_val_plt.append(round(val_f1,3))
        precision_val_plt.append(round(val_precision,3))
        recall_val_plt.append(round(val_recall,3))

        if(itr%5==0):
            print ("iter:", itr)
            print ('====================================')
            print ("val loss is: %3.3f" %val_loss)
            print ("val accuracy: %3.3f" %val_accuracy)
            print ("val auc is: %3.3f" %val_auc)
            print ("val f1 is: %3.3f" %val_f1)
            print ("val precission is: %3.3f" %val_precision)
            print ("val recall is: %3.3f" %val_recall)
            print ('====================================')
            print ("train loss is: %3.3f" %loss_train_plt[-1])
            print ("train accuracy: %3.3f" %acc_train_plt[-1])
            print ("train auc is: %3.3f" %auc_train_plt[-1])
            print ("train f1: %3.3f" %f1_train_plt[-1])
            print ("train precision is: %3.3f" %precision_train_plt[-1])
            print ("train recall: %3.3f" %recall_train_plt[-1])

        itr+=1 
    saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='model'))
    print('\n Saving model')

    saver.save(sess, fn)
    return loss_train_plt, acc_train_plt, auc_train_plt, f1_train_plt, precision_train_plt, recall_train_plt, \
            loss_val_plt, acc_val_plt, auc_val_plt, f1_val_plt, precision_val_plt, recall_val_plt
    

    
def train_plot(a,b,c,d,e,f,g,h,i,j,k,l, plt_name):
    plt.figure(figsize=(12,12))

    plt.subplots_adjust(wspace =0.2, hspace =0.2)
    plt.subplot(2,3,1)
    plt.plot(a, 'r', label = 'train_loss')
    plt.plot(g, 'g', label = 'vsl_loss')
    plt.title('loss')
    plt.legend(loc=1, ncol=2, fontsize = 'small')

    plt.subplot(2,3,2)
    plt.plot(b, 'r', label = 'train_acc')
    plt.plot(h, 'g', label = 'val_acc')
    plt.title('acc')
    plt.legend(loc=2, ncol=6,fontsize = 'small')

    plt.subplot(2,3,3)
    plt.plot(c, 'r', label = 'train_auc')
    plt.plot(i, 'g', label = 'val_auc')
    plt.title('auc')
    plt.legend(loc=2, ncol=2,fontsize = 'small')

    plt.subplot(2,3,4)
    plt.plot(d,'r', label = 'train_f1')
    plt.plot(j, 'g', label = 'val_f1')
    plt.title('f1')
    plt.legend(loc=2, ncol=2,fontsize = 'small')

    plt.subplot(2,3,5)
    plt.plot(e,'r', label = 'train_precision')
    plt.plot(k, 'g', label = 'val_precision')
    plt.title('precision')
    plt.legend(loc=2, ncol=2,fontsize = 'small')

    plt.subplot(2,3,6)
    plt.plot(f,'r', label = 'train_recall')
    plt.plot(l, 'g', label = 'val_recall')
    plt.title('recall')
    plt.legend(loc=2, ncol=2,fontsize = 'small')
    #plt.show()
    plt.savefig(plt_name)

def main(path, name):
    path = path
    fn = open(path+'/DATA_M_PAD_NORM.pkl', 'rb')
    D = pickle.load(fn)

    X = D[0]
    Y = D[1]
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]

    pos_index = np.nonzero(Y)[0]
    neg_index = np.nonzero(Y==0)[0]
    X_pos, Y_pos = X[pos_index], Y[pos_index]
    X_neg, Y_neg = X[neg_index], Y[neg_index]

    x_train_pos, y_train_pos, x_val_pos, y_val_pos, x_train_neg, y_train_neg,\
    x_val_neg, y_val_neg = make_data(X_pos, Y_pos, X_neg, Y_neg, 10)


    n_fold = 0
    validation_auc = []
    validation_f1 = []
    validation_pre = []
    validation_rec = []
    while(n_fold<5):
        print ("==================fold:", n_fold, "============================")
        fn =  './models' + '/{}_fold{}'.format(name, n_fold)+'/model'
        lt,acct,auct,ft,pt, rt, lv,accv, aucv,fv,pv,rv = train(BATCH_SIZE, 25, x_train_pos[n_fold], y_train_pos[n_fold], x_train_neg[n_fold], y_train_neg[n_fold],\
                            x_val_pos[n_fold], y_val_pos[n_fold], x_val_neg[n_fold], y_val_neg[n_fold], fn)
        plt_name =  './fold' + '{}'
        train_plot(lt,acct,auct,ft,pt, rt, lv,accv, aucv,fv,pv,rv,plt_name.format(n_fold))
        validation_auc.append(aucv[-1])
        validation_f1.append(fv[-1])
        validation_pre.append(pv[-1])
        validation_rec.append(rv[-1])
        n_fold +=1
    return validation_auc, validation_f1, validation_pre, validation_rec

if __name__ == '__main__':
    name = 'model_0208'
    path =  "/Users/wendy/Documents/XiongLab/Data/processed-mimic3-mortality"
    validation_auc, validation_f1, validation_pre, validation_rec = main(path, name)
    print ("val_auc is:", validation_auc)
    print ("val_f1 is:", validation_f1)
    print ("val_pre is:", validation_pre)
    print ("val_rec is:", validation_rec)