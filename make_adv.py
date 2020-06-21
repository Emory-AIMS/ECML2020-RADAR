import tensorflow as tf
import pickle
import numpy as np
from tensorflow.python.tools import inspect_checkpoint as chkp
from utilities import onehot, load_data, Gaussian_Observation, cal_l1, get_batch
import argparse


NUM_UNITS = 128
NUM_OUTPUTS = 2
MAX_TIMESTEP = 48
FEATURE_DIM = 19
FC_UNITS = 32
LEARNING_RATE = 0.001
BATCH_SIZE = 64

def model(x_pl):

    with tf.variable_scope('model'):
        with tf.variable_scope('rnn'):
            cell = tf.nn.rnn_cell.LSTMCell(NUM_UNITS)
            output, out_state = tf.nn.dynamic_rnn(cell = cell, inputs = x_pl, \
                                                   dtype=tf.float32)###must be float
            output = tf.transpose(output, [1,0,2])

        with tf.variable_scope('fc'):
            _, output_last_time = tf.split(output, [MAX_TIMESTEP-1 ,1], 0)
            z = tf.layers.dense(output_last_time[-1], units=FC_UNITS, activation=tf.nn.relu)

        with tf.variable_scope('softmax'):
            logits = tf.layers.dense(z, units=NUM_OUTPUTS, name='logits')
        y_pre = tf.nn.softmax(logits, name='y_pre')

    return y_pre, logits#####both dim[n,2]

def load_ckpt(path):
    tf.train.init_from_checkpoint(path, {'model/rnn/': './rnn'})
    tf.train.init_from_checkpoint(path, {'model/fc/': './fc'})
    tf.train.init_from_checkpoint(path, {'model/softmax/': './softmax'})


def cw(model, x, y_true, alpha, beta, lamb=0.0, optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.005), min_prob=0, mean_flag = True):

    xshape = x.get_shape().as_list()
    noise = tf.get_variable('noise', xshape, tf.float32, initializer=tf.initializers.zeros)

    noise_init_op = tf.assign(noise, tf.constant(0.0, shape = xshape))
    # ISTA
    cond1 = tf.cast(tf.greater(noise, lamb), tf.float32)
    cond2 = tf.cast(tf.less_equal(tf.abs(noise), lamb), tf.float32)
    cond3 = tf.cast(tf.less(noise, tf.negative(lamb)), tf.float32)

    assign_noise = tf.multiply(cond1,tf.subtract(noise,lamb)) + tf.multiply(cond2, noise) \
                + tf.multiply(cond3, tf.add(noise,lamb))
    setter = tf.assign(noise, assign_noise)

    # Adversarial
    xadv = x + noise
    ybar, logits = model(xadv)

    ydim = ybar.get_shape().as_list()[1]
    y = tf.argmin(ybar, axis=1, output_type=tf.int32)

    mask = tf.one_hot(y, ydim, on_value=0.0, off_value=float('inf'))
    yt = tf.reduce_max(logits - mask, axis=1)
    yo = tf.reduce_max(y_true, axis=1)####y_true has dim[n, 2], onehot to [n,1]
    loss0 = tf.nn.relu(yo - yt + min_prob)

    axis = list(range(1, len(xshape)))
    if mean_flag:
        loss1 = alpha*tf.reduce_mean((tf.abs(xadv - x)), axis=axis) ## reduce max(l_inf) or reduce mean(l_1)
    else:
        loss1 = alpha*tf.reduce_max((tf.abs(xadv - x)), axis=axis)
    loss2 = beta*Gaussian_Observation(xadv, x)
    #alpha = (loss0/loss1)//100 *100
    loss = loss0 + loss1 - loss2
    train_op = optimizer.minimize(loss, var_list=[noise])

    return train_op, ybar, xadv, noise, noise_init_op, setter, loss0, loss1, loss2, loss


def make_cw(x_clean, y_true, path, epochs, flag, alpha, beta, mean_flag = None, t = None):
    tf.reset_default_graph()
    x_pl = tf.placeholder(tf.float32, shape=[1, MAX_TIMESTEP, FEATURE_DIM], name='X_input')
    y_pl = tf.placeholder(tf.float32, shape=[1,NUM_OUTPUTS], name='Y_input')
    #y_pre, logits = model(x_pl)
    adv_train_op, ybar, xadv, noise, noise_init_op, setter, loss0, loss1, loss2, loss = cw(model, x_pl, y_pl, alpha = alpha, beta = beta, mean_flag = mean_flag)
    
    sess = tf.Session()
    init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    # init_op1 = tf.variables_initializer([noise])
    sess.run(init)
    saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='model'))
    saver.restore(sess, path)

    print ("===load from ckpt===")
    """    varss = tf.all_variables()
    vars_vals = sess.run(varss)
    for var, val in zip(varss, vars_vals):
        print("var: {}, value: {}".format(var.name, val))"""
    

    x_adv_list = np.empty_like(x_clean)
    diff_list = np.empty_like(x_clean)
    y_pre_list = []
    succes_index = []
    for i in range(x_clean.shape[0]):
        epoch = 0
        
        x_data = x_clean[i:i+1]
        y_data = y_true[i:i+1]
        feed_dict = {x_pl: x_data, y_pl: y_data}
        sess.run(noise_init_op)
        for epoch in range(epochs):
            # print('Epoch:', epoch)
            sess.run(adv_train_op, feed_dict=feed_dict)
            sess.run(setter, feed_dict=feed_dict)
            x_adv, y_pred, loss_0, loss_1, loss_2, loss_out = sess.run([xadv, ybar, loss0, loss1,loss2, loss], feed_dict=feed_dict)
            #print y_pred
            """if(epoch%100==0):
                #print y_pred
                print ("label",y_data, y_pred, loss_0, loss_1, loss_2, loss_out)
                diff = x_adv - x_clean[i:i+1]
                print (np.sum(np.abs(diff)))"""
            if flag == True:
                if np.not_equal(np.argmax(y_pred,1), np.argmax(y_data,1)):
                    print('Classification changed at Epoch {}!'.format(epoch))
                    succes_index.append(i)
                    diff = x_adv - x_clean[i:i+1]
                    x_adv_list[i] = x_adv
                    diff_list[i] = diff
                    y_pre_list.append(y_pred)
                    #sess.run(noise_init_op)
                    break
                
            else:
                diff = x_adv - x_clean[i:i+1]
                if np.sum(np.abs(diff))>t:
                    print('Training ends at {}!'.format(epoch))
                    print ("label",y_data, y_pred, loss_0, loss_1, loss_2, loss_out)
                    print (np.sum(np.abs(diff)))
                    x_adv_list[i] = x_adv
                    diff_list[i] = diff
                    #sess.run(noise_init_op)
                    break
                if np.not_equal(np.argmax(y_pred,1), np.argmax(y_data,1)):
                    print('Classification changed at Epoch {}!'.format(epoch))
                    succes_index.append(i)
                    x_adv_list[i] = x_adv
                    diff_list[i] = diff
                    break
    succ_rate = len(succes_index)/x_clean.shape[0]
    print(succ_rate)

    sess.close()
    return diff_list, x_adv_list, y_pre_list,succes_index
    

def main(ckpt_path):
    parser = argparse.ArgumentParser()
    parser.add_argument("--date",
                        default=None,
                        type=str,
                        required=True)
    parser.add_argument("--epochs",
                        default=1000,
                        type=int)
    parser.add_argument('--flip_flag',
                        default = False,
                        type=bool)
    
    parser.add_argument('--mean_flag',
                        type=bool)

    parser.add_argument('--t', 
                        type=float, 
                        default=1)
    parser.add_argument('--alpha', 
                        type=float, 
                        default=1)
    parser.add_argument('--beta', 
                        type=float, 
                        default=1)                   

    args = parser.parse_args()  


    X_pos, Y_pos, X_neg, Y_neg = load_data()
    if args.flip_flag == True:
        print ("============start picking smaples with correct labels!=============")
        neg_cor_index = np.load('./neg_cor_index.npy')
        pos_cor_index = np.load('./pos_cor_index.npy')
        
        x_clean = X_pos[pos_cor_index]
        y_data = onehot(Y_pos[pos_cor_index],2)

        print (" ============make adversarial samples for %d positive samples=============" %x_clean.shape[0])
        diff_list, x_adv_list, y_pre_list, succes_index = make_cw(x_clean, y_data, ckpt_path, args.epochs, args.flip_flag, args.alpha, args.beta, args.mean_flag)
        np.save("./adv_diff/pos_diff_list_{}.npy".format(args.date), diff_list)
        np.save("./adv_diff/pos_x_adv_list_{}.npy".format(dargs.ate), x_adv_list)
        np.save("./adv_diff/pos_index_list_{}.npy".format(args.date), succes_index)


        x_clean = X_neg[neg_cor_index]
        y_data = onehot(Y_neg[neg_cor_index],2)
        print (y_data.shape)
        print (" ============make adversarial samples for %d negative samples=============" %x_clean.shape[0])
        diff_list, x_adv_list, y_pre_list, succes_index = make_cw(x_clean, y_data, ckpt_path, args.epochs,args.flip_flag,  args.alpha, args.beta, args.mean_flag)
        np.save("./adv_diff/neg_diff_list_{}.npy".format(args.date), diff_list)
        np.save("./adv_diff/neg_x_adv_list_{}.npy".format(args.date), x_adv_list)
        np.save("./adv_diff/neg_index_list_{}.npy".format(args.date), succes_index)

    if args.flip_flag == False:
        print ("============make adversarial samples for positive samples=============" )
        x_clean, y_data = get_batch(150, X_pos, Y_pos)
        diff_list, x_adv_list, _, _ = make_cw(x_clean, y_data, ckpt_path, args.epochs, args.flip_flag,  args.alpha, args.beta,  args.mean_flag, args.t,)
        np.save("./adv_diff/pos_diff_list_{}.npy".format(args.date), diff_list)
        np.save("./adv_diff/pos_x_adv_list_{}.npy".format(args.date), x_adv_list) 
        np.save("./adv_diff/pos_y_adv_list_{}.npy".format(args.date), y_data)

        x_clean, y_data = get_batch(1000, X_neg, Y_neg)
        print (" ============make adversarial samples for negative samples=============" )
        diff_list, x_adv_list, _, _ = make_cw(x_clean, y_data, ckpt_path, args.epochs, args.flip_flag,  args.alpha, args.beta,  args.mean_flag, args.t)
        np.save("./adv_diff/neg_diff_list_{}.npy".format(args.date), diff_list)
        np.save("./adv_diff/neg_x_adv_list_{}.npy".format(args.date), x_adv_list)
        np.save("./adv_diff/neg_y_adv_list_{}.npy".format(args.date), y_data)
        

if __name__ == '__main__':
    ckpt_path =  "./models/model_0208_fold2/model"
    main(ckpt_path)


