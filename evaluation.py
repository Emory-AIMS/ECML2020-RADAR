import numpy as np
import scipy
from utilities import cal_normal_pdf, cal_l1, cal_l2, cal_l_inf, load_data, KL
import tensorflow as tf
import argparse

def main(diff_path_neg, diff_path_pos, adv_path_neg, adv_path_pos, ind_neg_path, ind_pos_path):
    diff_neg = np.load(diff_path_neg)
    neg_ind = np.load(ind_neg_path)
    l1_norm_neg = cal_l1(diff_neg[neg_ind])
    l2_norm_neg = cal_l2(diff_neg[neg_ind])
    l_inf_neg = cal_l_inf(diff_neg[neg_ind])
    diff_pos = np.load(diff_path_pos)
    pos_ind = np.load(ind_pos_path)
    l1_norm_pos = cal_l1(diff_pos[pos_ind])
    l2_norm_pos = cal_l2(diff_pos[pos_ind])
    l_inf_pos = cal_l_inf(diff_pos[pos_ind])
    print neg_ind.shape[0], "negative adversarial sample have been made"
    print pos_ind.shape[0], "positive adversarial sample have been made"
    print "l1 norm of negative sample is:", l1_norm_neg
    print "l_inf norm of negative sample is:", l_inf_neg
    print "l_2 norm of negative sample is:", l2_norm_neg

    print "l1 norm of positive sample is:", l1_norm_pos
    print "l_inf norm of positive sample is:", l_inf_pos
    print "l_2 norm of positive sample is:", l2_norm_pos

    X_pos, Y_pos, X_neg, Y_neg = load_data()
    neg_cor_index = np.load('./neg_cor_index.npy')
    pos_cor_index = np.load('./pos_cor_index.npy')
    X_pos = X_pos[pos_cor_index]
    X_neg = X_neg[neg_cor_index]

    xadv_pos = np.load(adv_path_pos)[pos_ind]
    pdf_pos = cal_normal_pdf(X_pos, xadv_pos)
    
    stand_pdf = cal_normal_pdf(X_pos, X_pos)
    print "Gaussian Observation: pdf mean of positive sample is ", pdf_pos.mean()
    a = pdf_pos-stand_pdf.mean()>0
    print "Gaussian Observation: prob that pdf of positive sample is higher than standard pdf mean is ", a.mean()

    xadv_neg = np.load(adv_path_neg)[neg_ind]
    pdf_neg = cal_normal_pdf(X_neg, xadv_neg)
 
    stand_pdf  = cal_normal_pdf(X_neg, X_neg)
    print "Gaussian Observation: pdf mean of negative sample is ", pdf_neg.mean()
    a = pdf_neg-stand_pdf.mean()>0
    print "Gaussian Observation: prob that pdf of negative sample is higher than standard pdf mean is ", a.mean()
    sess = tf.Session()
    print "KL Divergense of positive sample is :",sess.run(KL(X_pos, xadv_pos))
    print "KL Divergense of negative sample is :", sess.run(KL(X_neg, xadv_neg))
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--date",
                        default=None,
                        type=str,
                        required=True)
    args = parser.parse_args()  


    main("./adv_diff/neg_diff_list_{}.npy".format(args.date), "./adv_diff/pos_diff_list_{}.npy".format(args.date),  "./adv_diff/neg_x_adv_list_{}.npy".format(args.date), "./adv_diff/pos_x_adv_list_{}.npy".format(args.date),  "./adv_diff/neg_index_list_{}.npy".format(args.date), "./adv_diff/pos_index_list_{}.npy".format(args.date))
