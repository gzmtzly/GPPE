import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import LabelBinarizer

def comput_mAUC_mAP(y_ture, y_pred):
    label_binarizer = LabelBinarizer().fit(y_ture)
    y_ture_onehot = label_binarizer.transform(y_ture)
    AUC = []
    AP = []
    for i in range(y_ture_onehot.shape[1]):
        AUC.append(roc_auc_score(y_ture_onehot[:, i], y_pred[:, i], multi_class='ovr', average='macro'))
        AP.append(average_precision_score(y_ture_onehot[:, i], y_pred[:, i], average='macro'))

    return np.round(np.mean(AUC), 4), np.round(np.mean(AP), 4)

def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)
