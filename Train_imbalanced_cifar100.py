# @Time    : 4/24/23 12:25 PM
# @Author  : Zhou-Lin-yong
# @File    : Train_imbalanced_cifar100.py
# @SoftWare: PyCharm

import time, Model, math
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from sklearn.metrics import confusion_matrix
from collections import Counter
import Loss_Function
import sklearn.metrics as metrics
import torch.nn.functional as F

batch_size = 100
num_class = 2

'''
minority_class:  input minority class
'''
minority_class = 8
data_pare = '8_11'
data_num = '500_2500'
abs_path = './create_imbalaced_cifar100/' + data_pare
abs_path_train = abs_path + '/' + data_num

train_data_path = abs_path_train + '/imbalanced_x.npy'
train_label_path = abs_path_train + '/imbalanced_y.npy'
test_data_path = abs_path + '/imb_eval_x.npy'
test_label_path = abs_path + '/imb_eval_y.npy'

# Loss_fun = "CE"
# Loss_fun = "FL"
# Loss_fun = "ASL"
# Loss_fun = "CL"
# Loss_fun = "FTL"
# Loss_fun = "HFL"
Loss_fun = 'GPPE'

train_times = 1

if Loss_fun == 'CE':
    Loss = nn.CrossEntropyLoss().cuda()
    save_confu_path = abs_path + '/' + data_num + '/result/result_CE/' + str(train_times) + '/'
elif Loss_fun == 'FL':
    Loss = Loss_Function.Focal_Loss().cuda()
    save_confu_path = abs_path + '/' + data_num + '/result/result_FL/' + str(train_times) + '/'
elif Loss_fun == 'ASL':
    Loss = Loss_Function.ASLSingleLabel().cuda()
    save_confu_path = abs_path + '/' + data_num + '/result/result_ASL/' + str(train_times) + '/'
elif Loss_fun == 'CL':
    Loss = Loss_Function.Combo_Loss().cuda()
    save_confu_path = abs_path + '/' + data_num + '/result/result_CL/' + str(train_times) + '/'
elif Loss_fun == 'FTL':
    Loss = Loss_Function.Focal_Tversky_Loss_2().cuda()
    save_confu_path = abs_path + '/' + data_num + '/result/result_FTL/' + str(train_times) + '/'
elif Loss_fun == 'HFL':
    Loss = Loss_Function.Hybrid_Focal_Loss().cuda()
    save_confu_path = abs_path + '/' + data_num + '/result/result_HFL/' + str(train_times) + '/'
else:
    Loss = Loss_Function.GPPE().cuda()
    save_confu_path = abs_path + '/' + data_num + '/result/result_GPPE/' + str(train_times) + '/'

print('Loss:', Loss)
print('save_confu_path:', save_confu_path)

train_data = np.load(train_data_path)
train_label = np.load(train_label_path)
print('train:', Counter(train_label))

train_label = np.array([1 if train_label[i] == minority_class else 0 for i in range(len(train_label))])

test_data = np.load(test_data_path)
test_label = np.load(test_label_path)
print('test:', Counter(test_label))

test_label = np.array([1 if test_label[i] == minority_class else 0 for i in range(len(test_label))])

# shuffle
ssl_data_seed = 1
rng_data = np.random.RandomState(ssl_data_seed)

train_inds = rng_data.permutation(train_data.shape[0])
train_data = train_data[train_inds]
train_label = train_label[train_inds]

test_inds = rng_data.permutation(test_data.shape[0])
test_data = test_data[test_inds]
test_label = test_label[test_inds]

num_batch_train = math.ceil(train_data.shape[0] / batch_size)
test_num_bathces = math.ceil(test_data.shape[0] / batch_size)
print(test_num_bathces)

device = torch.device('cuda')
model = Model.Model_Cifar100(num_class).to(device=device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.5, 0.999))

if __name__ == "__main__":
    epochs = 50
    for epoch in range(epochs):
        print('epoch:', epoch)
        t1 = time.time()
        for iteration in range(num_batch_train):
            from_l_c = iteration * batch_size
            to_l_c = (iteration + 1) * batch_size

            data = train_data[from_l_c:to_l_c]
            label = train_label[from_l_c:to_l_c]

            data = Variable(torch.from_numpy(data).float()).cuda()
            label = Variable(torch.from_numpy(label).long()).cuda()

            
            optimizer.zero_grad()

            output_label = model(data)
            loss = Loss(output_label, label)

            loss.backward()
            optimizer.step()

        print('Train_time:', time.time() - t1)
        ground_truth_valid = []
        label_valid = []
        probility_predicted = []
        if epoch % 1 == 0:
            for iteration in range(test_num_bathces):
                from_l_c = iteration * batch_size
                to_l_c = (iteration + 1) * batch_size

                batch_data = test_data[from_l_c:to_l_c]
                data = Variable(torch.from_numpy(batch_data).float()).cuda()

                batch_label = test_label[from_l_c:to_l_c]

                output = model(data)
                predict_pro = F.softmax(output, dim=1)
                pi_1_cpu = predict_pro.data.cpu().numpy()

                label_batch = torch.max(output, 1)[1].data

                label_batch = np.int32(label_batch.cpu().numpy())
                label_valid.extend(label_batch)
                ground_truth_valid.extend(batch_label)
                probility_predicted.extend(pi_1_cpu)

        pp = np.array(probility_predicted)
        Confu_matir = confusion_matrix(ground_truth_valid, label_valid, labels=[0, 1])
        Acc = metrics.accuracy_score(ground_truth_valid, label_valid)
        print(Confu_matir)
        Precision = Confu_matir[1][1] / (Confu_matir[1][1] + Confu_matir[0][1])
        Recall = Confu_matir[1][1] / (Confu_matir[1][1] + Confu_matir[1][0])
        F_score = 2 * Precision * Recall / (Precision + Recall)
        AUC = metrics.roc_auc_score(ground_truth_valid, pp[:, 1])
        AP = metrics.average_precision_score(ground_truth_valid, pp[:, 1])

        # print('Precision:', Precision)
        # print('Recall:', Recall)
        print('F_score:', F_score)
        # print('ACC:', Acc)
        print('AUC:', AUC)
        print('AP:', AP)
        print('%'*50)


        if epoch > epochs - 20:
            np.save(save_confu_path + 'Confu_matir_' + str(epoch) + '.npy', Confu_matir)
            np.save(save_confu_path + 'predicted_probility_' + str(epoch) + '.npy', np.array(probility_predicted))
            np.save(save_confu_path + 'target_' + str(epoch) + '.npy', np.array(ground_truth_valid))



