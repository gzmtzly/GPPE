# @Time    : 10/11/22 12:01 AM
# @Author  : Zhou-Lin-yong
# @File    : Train_imbalanced_cifar10.py
# @SoftWare: PyCharm
import time, Model, math
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from sklearn.metrics import confusion_matrix, roc_auc_score, average_precision_score
from collections import Counter
import Loss_Function
import sklearn.metrics as metrics
import torch.nn.functional as F
batch_size = 100
num_class = 2

'''
minority_class:  input minority class
'''
minority_class = 3
data_pare = '3_8'
data_num = '3000'
abs_path = './create_imbalaced_cifar10/' + data_pare
abs_path_train = abs_path + '/500_' + data_num

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
    save_confu_path = abs_path + '/500_' + data_num + '/result/result_CE/' + str(train_times) + '/'
elif Loss_fun == 'FL':
    Loss = Loss_Function.Focal_Loss().cuda()
    save_confu_path = abs_path + '/500_' + data_num + '/result/result_FL/' + str(train_times) + '/'
elif Loss_fun == 'ASL':
    Loss = Loss_Function.ASLSingleLabel().cuda()
    save_confu_path = abs_path + '/500_' + data_num + '/result/result_ASL/' + str(train_times) + '/'
elif Loss_fun == 'CL':
    Loss = Loss_Function.Combo_Loss().cuda()
    save_confu_path = abs_path + '/500_' + data_num + '/result/result_CL/' + str(train_times) + '/'
elif Loss_fun == 'FTL':
    Loss = Loss_Function.Focal_Tversky_Loss_2().cuda()
    save_confu_path = abs_path + '/500_' + data_num + '/result/result_FTL/' + str(train_times) + '/' 
elif Loss_fun == 'HFL':
    Loss = Loss_Function.Hybrid_Focal_Loss().cuda()
    save_confu_path = abs_path + '/500_' + data_num + '/result/result_HFL/' + str(train_times) + '/'
else:
    Loss = Loss_Function.GPPE().cuda()
    save_confu_path = abs_path + '/500_' + data_num + '/result/result_GPPE/' + str(train_times) + '/'

print('Loss:', Loss)
print('save_confu_path:', save_confu_path)

train_data = np.load(train_data_path)
train_label = np.load(train_label_path)
print('train:', Counter(train_label))

train_label = np.array([1 if train_label[i] == minority_class else 0 for i in range(len(train_label))])

test_data = np.load(test_data_path)
test_label = np.load(test_label_path)
print('train:', Counter(test_label))

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
model = Model.Model_Cifar10(num_class).to(device=device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.5, 0.999))

if __name__ == "__main__":
    EPOCH = 100
    for epoch in range(EPOCH):
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

        Confu_matir = confusion_matrix(ground_truth_valid, label_valid, labels=[0, 1])
        ground_truth_valid = np.array(ground_truth_valid)
        probility_predicted = np.array(probility_predicted)
        # Acc = metrics.accuracy_score(ground_atruth_valid, label_valid)
        precision = metrics.precision_score(ground_truth_valid, label_valid)
        recall = metrics.recall_score(ground_truth_valid, label_valid)
        f_score = metrics.f1_score(ground_truth_valid, label_valid)
        AUC = roc_auc_score(ground_truth_valid, probility_predicted[:, -1])
        AP = average_precision_score(ground_truth_valid, probility_predicted[:, -1])

        print(Confu_matir)
        print('Precision:', precision)
        print('Recall:', recall)
        print('F_score:', f_score)
        print('AUC:', AUC)
        print('AP:', AP)
        print('&'*50)

        if epoch > 90:
            np.save(save_confu_path + 'Confu_matir_' + str(epoch) + '.npy', Confu_matir)
            np.save(save_confu_path + 'predicted_probility_' + str(epoch) + '.npy', probility_predicted)
            np.save(save_confu_path + 'target_' + str(epoch) + '.npy', ground_truth_valid)



