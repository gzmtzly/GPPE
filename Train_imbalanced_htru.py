# @Time    : 4/29/23 7:49 AM
# @Author  : Zhou-Lin-yong
# @File    : Train_imbalanced_htru.py
# @SoftWare: PyCharm
import time, math, Model, utils
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

abs_path = './Htru_result'

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
    save_confu_path = abs_path + '/result_CE/' + str(train_times) + '/'
    # save_confu_path = abs_path + '/result_CE_ROS/' + str(train_times) + '/'
    # save_confu_path = abs_path + '/result_CE_RUS/' + str(train_times) + '/'
elif Loss_fun == 'FL':
    Loss = Loss_Function.Focal_Loss().cuda()
    save_confu_path = abs_path + '/result_FL/' + str(train_times) + '/'
elif Loss_fun == 'ASL':
    Loss = Loss_Function.ASLSingleLabel().cuda()
    save_confu_path = abs_path + '/result_ASL/' + str(train_times) + '/'
elif Loss_fun == 'CL':
    Loss = Loss_Function.Combo_Loss().cuda()
    save_confu_path = abs_path + '/result_CL/' + str(train_times) + '/'
elif Loss_fun == 'FTL':
    Loss = Loss_Function.Focal_Tversky_Loss_2().cuda()
    save_confu_path = abs_path + '/result_FTL/' + str(train_times) + '/'
elif Loss_fun == 'HFL':
    Loss = Loss_Function.Hybrid_Focal_Loss().cuda()
    save_confu_path = abs_path + '/result_HFL/' + str(train_times) + '/'
else:
    Loss = Loss_Function.GPPE(beta_neg=0).cuda()
    save_confu_path = abs_path + '/result_GPPE/' + str(train_times) + '/'


print('Loss:', Loss)
print('save_confu_path:', save_confu_path)

### train data
train_data_pulsar_path = np.load("./htru_file_info_trans64/pulsar_path_trans64.npy")[:360]
train_data_rfi_path = np.load("./htru_file_info_trans64/rfi_path_trans64.npy")[:27000]
print('train_data_pulsar_path:', len(train_data_pulsar_path))
print('train_data_rfi_path:', len(train_data_rfi_path))

### test data
test_data_pulsar_path = np.load("./htru_file_info_trans64/pulsar_path_trans64.npy")[600:]
test_data_rfi_path = np.load("./htru_file_info_trans64/rfi_path_trans64.npy")[45000:]
print('test_data_pulsar_path:', len(test_data_pulsar_path))
print('test_data_rfi_path:', len(test_data_rfi_path))

ssl_data_seed = 1
rng_data = np.random.RandomState(ssl_data_seed)

train_data_path = np.concatenate((train_data_pulsar_path, train_data_rfi_path))
test_data_path = np.concatenate((test_data_pulsar_path, test_data_rfi_path))

train_inds = rng_data.permutation(train_data_path.shape[0])
train_data_path = train_data_path[train_inds]

test_inds = rng_data.permutation(test_data_path.shape[0])
test_data_path = test_data_path[test_inds]

train_data = []
train_label = []
for i in train_data_path:
    data, label = np.load(i, allow_pickle=True)
    train_data.append(data)
    train_label.append(label)

train_data = np.array(train_data)
train_label = np.array(train_label)
print('train:', Counter(train_label))

test_data = []
test_label = []
for i in test_data_path:
    data, label = np.load(i, allow_pickle=True)
    test_data.append(data)
    test_label.append(label)

test_data = np.array(test_data)
test_label = np.array(test_label)
print('test:', Counter(test_label))

train_data = train_data.reshape((train_data.shape[0], 1, 64, 64))
test_data = test_data.reshape((test_data.shape[0], 1, 64, 64))

train_label = np.int32(train_label)
test_label = np.int32(test_label)

# print('train_data:', train_data.shape)
# print('test_data:', test_data.shape)

num_batch_train = math.ceil(train_data.shape[0] / batch_size)
test_num_bathces = math.ceil(test_data.shape[0] / batch_size)

CONV = Model.Model_Ttru().cuda()

optimer = torch.optim.Adam(CONV.parameters(), lr=0.02, betas=(0.5, 0.999))   

print('---------- Networks architecture -------------')
utils.print_network(CONV)
print('-----------------------------------------------')

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

        output_label = CONV(data)
        loss = Loss(output_label, label)

        optimer.zero_grad()
        loss.backward()
        optimer.step()

    print('Training time:', time.time() - t1)
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

            output = CONV(data)
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
    # Acc = metrics.accuracy_score(ground_truth_valid, label_valid)
    precision = metrics.precision_score(ground_truth_valid, label_valid)
    recall = metrics.recall_score(ground_truth_valid, label_valid)
    f_score = metrics.f1_score(ground_truth_valid, label_valid)

    print(Confu_matir)
    print('Precision:', precision)
    print('Recall:', recall)
    print('F_score:', f_score)

    if epoch > EPOCH - 20:
        np.save(save_confu_path + 'Confu_matir_' + str(epoch) + '.npy', Confu_matir)
        np.save(save_confu_path + 'predicted_probility_' + str(epoch) + '.npy', probility_predicted)
        np.save(save_confu_path + 'target_' + str(epoch) + '.npy', ground_truth_valid)
