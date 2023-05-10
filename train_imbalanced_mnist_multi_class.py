# @Time    : 4/12/23 3:35 PM
# @Author  : Zhou-Lin-yong
# @File    : train_imbalanced_mnist_multi_class.py
# @SoftWare: PyCharm
import time, utils, math
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from sklearn.metrics import confusion_matrix
import torch.nn.functional as F
from sklearn import metrics
# from sklearn.metrics import precision_score, f1_score, recall_score
# from sklearn.metrics import roc_curve, precision_recall_curve
from collections import Counter
import Loss_Function
import os

batch_size = 100
num_class = 10

'''
minority_class:  输入少数类的标签值
'''
maj_class = [1,3,5,7,9]
data_pare = 'even_class'
data_num = '5000'
abs_path = './create_imbalaced_mnist/' + data_pare
abs_path_train = abs_path + '/50_' + data_num

train_data_path = abs_path_train + '/imbalanced_x.npy'
train_label_path = abs_path_train + '/imbalanced_y.npy'
test_data_path = abs_path + '/eval_x.npy'
test_label_path = abs_path + '/eval_y.npy'

# Loss_fun = "CE"
# Loss_fun = "FL"
# Loss_fun = "ASL"
# Loss_fun = "CL"
# Loss_fun = "FTL"
# Loss_fun = "HFL"
Loss_fun = 'GPPE'

train_times = 5

if Loss_fun == 'CE':
    Loss = nn.CrossEntropyLoss().cuda()
    save_confu_path = abs_path + '/50_' + data_num + '/result/result_CE/' + str(train_times) + '/'
elif Loss_fun == 'FL':
    Loss = Loss_Function.Focal_Loss().cuda()
    save_confu_path = abs_path + '/50_' + data_num + '/result/result_FL/' + str(train_times) + '/'
elif Loss_fun == 'ASL':
    Loss = Loss_Function.ASLSingleLabel().cuda()
    save_confu_path = abs_path + '/50_' + data_num + '/result/result_ASL/' + str(train_times) + '/'
elif Loss_fun == 'DSCL':
    Loss = Loss_Function.MultiDSCLoss().cuda()
    save_confu_path = abs_path + '/50_' + data_num + '/result/result_DSCL/' + str(train_times) + '/'
elif Loss_fun == 'CL':
    Loss = Loss_Function.Combo_Loss().cuda()
    save_confu_path = abs_path + '/50_' + data_num + '/result/result_CL/' + str(train_times) + '/'
elif Loss_fun == 'FTL':
    Loss = Loss_Function.Focal_Tversky_Loss().cuda()
    save_confu_path = abs_path + '/50_' + data_num + '/result/result_FTL/' + str(train_times) + '/' ## ???
elif Loss_fun == 'HFL':
    Loss = Loss_Function.Hybrid_Focal_Loss().cuda()
    save_confu_path = abs_path + '/50_' + data_num + '/result/result_HFL/' + str(train_times) + '/'
else:
    Loss = Loss_Function.GPPE_Multi_Class().cuda()
    save_confu_path = abs_path + '/50_' + data_num + '/result/result_GPPE/' + str(train_times) + '/'

print('Loss:', Loss)
print('save_confu_path:', save_confu_path)

train_data = np.load(train_data_path)
train_label = np.load(train_label_path)
print(Counter(train_label))

test_data = np.load(test_data_path)
test_label = np.load(test_label_path)
print(Counter(test_label))

# 随机打乱
ssl_data_seed = 1
rng_data = np.random.RandomState(ssl_data_seed)

train_inds = rng_data.permutation(train_data.shape[0])
train_data = train_data[train_inds]
train_label = train_label[train_inds]

test_inds = rng_data.permutation(test_data.shape[0])
test_data = test_data[test_inds]
test_label = test_label[test_inds]

train_data = train_data.reshape((train_data.shape[0], 1, 28, 28))
test_data = test_data.reshape((test_data.shape[0], 1, 28, 28))

print(train_data.shape)

num_batch_train = math.ceil(train_data.shape[0] / batch_size)
test_num_bathces = math.ceil(test_data.shape[0] / batch_size)
print(test_num_bathces)

device = torch.device('cuda')
model = utils.Model(num_class=num_class).to(device=device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.5, 0.999))


maj_min_express = [0 if train_label[i] in maj_class else 1 for i in range(len(train_label))]
maj_min_express = Variable(torch.from_numpy(np.array(maj_min_express)).long()).cuda()
maj_min_express = maj_min_express.view(-1, 1)


if __name__ == "__main__":
    epochs = 100
    for epoch in range(epochs):
        print('epoch:', epoch)
        t1 = time.time()
        for iteration in range(num_batch_train):
            from_l_c = iteration * batch_size
            to_l_c = (iteration + 1) * batch_size
            data = train_data[from_l_c:to_l_c]
            label = train_label[from_l_c:to_l_c]
            maj_min_express_ = maj_min_express[from_l_c:to_l_c]

            data = Variable(torch.from_numpy(data).float()).cuda()
            label = Variable(torch.from_numpy(label).long()).cuda()

            # 梯度清零
            optimizer.zero_grad()
            output_label = model(data)
            if Loss_fun == 'GPPE':
                loss = Loss(output_label, label, maj_min_express_)
            else:
                loss = Loss(output_label, label)

            loss.backward()
            optimizer.step()

        print('Train_time:', time.time() - t1)

        te_data = Variable(torch.from_numpy(test_data).float()).cuda()
        te_label = test_label
        output = model(te_data)
        predict_probility = F.softmax(output, dim=1)
        pi_1_cpu = predict_probility.data.cpu().numpy()

        predict_label = torch.max(output, 1)[1].data

        predict_label_cpu = np.int32(predict_label.cpu().numpy())
        Acc = metrics.accuracy_score(te_label, predict_label.data.cpu().numpy())
        Confu_matir = confusion_matrix(te_label, predict_label_cpu, labels=[0,1,2,3,4,5,6,7,8,9])
        print('ACC:', Acc)

        if epoch > 30:
            np.save(save_confu_path + 'Confu_matir_' + str(epoch) + '.npy', Confu_matir)
            np.save(save_confu_path + 'predicted_probility_' + str(epoch) + '.npy', pi_1_cpu)
            np.save(save_confu_path + 'target_' + str(epoch) + '.npy', te_label)

        print(Confu_matir)



