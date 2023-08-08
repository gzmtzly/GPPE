# @Time    : 4/24/23 6:17 PM
# @Author  : Zhou-Lin-yong
# @File    : Train_imbalanced_cifar100_multiclass.py
# @SoftWare: PyCharm
import time, math
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from sklearn.metrics import confusion_matrix
from collections import Counter
import sklearn.metrics as metrics
import torch.nn.functional as F
import Loss_Function, Model, utils
batch_size = 100
num_class = 100

maj_class = [0,1,2,3,4]
data_pare = 'first_5'  
data_num = '100_500'
abs_path = './create_imbalaced_cifar100/' + data_pare
abs_path_train = abs_path + '/' + data_num


train_data_path = abs_path_train + '/imbalanced_x.npy'
train_label_path = abs_path_train + '/imbalanced_y.npy'
test_data_path = abs_path + '/imb_eval_x.npy'
test_label_path = abs_path + '/imb_eval_y.npy'

# Loss_fun = "CE"
# Loss_fun = "FL"
# Loss_fun = "ASL"
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
    Loss = Loss_Function.Focal_Tversky_Loss().cuda()
    save_confu_path = abs_path + '/' + data_num + '/result/result_FTL/' + str(train_times) + '/'
elif Loss_fun == 'HFL':
    Loss = Loss_Function.Hybrid_Focal_Loss().cuda()
    save_confu_path = abs_path + '/' + data_num + '/result/result_HFL/' + str(train_times) + '/'
else:
    Loss = Loss_Function.GPPE_Multi_Class().cuda()
    save_confu_path = abs_path + '/' + data_num + '/result/result_GPPE/' + str(train_times) + '/'

print('Loss:', Loss)
print('save_confu_path:', save_confu_path)


train_data = np.load(train_data_path)
train_label = np.load(train_label_path)
print('train:', Counter(train_label))

test_data = np.load(test_data_path)
test_label = np.load(test_label_path)
print('test:', Counter(test_label))

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

optimizer = torch.optim.Adam(model.parameters(), lr=0.0002, betas=(0.5, 0.999))


maj_min_express = [0 if train_label[i] in maj_class else 1 for i in range(len(train_label))]
maj_min_express = Variable(torch.from_numpy(np.array(maj_min_express)).long()).cuda()
maj_min_express = maj_min_express.view(-1, 1)

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

            maj_min_express_ = maj_min_express[from_l_c:to_l_c]

            data = Variable(torch.from_numpy(data).float()).cuda()
            label = Variable(torch.from_numpy(label).long()).cuda()

            optimizer.zero_grad()

            output_label = model(data)
            if Loss_fun == 'GPPE':
                loss = Loss(output_label, label, maj_min_express_)
            else:
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
        Confu_matir = confusion_matrix(ground_truth_valid, label_valid, labels=list(np.arange(100)))
        Acc = metrics.accuracy_score(ground_truth_valid, label_valid)
        f1 = metrics.f1_score(ground_truth_valid, label_valid, average='macro')
        mAUC, mAP = utils.comput_mAUC_mAP(ground_truth_valid, pp)

        print('Confu_matir:', Confu_matir)
        print('ACC:', Acc)
        print('mF1:', f1)
        print('mAUC:', mAUC)
        print('mAP:', mAP)

        print('%'*50)

        if epoch > epochs - 30:
            np.save(save_confu_path + 'Confu_matir_' + str(epoch) + '.npy', Confu_matir)
            np.save(save_confu_path + 'predicted_probility_' + str(epoch) + '.npy', np.array(probility_predicted))
            np.save(save_confu_path + 'target_' + str(epoch) + '.npy', np.array(ground_truth_valid))
