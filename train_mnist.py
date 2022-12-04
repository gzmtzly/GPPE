# @Time    : 12/4/22 10:53 PM
# @Author  : Zhou-Lin-yong
# @File    : train_mnist.py
# @SoftWare: PyCharm

import time, math, loss, model
import numpy as np
import torch
from torch.autograd import Variable
from sklearn.metrics import confusion_matrix
from collections import Counter
batch_size = 100
num_class = 2

minority_class = 3
data_pare = '3_5'
data_num = '3000'
abs_path = './create_imbalaced_mnist/' + data_pare
abs_path_train = abs_path + '/50_' + data_num

train_data_path = abs_path_train + '/imbalanced_x.npy'
train_label_path = abs_path_train + '/imbalanced_y.npy'
test_data_path = abs_path + '/imb_eval_x.npy'
test_label_path = abs_path + '/imb_eval_y.npy'

Loss = loss.GPPE().cuda()
save_confu_path = abs_path_train = abs_path + '/50_' + data_num + '/result/result_OUR/1/'


print('save_confu_path:', save_confu_path)
print('Loss:', Loss)

train_data = np.load(train_data_path)

train_label = np.load(train_label_path)
train_label = np.array([1 if train_label[i] == minority_class else 0 for i in range(len(train_label))])

print(Counter(train_label))

test_data = np.load(test_data_path)
test_label = np.load(test_label_path)
test_label = np.array([1 if test_label[i] == minority_class else 0 for i in range(len(test_label))])
print(Counter(test_label))


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
model = model.Model_Minst().to(device=device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.5, 0.999))

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

            data = Variable(torch.from_numpy(data).float()).cuda()
            label = Variable(torch.from_numpy(label).long()).cuda()
            label = label.view(data.shape[0], 1)

            optimizer.zero_grad()

            output_label = model(data)
            loss = Loss(output_label, label, epoch)
            loss.backward()
            optimizer.step()

        print('Train_time:', time.time() - t1)
        ground_truth_valid = []
        label_valid = []
        if epoch % 1 == 0:
            for iteration in range(test_num_bathces):
                from_l_c = iteration * batch_size
                to_l_c = (iteration + 1) * batch_size

                batch_data = test_data[from_l_c:to_l_c]
                data = Variable(torch.from_numpy(batch_data).float()).cuda()

                batch_label = test_label[from_l_c:to_l_c]
                label_batch = torch.max(model(data), 1)[1].data

                label_batch = np.int32(label_batch.cpu().numpy())
                label_valid.extend(label_batch)
                ground_truth_valid.extend(batch_label)

        Confu_matir = confusion_matrix(ground_truth_valid, label_valid, labels=[0, 1])
        if epoch > 30:
            np.save(save_confu_path + 'Confu_matir_'+str(epoch)+'.npy', Confu_matir)

        print(Confu_matir)
        print('#' * 40)



