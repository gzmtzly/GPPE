
import torch, time, utils
import read_caltech101, Loss_function
import Image_transform
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
import sklearn.metrics as metrics

batch_size = 100
num_class = 100
# '''
# maj_class:  输入少数类的标签值
# '''
maj_class = [44, 89, 29, 82, 30, 49]

# Loss_fun = "CE"
# Loss_fun = "FL"
# Loss_fun = "ASL"
Loss_fun = 'GPPE'

train_times = 4

if Loss_fun == 'CE':
    Loss = nn.CrossEntropyLoss().cuda()
    save_confu_path = './result/result_CE/' + str(train_times) + '/'
elif Loss_fun == 'FL':
    Loss = Loss_function.Focal_Loss().cuda()
    save_confu_path = './result/result_FL/' + str(train_times) + '/'
elif Loss_fun == 'ASL':
    Loss = Loss_function.ASLSingleLabel().cuda()
    save_confu_path = './result/result_ASL/' + str(train_times) + '/'
elif Loss_fun == 'CL':
    Loss = Loss_function.Combo_Loss().cuda()
    save_confu_path = './result/result_CL/' + str(train_times) + '/'
elif Loss_fun == 'FTL':
    Loss = Loss_function.Focal_Tversky_Loss().cuda()
    save_confu_path = './result/result_FTL/' + str(train_times) + '/'
elif Loss_fun == 'HFL':
    Loss = Loss_function.Hybrid_Focal_Loss().cuda()
    save_confu_path = './result/result_HFL/' + str(train_times) + '/'
else:
    Loss = Loss_function.GPPE_Multi_Class().cuda()
    save_confu_path = './result/result_GPPE/' + str(train_times) + '/'

print('Loss:', Loss)
print('save_confu_path:', save_confu_path)

### load data
train_loader, val_loader, test_loader = Image_transform.data_loader(batch_size=batch_size)

device = torch.device('cuda')
model = Caltech101_Model.AlexNet(init_weights=True).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.5, 0.999), weight_decay=0.0005)


if __name__ == "__main__":
    epochs = 100
    for epoch in range(epochs):
        print('epoch:', epoch)
        t1 = time.time()
        for batch_idx, (data, label) in enumerate(train_loader):
            data, label = data.to(device), label.to(device)
            # print('data:', data.shape)

            label = label.to(torch.int64)  # 类型转换
            # label_maj = [0 if label[i] in maj_class else 1 for i in range(label.shape[0])]

            # 梯度清零
            optimizer.zero_grad()
            output_label = model(data)
            if Loss_fun == 'GPPE':
                maj_min_express = [0 if label[i] in maj_class else 1 for i in range(label.shape[0])]
                maj_min_express = Variable(torch.from_numpy(np.array(maj_min_express)).long()).cuda()
                maj_min_express_ = maj_min_express.view(-1, 1)
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
            for batch_idx, (data, label) in enumerate(test_loader):
                # data, label = data.to(device), label.to(device)
                data = data.to(device)
                output = model(data)
                predict_pro = F.softmax(output, dim=1)
                pi_1_cpu = predict_pro.data.cpu().numpy()

                label_batch = torch.max(output, 1)[1].data

                label_batch = np.int32(label_batch.cpu().numpy())
                label_valid.extend(label_batch)
                ground_truth_valid.extend(label.numpy())
                probility_predicted.extend(pi_1_cpu)

        pp = np.array(probility_predicted)
        Confu_matir = confusion_matrix(ground_truth_valid, label_valid, labels=list(np.arange(101)))
        f_score = metrics.f1_score(ground_truth_valid, np.argmax(pp, axis=1), average='macro')
        Acc = metrics.accuracy_score(ground_truth_valid, label_valid)
        mAUC, mAP = caltech101_utils.comput_mAUC_mAP(ground_truth_valid, pp)

        print('Confu_matir:', Confu_matir)
        print('ACC:', Acc)
        print('F_score:', f_score)
        print('mAUC:', mAUC)
        print('mAP:', mAP)

        print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')

        if epoch > epochs - 70:
            np.save(save_confu_path + 'Confu_matir_' + str(epoch) + '.npy', Confu_matir)
            np.save(save_confu_path + 'predicted_probility_' + str(epoch) + '.npy', np.array(probility_predicted))
            np.save(save_confu_path + 'target_' + str(epoch) + '.npy', np.array(ground_truth_valid))