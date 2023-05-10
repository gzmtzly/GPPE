
import os, cv2
import numpy as np
from imutils import paths

def data_processor(image_paths, size=65):
    """
    将文件中的图片读取出来并整理成data和labels 共101类
    :return:
    """
    data = []
    labels = []
    label_name = []
    name2label = {}
    for idx, image_path in enumerate(image_paths):
        name = image_path.split(os.path.sep)[-2]  #获取类别名
        # if name == 'BACKGROUND_Google':
        #     continue
        #读取图像并进行处理
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  #将BGR转化为RGB
        image = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
        data.append(image)
        label_name.append(name)
    # print('label_name:', label_name)

    data = np.array(data)
    label_name = list(dict.fromkeys(label_name))  #利用字典进行去重
    label_name = np.array(label_name)
    # print('label_name:', label_name)

    # 生成0-100的类标签 对应label_name中的文件名
    for idx, name in enumerate(label_name):
        name2label[name] = idx  #每个类别分配一个标签
    for idx, image_path in enumerate(image_paths):
        labels.append(name2label[image_path.split(os.path.sep)[-2]])
    labels = np.array(labels)
    return data, name2label, labels


