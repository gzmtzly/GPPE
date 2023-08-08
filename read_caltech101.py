
import os, cv2
import numpy as np
from imutils import paths

def data_processor(image_paths, size=65):
    data = []
    labels = []
    label_name = []
    name2label = {}
    for idx, image_path in enumerate(image_paths):
        name = image_path.split(os.path.sep)[-2]  
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
        image = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
        data.append(image)
        label_name.append(name)

    data = np.array(data)
    label_name = list(dict.fromkeys(label_name)) 
    label_name = np.array(label_name)
   
    for idx, name in enumerate(label_name):
        name2label[name] = idx  
    for idx, image_path in enumerate(image_paths):
        labels.append(name2label[image_path.split(os.path.sep)[-2]])
    labels = np.array(labels)
    return data, name2label, labels


