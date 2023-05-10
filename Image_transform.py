from imutils import paths
import read_caltech101
from collections import Counter
from sklearn.model_selection import train_test_split
from torchvision.transforms import transforms
from torch.utils.data import DataLoader, Dataset

image_paths = list(paths.list_images('../data/caltech-101/101_ObjectCategories'))
data, name2label, labels = read_caltech101.data_processor(image_paths, size=65)
print(Counter(labels))
# Counter({44: 800, 89: 798, 29: 435, 82: 435, 30: 239, 49: 200, 69: 128, 4: 123, 17: 114, 83: 107, 26: 100…………})
# print(len(labels))


train_transform = transforms.Compose(
    [transforms.ToPILImage(),
     # transforms.Resize((224, 224)),
     transforms.ToTensor(),
     transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])])
val_transform = transforms.Compose(
    [transforms.ToPILImage(),
     # transforms.Resize((224, 224)),
     transforms.ToTensor(),
     transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])])

(X, x_val, Y, y_val) = train_test_split(data, labels,
                                        test_size=0.2,
                                        stratify=labels,
                                        random_state=42)
(x_train, x_test, y_train, y_test) = train_test_split(X, Y,
                                                      test_size=0.25,
                                                      random_state=42)
print(f"x_train examples: {x_train.shape}\nx_test examples: {x_test.shape}\nx_val examples: {x_val.shape}")

class ImageDataset(Dataset):
    def __init__(self, images, labels=None, transforms=None):
        self.X = images
        self.y = labels
        self.transforms = transforms

    def __len__(self):
        return (len(self.X))

    def __getitem__(self, i):
        data = self.X[i][:]

        if self.transforms:
            data = self.transforms(data)

        if self.y is not None:
            return (data, self.y[i])
        else:
            return data

def data_loader(batch_size=64):
    train_data = ImageDataset(x_train, y_train, train_transform)
    val_data = ImageDataset(x_val, y_val, val_transform)
    test_data = ImageDataset(x_test, y_test, val_transform)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader