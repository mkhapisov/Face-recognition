import os
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2
from torchvision.models import resnet18
import matplotlib.pyplot as plt
from tqdm import tqdm
import aligner
import recognizer

class FacesDataset(Dataset):
    def __init__(self, data, mode):
        super().__init__()
        self.data = data.copy()
        self.mode = mode

        if self.mode not in ['train', 'val', 'test']:
            raise ValueError("mode must be either 'train', 'val' or 'test'")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        data_transforms = {
            'train': v2.Compose([
                v2.Resize((250, 250)),
                v2.RandomHorizontalFlip(p=0.5),
                v2.RandomRotation(20),
                v2.RandomChoice([
                    v2.RandomAdjustSharpness(sharpness_factor=2),
                    v2.RandomAdjustSharpness(sharpness_factor=1),
                    v2.RandomAdjustSharpness(sharpness_factor=0)
                ]),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]),

            'val': v2.Compose([
                v2.Resize((250, 250)),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]),

            'test': v2.Compose([
                v2.Resize((250, 250)),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]),
        }

        if self.mode == 'test':
            x = self.data[index]
            x = torch.tensor(x / 255, dtype=torch.float32)
            x = data_transforms[self.mode](x.permute(2, 0, 1))
            return x
        else:
            x, y = self.data[index]
            x = torch.tensor(x / 255, dtype=torch.float32)
            x = data_transforms[self.mode](x.permute(2, 0, 1))
            return x, y

if __name__ == '__main__':
    root = '../dataset'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    data = {}
    faces = []
    label_encoder = []
    labels = []
    for cur_root, cur_dirs, cur_files in os.walk(root):
        if len(cur_files) > 50:
            name = cur_root[11:] # magic number
            label_encoder.append(name)
            data[name] = []
            for i in range(len(cur_files)):
                file = '/'.join((root, name, cur_files[i]))
                data[name].append(file)
                labels.append(label_encoder.index(name))
                faces.append(file)
                if len(data[name]) > 10:
                    break

    print(len(label_encoder))

    print(label_encoder)

    for i in range(5):
        fig, ax = plt.subplots(nrows=1, ncols=5, figsize=(15, 5))
        for j in range(5):
            VP = data['George_W_Bush']
            idx = np.random.randint(len(VP))
            image = plt.imread(VP[idx])
            ax[j].imshow(image)
            ax[j].set_title('George_W_Bush')
        plt.tight_layout()
        plt.show()

    train_faces, val_faces, train_labels, val_labels = train_test_split(faces, labels, test_size=0.3, random_state=42, shuffle=True)
    #shuffle=True is needed because otherwise there would be a small amount of different people in train

    print(len(train_faces))

    aligned_train_data = [(aligner.get_faces(train_faces[i])[0], train_labels[i]) for i in tqdm(range(len(train_faces)))]
    aligned_val_data = [(aligner.get_faces(val_faces[i])[0], val_labels[i]) for i in tqdm(range(len(val_faces)))]

    aligned_train_dataset = FacesDataset(aligned_train_data, mode='train')
    aligned_val_dataset = FacesDataset(aligned_val_data, mode='val')

    for i in range(5):
        fig, ax = plt.subplots(nrows=1, ncols=5, figsize=(15, 5))
        for j in range(5):
            idx = np.random.randint(len(aligned_train_dataset))
            image, name = aligned_train_dataset[idx]
            ax[j].imshow(image.permute(1, 2, 0))
            ax[j].set_title(label_encoder[name])
        plt.tight_layout()
        plt.show()

    model = resnet18(weights='DEFAULT').to(device)
    model.fc = nn.Linear(model.fc.in_features, len(label_encoder))
    print(model)

    criterion = nn.CrossEntropyLoss()
    init_lr = 1e-3
    optimizer = torch.optim.AdamW(model.parameters(), lr=init_lr)

    batch_size = 32
    epochs = 50
    r = recognizer.Recognizer(model, criterion, optimizer, aligned_train_dataset, aligned_val_dataset, batch_size, epochs, device, lr_step=10, lr_coef=0.5)

    layers_to_unfreeze = 30
    r.freeze_layers(layers_to_unfreeze)

    history = r.train()
    
    plt.figure(figsize=(15, 10))
    plt.grid()
    plt.plot(history['train_loss'], color='red')
    plt.plot(history['val_loss'], color='navy')
    plt.title("Loss")
    plt.show()

    plt.figure(figsize=(15, 10))
    plt.grid()
    plt.plot(history['train_acc'], color='red')
    plt.plot(history['val_acc'], color='navy')
    plt.title("Accuracy")
    plt.show()

    root = '../test'
    test_faces = []
    for cur_root, cur_dirs, cur_files in os.walk(root):
        for i in range(len(cur_files)):
            file = '/'.join((root, cur_files[i]))
            test_faces.append(file)

    print(len(test_faces))

    for i in range(5):
        fig, ax = plt.subplots(nrows=1, ncols=5, figsize=(15, 5))
        for j in range(5):
            idx = np.random.randint(len(test_faces))
            image = plt.imread(test_faces[idx])
            ax[j].imshow(image)
        plt.tight_layout()
        plt.show()
    
    aligned_test_data = [aligner.get_faces(test_faces[i])[0] for i in tqdm(range(len(test_faces)))]
    aligned_test_dataset = FacesDataset(aligned_test_data, mode='test')
    test_loader = DataLoader(aligned_test_dataset, batch_size=1, shuffle=False)

    i = 0
    fig, ax = plt.subplots(nrows=1, ncols=5, figsize=(15, 5))
    for inputs in test_loader:
        inputs = torch.FloatTensor(inputs).to(device)
        preds = r.predict_one_sample(inputs)
        predicted = np.argmax(preds)
        image = plt.imread(test_faces[i])
        ax[i % 5].imshow(image)
        ax[i % 5].set_title(label_encoder[predicted])
        i += 1
        if i % 5 == 0:
            plt.tight_layout()
            plt.show()
            fig, ax = plt.subplots(nrows=1, ncols=5, figsize=(15, 5))


