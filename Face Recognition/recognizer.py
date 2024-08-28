import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

class Recognizer:
    def __init__(self, model, criterion, optimizer, train_dataset, val_dataset, batch_size: int, epochs: int, device, lr_step=1000, lr_coef=1.0):
        self.batch_size = batch_size
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        self.epochs = epochs

        self.device = device
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer

        self.lr_step = lr_step
        self.lr_coef = lr_coef
    
    def freeze_layers(self, layers_to_unfreeze: int):
        for param in list(self.model.parameters())[:-layers_to_unfreeze]:
            param.requires_grad = False
    
    def train_epoch(self):
        self.model.train()
        running_loss = 0.0
        running_corrects = 0
        processed_data = 0

        for inputs, labels in tqdm(self.train_loader):
            inputs = torch.FloatTensor(inputs).to(self.device)
            labels = torch.LongTensor(labels).to(self.device)
            self.optimizer.zero_grad()

            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            preds = torch.argmax(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            processed_data += inputs.size(0)

        
        train_loss = running_loss / processed_data
        train_acc = running_corrects.double() / processed_data
        
        return train_loss, train_acc.item()
    
    def eval_epoch(self):
        self.model.eval()
        running_loss = 0.0
        running_corrects = 0
        processed_data = 0

        for inputs, labels in tqdm(self.val_loader):
            inputs = torch.FloatTensor(inputs).to(self.device)
            labels = torch.LongTensor(labels).to(self.device)

            with torch.no_grad():
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                preds = torch.argmax(outputs, 1)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                processed_data += inputs.size(0)
        
        train_loss = running_loss / processed_data
        train_acc = running_corrects.double() / processed_data
        
        return train_loss, train_acc.item()
    
    def train(self):
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        max_score = 0.0
        min_loss = 1000.0
        epoch_max_score = -1
        for epoch in range(1, self.epochs + 1):
            print(f"Epoch: {epoch}")

            train_loss, train_acc = self.train_epoch()
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)

            print("train_loss - %f, train_acc - %f" % (train_loss, train_acc))

            val_loss, val_acc = self.eval_epoch()
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)

            print("val_loss - %f, val_acc - %f" % (val_loss, val_acc))

            if val_acc >= max_score and val_loss < min_loss:
                max_score = val_acc
                min_loss = val_loss
                epoch_max_score = epoch
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_loss': train_loss,
                    'train_acc': train_acc,
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                }, 'best_recognizer.pth')
                print("Model saved!")
            
            if epoch % self.lr_step == 0:
                print("Decreasing learning rate: %f -> %f" % (self.optimizer.param_groups[0]['lr'], self.optimizer.param_groups[0]['lr'] * self.lr_coef))
                self.optimizer.param_groups[0]['lr'] *= self.lr_coef

        print("Max score: %f, Min loss: %f, Epoch: %d" % (max_score, min_loss, epoch_max_score))
        if epoch_max_score > -1:
            checkpoint = torch.load('best_recognizer.pth')
            self.model.load_state_dict(checkpoint['model_state_dict'])
        return history
    
    def predict(self, test_loader):
        self.model.eval()
        logits = []
        for inputs, labels in test_loader:
            inputs = torch.FloatTensor(inputs).to(self.device)
            with torch.no_grad():
                outputs = self.model(inputs).cpu()
                logits.append(outputs)
        
        probs = F.softmax(torch.cat(logits), dim=-1).numpy()
        return probs
    
    def predict_one_sample(self, sample):
        with torch.no_grad():
            self.model.eval()
            sample = sample.to(self.device)
            logit = self.model(sample).cpu()
            probs = F.softmax(logit, dim=-1).numpy()
        return probs


                
    
