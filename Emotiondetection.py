!git clone https://github.com/parth1620/Facial-Expression-Dataset.git
!pip install -U git+https://github.com/albumentations-team/albumentations
!pip install timm
!pip install --upgrade opencv-contrib-python
import numpy as np
import matplotlib.pyplot as plt
import torch
TRAIN_IMG_FOLDER_PATH='/content/Facial-Expression-Dataset/train'
VALID_IMG_FOLDER_PATH='/content/Facial-Expression-Dataset/validation'
LR=0.001
BATCH_SIZE=32
EPOCHS=15
DEVICE='cuda'
MODEL_NAME='efficientnet_b0'
from torchvision.datasets import ImageFolder
from torchvision import transforms as T
train_augs=T.Compose([
    T.RandomHorizontalFlip(p=0.5),
    T.RandomRotation(degrees=(-20,+20)),
    T.ToTensor()
])
valid_augs=T.Compose([T.ToTensor()])
trainset=ImageFolder(TRAIN_IMG_FOLDER_PATH,transform=train_augs)
validset=ImageFolder(VALID_IMG_FOLDER_PATH,transform=valid_augs)
print(f"Total no. of examples in trainset : {len(trainset)}")
print(f"Total no. of examples in validset : {len(validset)}")
print(trainset.class_to_idx)
image,label=trainset[20]
plt.imshow(image.permute(1,2,0))
plt.title(label)
from torch.utils.data import DataLoader
trainloader=DataLoader(trainset,batch_size=BATCH_SIZE,shuffle=True)
validloader=DataLoader(validset,batch_size=BATCH_SIZE)
print(f"Total no. of batches in trainloader : {len(trainloader)}")
print(f"Total no. of batches in validloader : {len(validloader)}")
for images,labels in trainloader:
  break;

print(f"One image batch shape : {images.shape}")
print(f"One label batch shape : {labels.shape}")
import timm
from torch import nn
class FaceModel(nn.Module):
  def __init__(self):
     super().__init__()
     self.eff_net=timm.create_model('efficientnet_b0',pretrained=True,num_classes=7)
  def forward(self,images,labels=None):
    logits=self.eff_net(images)
    if labels != None:
      loss=nn.CrossEntropyLoss()(logits,labels)
      return logits,loss
    return logits
model=FaceModel()
model.to(DEVICE)
images = images.to(DEVICE)
labels = labels.to(DEVICE)
from tqdm import tqdm
def multiclass_accuracy(y_pred,y_true):
    top_p,top_class = y_pred.topk(1,dim = 1)
    equals = top_class == y_true.view(*top_class.shape)
    return torch.mean(equals.type(torch.FloatTensor))
def train_fn(model,dataloader,optimizer,current_epo):
  model.train()
  total_loss=0.0
  total_acc=0.0
  tk=tqdm(dataloader,desc="EPOCH"+"{TRAIN}"+str(current_epo+1)+"/"+str(EPOCHS))
  for t,(images,labels) in enumerate(tk):
    images, labels=images.to(DEVICE),labels.to(DEVICE)
    optimizer.zero_grad()
    logits,loss=model(images,labels)
    loss.backward()
    optimizer.step()
    total_loss+=loss.item()
    total_acc+=multiclass_accuracy(logits,labels)
    tk.set_postfix({'loss':'%6f' %float(total_loss/(t+1)),'acc':'%6f' %float(total_acc/(t+1)),})
  return total_loss/len(dataloader),total_acc/len(dataloader)
def eval_fn(model, dataloader, current_epo):
    model.eval()
    total_loss = 0.0
    total_acc = 0.0

    all_preds = []
    all_labels = []

    tk = tqdm(dataloader, desc=f"EPOCH{{VALID}} {current_epo+1}/{EPOCHS}")
    with torch.no_grad():
        for t, (images, labels) in enumerate(tk):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            logits, loss = model(images, labels)
            total_loss += loss.item()
            total_acc += multiclass_accuracy(logits, labels)

            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            tk.set_postfix({
                'loss': f'{total_loss / (t + 1):.6f}',
                'acc': f'{total_acc / (t + 1):.6f}'
            })

    return total_loss / len(dataloader), total_acc / len(dataloader), all_preds, all_labels
optimizer= torch.optim.Adam(model.parameters(),lr=LR)
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

best_valid_loss = np.inf
final_preds = []
final_labels = []

for i in range(EPOCHS):
    train_loss, train_acc = train_fn(model, trainloader, optimizer, i)
    valid_loss, valid_acc, all_preds, all_labels = eval_fn(model, validloader, i)

    if valid_loss < best_valid_loss:
        torch.save(model.state_dict(), 'best-weights.pt')
        print("SAVED-BEST-WEIGHTS")
        best_valid_loss = valid_loss

    # Store final predictions and labels
    final_preds = all_preds
    final_labels = all_labels

# After training, display single confusion matrix for full validation set
cm = confusion_matrix(final_labels, final_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.title("Final Confusion Matrix on Validation Set")
plt.show()
