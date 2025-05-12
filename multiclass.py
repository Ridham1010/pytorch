import torch
from torch import nn
import sklearn
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
NUM_CLASSES=4
NUM_FEATURES=2
RANDOM_SEED=42
X_blob,y_blob=make_blobs(n_samples=1000,n_features=NUM_FEATURES,centers=NUM_CLASSES,cluster_std=1.5,random_state=RANDOM_SEED)
X_blob=torch.from_numpy(X_blob).type(torch.float)
y_blob=torch.from_numpy(y_blob).type(torch.LongTensor)
X_blob_train,X_blob_test,y_blob_train,y_blob_test=train_test_split(X_blob,y_blob,test_size=0.2,random_state=RANDOM_SEED)

import matplotlib.pyplot as plt

def plot_training_data(X, y, title="Training Data"):
    """
    Plots 2D training data points colored by their label.
    
    Args:
        X (ndarray): Feature matrix of shape (n_samples, 2)
        y (ndarray): Labels array of shape (n_samples,)
        title (str): Title of the plot
    """
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', edgecolors='k', s=40)
    plt.title(title)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.grid(True)
    plt.legend(*scatter.legend_elements(), title="Class")
    plt.show()
plot_training_data(X_blob_train,y_blob_train)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
class multiclass(nn.Module):
  def __init__(self,input_features,output_features,hidden_units=8):
     super().__init__()
     self.linear_layer_stack=nn.Sequential(
         nn.Linear(in_features=input_features,out_features=hidden_units),
         nn.ReLU(),
         nn.Linear(in_features=hidden_units,out_features=hidden_units),
         nn.ReLU(),
         nn.Linear(in_features=hidden_units,out_features=output_features)
     )
  def forward(self,x):
    return self.linear_layer_stack(x)   
model=multiclass(input_features=2,output_features=4,hidden_units=8)    
model
loss_fn=nn.CrossEntropyLoss()
optimizer=torch.optim.SGD(params=model.parameters(),lr=0.1)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
epochs=100
for epoch in range(epochs):
  model.train()
  y_logits=model(X_blob_train)
  y_pred=torch.softmax(y_logits,dim=1).argmax(dim=1)
  loss=loss_fn(y_logits,y_blob_train)
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()
  model.eval()
  with torch.inference_mode():
    test_logits=model(X_blob_test)
    test_pred=torch.softmax(test_logits,dim=1).argmax(dim=1)
    test_loss=loss_fn(test_logits,y_blob_test)
  if epoch%10==0:
    print(f"epoch={epoch} training loss={loss} testing loss={test_loss}")  
def accuracy_fn(y_true, y_pred):
    y_true = y_true.detach().cpu()
    y_pred = y_pred.detach().cpu()
    correct = (y_pred == y_true).sum().item()
    acc = correct / len(y_true)
    return acc
training_acc=accuracy_fn(y_blob_train,y_pred)
test_acc=accuracy_fn(y_blob_test,test_pred)
training_acc,test_acc
plot_training_data(X_blob_train,y_pred)
plot_training_data(X_blob_test,test_pred,title="testing data")
