import sklearn
import torch
from sklearn.datasets import make_circles
n_samples=1000
X,y=make_circles(n_samples,noise=0.03,random_state=42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
X.shape,y.shape
X=torch.from_numpy(X).type(torch.float)
y=torch.from_numpy(y).type(torch.float)
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
from torch import nn
class Binary(nn.Module):
  def __init__(self):
    super().__init__()
    #self.layer1=nn.Linear(in_features=2,out_features=5)
    #self.layer2=nn.Linear(in_features=5,out_features=1)
    #Implement same using sequential
    self.two_linear_layers=nn.Sequential(
        nn.Linear(in_features=2,out_features=10),
        nn.ReLU(),
        nn.Linear(in_features=10,out_features=10),
        nn.ReLU(),
        nn.Linear(in_features=10,out_features=1)
    )
  def forward(self,x):
    return self.two_linear_layers(x)
model=Binary().to(device)
model
loss_fn=nn.BCEWithLogitsLoss()
optimizer=torch.optim.SGD(params=model.parameters(),lr=0.1)
def accuracy_fn(y_true, y_pred):
    y_true = y_true.detach().cpu()
    y_pred = y_pred.detach().cpu()
    correct = (y_pred == y_true).sum().item()
    acc = correct / len(y_true)
    return acc
torch.manual_seed(42)
torch.cuda.manual_seed(42)
epochs=1100
X_train,y_train=X_train.to(device),y_train.to(device)
X_test,y_test=X_test.to(device),y_test.to(device)
for epoch in range(epochs):
  model.train()
  y_logits=model(X_train).squeeze()
  y_pred=torch.round(torch.sigmoid(y_logits))
  loss=loss_fn(y_logits,y_train)
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()
  model.eval()
  with torch.inference_mode():
    test_logits=model(X_test).squeeze()
    test_pred=torch.round(torch.sigmoid(test_logits))
    test_loss=loss_fn(test_logits,y_test)
  if epoch%10==0:
    print(f"Epoch:{epoch} training loss:{loss:.2f} test loss:{test_loss:.2f}") 
print(accuracy_fn(y_train,y_pred))
print(accuracy_fn(y_test,test_pred))     
import matplotlib.pyplot as plt

def plot_predictions(model, X, y):
    model.eval()
    with torch.inference_mode():
        # Generate a grid of values over the feature space
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = torch.meshgrid(torch.linspace(x_min, x_max, 100),
                                torch.linspace(y_min, y_max, 100),
                                indexing='xy')
        grid = torch.cat((xx.reshape(-1,1), yy.reshape(-1,1)), dim=1).to(X.device)

        # Get predictions
        logits = model(grid)
        probs = torch.sigmoid(logits).reshape(xx.shape)
        preds = (probs > 0.5).float()

        # Plot decision boundary
        plt.contourf(xx.cpu(), yy.cpu(), preds.cpu(), alpha=0.5, cmap='coolwarm')

        # Plot data points
        plt.scatter(X[:, 0].cpu(), X[:, 1].cpu(), c=y.cpu(), cmap='coolwarm', edgecolors='k')
        plt.title("Model Predictions and Training Data")
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.show()
plot_predictions(model, X_train, y_train)
