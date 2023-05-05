import torch
import torch.nn as nn
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# -------------------------------------------------------------------- No magic numbers
# Hyperparameters
input_size = 784
hidden_size = 500
num_classes = 10
num_epochs = 5
batch_size = 100
learning_rate = 0.001
# -------------------------------------------------------------------- //


# --------------------------------------------------------------------  Load the MNIST train and test sets
# Load the MNIST train set
train_dataset = datasets.MNIST(root='./data', train=True,
                               download=True, transform=transforms.ToTensor())

# Load the MNIST test set
test_dataset = datasets.MNIST(root='./data', train=False,
                              download=True, transform=transforms.ToTensor())

train_dataloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False)
# -------------------------------------------------------------------- //


# -------------------------------------------------------------------- Implement a basic two-layer fully connected neural network
data, target = next(iter(train_dataloader))

# Define the neural network


class Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out
# -------------------------------------------------------------------- //


# -------------------------------------------------------------------- Define the cross entropy loss function and optimizer
model = Net(input_size, hidden_size, num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# -------------------------------------------------------------------- //


# --------------------------------------------------------------------  Implement train and evaluation procedures
# Define the training function
def train(model, dataloader, optimizer, criterion):
    model.train()
    for test, (data, target) in enumerate(dataloader):
        data = data.reshape(-1, input_size).to(device)
        target = target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

# Define the evaluation function


def evaluate(model, dataloader, criterion):
    model.eval()
    running_loss = 0
    running_accuracy = 0
    with torch.no_grad():
        for data, target in dataloader:
            data = data.reshape(-1, input_size).to(device)
            target = target.to(device)
            output = model(data)
            loss = criterion(output, target)
            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            running_accuracy += (predicted == target).sum().item()
    epoch_loss = running_loss / len(dataloader)
    epoch_accuracy = 100. * running_accuracy / len(dataloader.dataset)
    return epoch_loss, epoch_accuracy


# train the model
last_lose_test = []
last_lose_train = []
train_loss_history = []
test_loss_history = []
test_acc_history = []

total_step = len(train_dataloader)
for epoch in range(num_epochs):
    train(model, train_dataloader, optimizer, criterion)
    train_loss, train_acc = evaluate(model, train_dataloader, criterion)
    test_loss, test_acc = evaluate(model, test_dataloader, criterion)
    train_loss_history.append(train_loss)
    test_loss_history.append(test_loss)
    test_acc_history.append(test_acc)
    print(f'Epoch {epoch+1}: Train Loss {train_loss:.6f}, Train Acc {train_acc:.2f}%, Test Loss {test_loss:.6f}, Test Acc {test_acc:.2f}%')

last_lose_train.append(train_loss_history[4])
last_lose_test.append(test_loss_history[4])

plt.plot(train_loss_history, label='Train Loss')
plt.plot(test_loss_history, label='Test Loss')
plt.legend()
plt.show()

# misclassified = []
# model.eval()
# with torch.no_grad():
#     for data, target in test_dataloader:
#         data = data.reshape(-1, input_size).to(device)
#         output = model(data)
#         pred = output.argmax(dim=1, keepdim=True)
#         misclassified.extend([i for i, x in enumerate(
#             pred.eq(target.view_as(pred))) if not x])
#         if len(misclassified) >= 20:
#             break
# fig, axs = plt.subplots(4, 5, figsize=(10, 8))
# for i, idx in enumerate(misclassified):
#     img, label = test_dataset[idx]
#     axs[i // 5, i % 5].imshow(img.squeeze().numpy(), cmap='gray')
#     axs[i // 5, i % 5].set_title(f'True: {label}, Pred: {pred[idx][0]}')
#     axs[i // 5, i % 5].axis('off')
# plt.show()
# -------------------------------------------------------------------- //


# -------------------------------------------------------------------- ex3.5
def task5(model, i):
    train_features = []
    train_labels = []
    with torch.no_grad():
        for data, target in train_dataloader:
            data = data.reshape(-1, input_size).to(device)
            target = target.to(device)
            if (i == 1):
                features = model(data)
            elif (i == 2):
                features = (model.fc1(data)).relu()
            train_features.append(features)
            train_labels.append(target)
    print("check 1")
    train_features = torch.cat(train_features).cpu().numpy()
    train_labels = torch.cat(train_labels).cpu().numpy()

    tsne = TSNE(n_components=2, perplexity=30, n_iter=300)
    z_tsne = tsne.fit_transform(train_features)
    # Plot the t-SNE embedding
    plt.figure(figsize=(10, 8))
    print("check 2")
    for i in range(10):
        plt.scatter(z_tsne[train_labels == i, 0],
                    z_tsne[train_labels == i, 1], label=str(i))
    # plt.title('tSNE graph of ${str}')
    plt.legend()
    plt.show()


model_no_fc2 = Net(input_size, hidden_size, num_classes).to(device)
model_no_fc2.fc1 = model.fc1
model_no_fc2.relu = model.relu
# model_no_fc2.train()
task5(model_no_fc2, 1)
task5(model_no_fc2, 2)
