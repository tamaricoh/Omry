import torch
import torch.nn as nn
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# -------------------------------------------------------------------- No magic numbers
# Hyperparameters
input_size = 784
hidden_size = 128
num_classes = 10
num_epochs = 5
batch_size = 100
learning_rate = 0.01
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

    # def forward(self, x):
    #     x = x.view(-1, 28 * 28)
    #     x = torch.relu(self.fc1(x))
    #     x = self.fc2(x)
    #     return x
# -------------------------------------------------------------------- //


# -------------------------------------------------------------------- Define the cross entropy loss function and optimizer
# model = Net(input_size, hidden_size, num_classes)
model = Net(input_size, hidden_size, num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# -------------------------------------------------------------------- //


# --------------------------------------------------------------------  Implement train and evaluation procedures
# Define the training function
def train(model, dataloader, optimizer, criterion, i):
    model.train()
    for test, (data, target) in enumerate(dataloader):
        # data = data.to(device)
        data = data.reshape(-1, input_size).to(device)
        target = target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if (i+1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch+1, num_epochs, i+1, total_step, loss.item()))


# Define the evaluation function
def evaluate(model, dataloader, criterion):
    model.eval()
    running_loss = 0
    running_accuracy = 0
    with torch.no_grad():
        for data, target in dataloader:
            data = data.to(device)
            target = target.to(device)
            output = model(data)
            loss = criterion(output, target)
            running_loss += loss.item()
            # running_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            # predicted = torch.max(output, 1)
            # running_accuracy += (predicted == target).sum().item()
            running_accuracy += pred.eq(target.view_as(pred)).sum().item()
            running_accuracy = 100. * \
                running_accuracy / len(dataloader.dataset)
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_accuracy = running_accuracy / len(dataloader.dataset)
    return epoch_loss, epoch_accuracy
# -------------------------------------------------------------------- //

# # Train the network
# last_lose_test = []
# last_lose_train = []
# train_loss_history = []
# test_loss_history = []
# test_acc_history = []
# for epoch in range(5):
#     train(model, train_dataloader, optimizer, criterion)
#     train_loss, train_acc = evaluate(model, train_dataloader, criterion)
#     test_loss, test_acc = evaluate(model, test_dataloader, criterion)
#     train_loss_history.append(train_loss)
#     test_loss_history.append(test_loss)
#     test_acc_history.append(test_acc)


# print("std of train : ", np.std(last_lose_train))
# print("std of test : ", np.std(last_lose_test))
# print("mean of train : ", np.mean(last_lose_train))
# print("mean of test : ", np.mean(last_lose_test))
# # Plot the train and test loss curves


# plt.plot(train_loss_history, label='Train Loss')
# plt.plot(test_loss_history, label='Test Loss')
# # plt.legend()
# # plt.show()
# plt.savefig("output_plt.jpg")

# # if task == 1:
# # Plot some of the misclassified images
# misclassified = []
# model.eval()
# with torch.no_grad():
#     for data, target in test_dataloader:
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
# # plt.savefig("output_nums.jpg")
# plt.show()


# train the model

last_lose_test = []
last_lose_train = []
train_loss_history = []
test_loss_history = []
test_acc_history = []

total_step = len(train_dataloader)
for epoch in range(num_epochs):
    for i, (data, label) in enumerate(train_dataloader):
        train(model, train_dataloader, optimizer, criterion, i)
        train_loss, train_acc = evaluate(model, train_dataloader, criterion)
        test_loss, test_acc = evaluate(model, test_dataloader, criterion)
        train_loss_history.append(train_loss)
        test_loss_history.append(test_loss)
        test_acc_history.append(test_acc)
        print(f'Epoch {epoch}: Train Loss {train_loss:.6f}, Train Acc {train_acc:.2f}%, Test Loss {test_loss:.6f}, Test Acc {test_acc:.2f}%')
last_lose_train.append(train_loss_history[4])
last_lose_test.append(test_loss_history[4])

# print('Accuracy of the network on the 10000 test images: {} %'.format(
#     100 * correct / total))
