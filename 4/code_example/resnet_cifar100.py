import torch
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
import time




# We provide the code for loading CIFAR100 data
num_epochs = 1
batch_size = 128
# torch.manual_seed(0)
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


trainset = torchvision.datasets.CIFAR100(root='~/scratch/', train=True, download=False, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=8)

testset = torchvision.datasets.CIFAR100(root='~/scratch/', train=False, download=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=8)




device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# YOUR CODE GOES HERE
# Change to your ResNet
model = torch.nn.Module()
for epoch in range(num_epochs):
    # Train the model
    model.train()
    for batch_idx, (X_train_batch, Y_train_batch) in enumerate(trainloader):
        X_train_batch,Y_train_batch = X_train_batch.to(device),Y_train_batch.to(device)
       
       
    # Test the model
    with torch.no_grad():
        model.eval()
        for batch_idx, (X_test_batch, Y_test_batch) in enumerate(testloader):
            X_test_batch, Y_test_batch= X_test_batch.to(device),Y_test_batch.to(device)
           

   
