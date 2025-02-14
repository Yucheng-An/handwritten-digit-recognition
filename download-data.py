import torch
import torchvision
import torchvision.transforms as transforms

# Define data preprocessing
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

#Download training data and test data
trainset = torchvision.datasets.MNIST(root="./data", train=True, download=False, transform=transform)
testset = torchvision.datasets.MNIST(root="./data", train=False, download=False, transform=transform)

#Create a data loader
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)
