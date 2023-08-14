
print("==> start retraining the model with clean and poisoned data")

# define the dataloader to load the clean and poisoned data

data_p = torch.load('poisoned_models/data_p_{}.pt'.format(epsilon)).to('cpu')
target_p = torch.load('poisoned_models/target_p_{}.pt'.format(epsilon)).to('cpu')

class PoisonedDataset(Dataset):
    def __init__(self, X, y):
        assert X.size()[0] == y.size()[0]
        self.X = X
        self.y = y
    
    def __len__(self):
        return self.X.size()[0]
    
    def __getitem__(self, idx):
        return [self.X[idx], int(self.y[idx])]
    
dataset_p = PoisonedDataset(data_p,target_p)  
dataset_total = torch.utils.data.ConcatDataset([dataset1, dataset_p])
train_loader_retrain = torch.utils.data.DataLoader(dataset_total, batch_size=128,shuffle=True)
test_loader_retrain = torch.utils.data.DataLoader(dataset2,batch_size=1000)  

model1 = ModelBase(feature_dim=512, arch='resnet18').to(device)
    

# freeze all layers but the last fc
for name, param in model.named_parameters():
    if name not in ['fc.weight', 'fc.bias']:
        param.requires_grad = False
# init the fc layer
model.fc.weight.data.normal_(mean=0.0, std=0.01)
model.fc.bias.data.zero_()

print("=> loading checkpoint '{}'".format('GradPC/models/model_last.pth'))
checkpoint = torch.load('GradPC/models/model_last.pth', map_location="cpu")

# rename moco pre-trained keys
state_dict = checkpoint['state_dict']
for k in list(state_dict.keys()):
    # retain only encoder_q up to before the embedding layer
    if k.startswith('encoder_q') and not k.startswith('encoder_q.fc'):
        # remove prefix
        state_dict[k[len("encoder_q."):]] = state_dict[k]
    # delete renamed or unused k
    del state_dict[k]

msg = model.load_state_dict(state_dict, strict=False)
print(msg)
assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}

print("=> loaded pre-trained model '{}'".format('GradPC/models/model_last.pth'))

optimizer1 = optim.SGD(model.parameters(), lr=0.1,momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

# normal training on D_c \cup D_p

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in tqdm(enumerate(train_loader)):
        criterion = nn.CrossEntropyLoss()
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 200 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            criterion = nn.CrossEntropyLoss()
            data, target = data.to(device), target.to(device)
            #output = model(data.view(data.size(0), -1))
            output = model(data)
            test_loss += criterion(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    
for epoch in range(100):
    train(model, device, train_loader_retrain, optimizer1, epoch)
    test(model, device, test_loader_retrain)
    scheduler.step()

