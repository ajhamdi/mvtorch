# to import files from parent dir
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from mvtorch.data import ScanObjectNN, CustomDataLoader
from mvtorch.mvaggregate import MVAggregate
from mvtorch.view_selector import MVTN
from mvtorch.mvrenderer import MVRenderer

# Create dataset and dataloader
dset_train = ScanObjectNN(data_dir='./data/ScanObjectNN', split='train', nb_points=2500)
dset_val = ScanObjectNN(data_dir='./data/ScanObjectNN', split='test', nb_points=2500)
train_loader = CustomDataLoader(dset_train, batch_size=5, shuffle=True, drop_last=True)
test_loader = CustomDataLoader(dset_train, batch_size=5, shuffle=False, drop_last=False)

# Create backbone multi-view network (ResNet18)
mvnetwork = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
mvnetwork.fc = torch.nn.Sequential()
mvnetwork = MVAggregate(mvnetwork, agr_type="max", feat_dim=512, num_classes=len(dset_train.classes)).cuda()

# Create backbone optimizer
optimizer = torch.optim.AdamW(mvnetwork.parameters(), lr=0.00001, weight_decay=0.03)

# Create view selector
mvtn = MVTN(nb_views=1).cuda()

# Create optimizer for view selector (In case views are not fixed, otherwise set to None)
# mvtn_optimizer = torch.optim.AdamW(mvtn.parameters(), lr=0.0001, weight_decay=0.01)
mvtn_optimizer = None

# Create multi-view renderer
mvrenderer = MVRenderer(nb_views=1, return_mapping=False)

# Create loss function for training
criterion = torch.nn.CrossEntropyLoss()

epochs = 100
for epoch in range(epochs):
    print(f"\nEpoch {epoch + 1}/{epochs}")

    print("Training...")
    mvnetwork.train()
    mvtn.train()
    mvrenderer.train()
    running_loss = 0
    for i, (targets, meshes, points) in enumerate(train_loader):
        azim, elev, dist = mvtn(points, c_batch_size=len(targets))
        rendered_images, _ = mvrenderer(meshes, points, azim=azim, elev=elev, dist=dist)
        outputs = mvnetwork(rendered_images)[0]

        loss = criterion(outputs, targets.cuda())
        running_loss += loss.item()
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()
        if mvtn_optimizer is not None:
            mvtn_optimizer.step()
            mvtn_optimizer.zero_grad()
        
        if (i + 1) % int(len(train_loader) * 0.25) == 0:
            print(f"\tBatch {i + 1}/{len(train_loader)}: Current Average Training Loss = {(running_loss / (i + 1)):.5f}")
    print(f"Total Average Training Loss = {(running_loss / len(train_loader)):.5f}")

    print("Testing...")
    mvnetwork.eval()
    mvtn.eval()
    mvrenderer.eval()
    running_loss = 0
    for i, (targets, meshes, points) in enumerate(test_loader):
        with torch.no_grad():
            azim, elev, dist = mvtn(points, c_batch_size=len(targets))
            rendered_images, _ = mvrenderer(meshes, points, azim=azim, elev=elev, dist=dist)
            outputs = mvnetwork(rendered_images)[0]

            loss = criterion(outputs, targets.cuda())
            running_loss += loss.item()

            if (i + 1) % int(len(test_loader) * 0.25) == 0:
                print(f"\tBatch {i + 1}/{len(test_loader)}: Current Average Test Loss = {(running_loss / (i + 1)):.5f}")
    print(f"Total Average Test Loss = {(running_loss / len(test_loader)):.5f}")