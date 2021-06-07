from custom_dataset import CustomImageDataset
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from engine import train_one_epoch, evaluate

# train on the GPU or on the CPU, if a GPU is not available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

##DATA Loder
dataset = CustomImageDataset(
    r"D:\Eren\Python Scripts\pytorch_tutorials\kangaroo-master\annots",
    r"D:\Eren\Python Scripts\pytorch_tutorials\kangaroo-master\images"
    )

data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

"""
#Pretrained model RCNN Fine tunning
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
num_classes = 2  # 1 class (person) + background
in_features = model.roi_heads.box_predictor.cls_score.in_features
# replace the pre-trained head with a new one
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
"""

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.to(device)

# construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005,
                            momentum=0.9, weight_decay=0.0005)
# and a learning rate scheduler
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=3,
                                                gamma=0.1)
# let's train it for 10 epochs
num_epochs = 10
for epoch in range(num_epochs):
    # train for one epoch, printing every 10 iterations
    train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
    # update the learning rate
    lr_scheduler.step()
    # evaluate on the test dataset
    evaluate(model, data_loader_test, device=device)

print("That's it!")
"""
# For Training
images,targets = next(iter(data_loader))
images = list(image for image in images)
images[0] = images[0].reshape((3,images[0].shape[0],images[0].shape[1]))

a = [t for t in targets]
#targets = [{k: v for k, v in t.items()} for t in targets]
targets = [{k: v for k, v in targets.items()}]
targets[0]["boxes"] = targets[0]["boxes"].reshape((-1,4))
targets[0]["labels"] = targets[0]["labels"].reshape((1))

output = model(images,targets)   # Returns losses and detections
# For inference
model.eval()
x = [torch.rand(3, 300, 400)]
predictions = model(x)  
print(predictions)
"""