# __author__ = 'SherlockLiao'
# Link: https://github.com/L1aoXingyu/pytorch-beginner/blob/master/08-AutoEncoder/conv_autoencoder.py

import torch
import torchvision
from torchsummary import summary
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from torchvision.datasets import MNIST
import os
from spatial_softmax import SpatialSoftmax

if not os.path.exists('dc_img'):
    os.mkdir('dc_img')

num_epochs = 50
batch_size = 16
learning_rate = 1e-3

class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2),  # 3, 64, 117, 117
            nn.ReLU(True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 32, 5, stride=1),  # 3, 64, 117, 117
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 16, 5, stride=1),  # b, 8, 3, 3
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            SpatialSoftmax(109, 109, 16, temperature=None)
        )


        self.decoder = nn.Sequential(
            nn.Linear(32, 3600),
            nn.Sigmoid()
            )


    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)

        return x


model = autoencoder().cuda()

data_path = "images"
train_dataset = torchvision.datasets.ImageFolder(
    root=data_path,
    transform=torchvision.transforms.ToTensor()
)

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batch_size,
    num_workers=0,
    shuffle=True
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

transformations = transforms.Compose([transforms.ToPILImage(), transforms.Resize((60,60)),
                                          transforms.Grayscale(num_output_channels=1),
                                          transforms.ToTensor()])

criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate,
                             momentum=0.9)

for epoch in range(num_epochs):
    for data in train_loader:
        img, _ = data
        img = Variable(img).cuda()
        # print(img.shape)
        # ===================forward=====================
        output = model(img)

        grey_out_image = torch.zeros([batch_size, 1, 60, 60], dtype=torch.float)
        cnt = 0
        for im in img:
            grey_tensor = transformations(im.cpu().view(3, 240, 240))
            grey_out_image[cnt] = grey_tensor
            cnt += 1

        loss = criterion(output.view(-1, 1, 60, 60), grey_out_image.to(device))
        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # ===================log========================
    print('epoch [{}/{}], loss:{:.4f}'
          .format(epoch+1, num_epochs, loss.item()))
    if epoch % 1 == 0:
        pic = output[0].cpu().data.view(1, 60, 60)
        save_image(pic, './dc_img/image_{}.png'.format(epoch))

torch.save(model.state_dict(), './conv_autoencoder.pth')


print("End")
