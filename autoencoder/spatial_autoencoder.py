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

if not os.path.exists('./dc_img'):
    os.mkdir('./dc_img')


def to_img(x):
    x = x.view(1, 240, 240)
    return x


num_epochs = 50
batch_size = 128
learning_rate = 1e-3

# img_transform = transforms.Compose([transforms.ToTensor(),
#   transforms.Normalize((0.5,), (0.5,))
# ])


class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2),  # 3, 64, 117, 117
            nn.ReLU(True),
            nn.Conv2d(64, 32, 5, stride=1),  # 3, 64, 117, 117
            nn.ReLU(True),
            nn.Conv2d(32, 16, 5, stride=1),  # b, 8, 3, 3
            nn.ReLU(True),
            SpatialSoftmax(109, 109, 16, temperature=None)

        )


        self.decoder = nn.Sequential(
            nn.Linear(32, 57600),
            nn.ReLU(True)
        )


    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


model = autoencoder().cuda()

data_path = "~/Dropbox/ML-Projects/Methods/autoencoder"
train_dataset = torchvision.datasets.ImageFolder(
    root=data_path,
    transform=torchvision.transforms.ToTensor()
)

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=1,
    num_workers=0,
    shuffle=True
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

transformations = transforms.Compose([transforms.ToPILImage(),
                                          transforms.Grayscale(num_output_channels=1),
                                          transforms.ToTensor()])

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                             weight_decay=1e-5)

for epoch in range(num_epochs):
    for data in train_loader:
        img, _ = data
        img = Variable(img).cuda()
        # ===================forward=====================
        output = model(img)

       
        grey_tensor = transformations(img.cpu().view(3, 240, 240))


        # print(output.view(1, 60, 60).shape, grey_tensor.shape)
        loss = criterion(output.view(1, 240, 240), grey_tensor.to(device))
# #         # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
# #     # ===================log========================
    print('epoch [{}/{}], loss:{:.4f}'
          .format(epoch+1, num_epochs, loss.item()))
    if epoch % 10 == 0 and epoch >0:
        pic = to_img(output.cpu().data)
        save_image(pic, './dc_img/image_{}.png'.format(epoch))

torch.save(model.state_dict(), './conv_autoencoder.pth')


print("End")