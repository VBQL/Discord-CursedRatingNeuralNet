import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import discord
import requests

device = torch.device("cuda:0")

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 16)
        self.conv2 = nn.Conv2d(32, 64, 16)
        self.conv3 = nn.Conv2d(64, 128, 16)
        
        x = torch.randn(256,256).view(-1,1,256,256)
        self._to_linear = None
        
        self.convs(x)
        
        self.fc1 = nn.Linear(self._to_linear, 512)

        self.fc2 = nn.Linear(512, 256)
        
        self.fc3 = nn.Linear(256, 2) 
    
    # Does pyTorch have flatten yet??
    def convs(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (3,3)) # Shape of pooling will be 3 by 3
        x = F.max_pool2d(F.relu(self.conv2(x)), (3,3))
        x = F.max_pool2d(F.relu(self.conv3(x)), (3,3))
        
        if self._to_linear is None:
            self._to_linear = x[0].shape[0] * x[0].shape[1] * x[0].shape[2]
            
        return x
    
    def forward(self,x):
        x = self.convs(x)
        x = x.view(-1, self._to_linear)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.softmax(x, dim = 1) # Dimension is 1 due to batches

model = torch.load("model_final_256_256").to(device)

def curseDetermine(link):
    resp = requests.get(link.strip(), stream=True, headers={'User-agent': 'Mozilla/5.0'}).raw
    img = np.asarray(bytearray(resp.read()), dtype="uint8")
    img = cv2.imdecode(img, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (256, 256))

    target = []
    target.append([np.array(img)])
    x = torch.Tensor([i[0] for i in target])

    output = model(x[0].view(-1,1,256,256).to(device))[0]
    blessed = float(output[1]) * 100
    cursed = float(output[0]) * 100


    return ("I am **" + np.format_float_positional(cursed, trim='-') + "%** confident that this image is **cursed** and **" + np.format_float_positional(blessed, trim='-') + "%** confident that this image is **blessed**")

class curseNet(discord.Client):
    async def on_ready(self):
        print('Logged in as')
        print(self.user.name)
        print(self.user.id)
        print('------')

    async def on_message(self, message):

        if message.author.id == self.user.id:
            return

        if message.content.startswith('!howcursed'):
            try:
                if len(message.content) > 10:
                    result = curseDetermine(message.content[10:])
                else:
                    result = curseDetermine(message.attachments[0].url)
                await message.channel.send(result.format(message))
            except:
                await message.channel.send('Unable to process image {0.author.mention}'.format(message))



client = curseNet()
client.run("[ENTER API]")
