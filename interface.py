import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import os
import torch.nn as nn
import torch.nn.functional as F
import timm
import torchvision.models as models
import PIL
import cv2
import tkinter as tk
from tkinter import filedialog
from tkinter.filedialog import askopenfile
import matplotlib.pyplot as plt
from PIL import Image, ImageTk
from tkinter import *


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def upload_file():
    for widget in frame.winfo_children():
        widget.destroy()
    f_types = [('Png Files', '*.png'), ('Jpg Files', '*.jpg')]
    filename = filedialog.askopenfilename(filetypes=f_types)
    img=PIL.Image.open(filename)
    get_probabilities(filename)


def eucliedean_dist(img_enc, anc_enc_arr):
    dist = np.sqrt(np.dot(img_enc-anc_enc_arr, (img_enc-anc_enc_arr).T))
    return dist


def get_probabilities(image):
    
    global title
    
    canvas.itemconfig(title, text='')
    
    img = cv2.imread(image)
    img = cv2.resize(img, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
    img = torch.from_numpy(img).permute(2, 0, 1)/255.0
    store = img
    
    model.eval()
    with torch.no_grad():
        img = img.to(device)
        img_enc = model(img.unsqueeze(0)).to(device)
        img_enc = img_enc.detach().cpu().numpy()
        
    anc_enc_arr = df_enc.iloc[:, 1:513].to_numpy()
    anc_img_names = df_enc['Anchor']
    
    distance = []
    
    for i in range(anc_enc_arr.shape[0]):
        dist = eucliedean_dist(img_enc, anc_enc_arr[i : i+1, :])
        
        distance = np.append(distance, dist)
        
    closest_idx = np.argsort(distance)
    
    y_pred_label = []
    S_name = []
    
    for s in range(1):
        S_name.append(anc_img_names.iloc[closest_idx[s]])
        
    wanted_label = list(set(S_name))
    
    res_name = wanted_label[0]
    
    temp2 = np.unique(whole_df[whole_df['Anchor']==res_name]['Label'])

    y_pred_label.append(temp2[0])
    
    similar_enc = np.array(df_enc[df_enc['Anchor']==res_name])
    similar_enc = np.array(similar_enc[:, 1:513])
    similar_enc = np.vstack(similar_enc).astype(float)
    
    img_enc = torch.from_numpy(img_enc)
    similar_enc = torch.from_numpy(similar_enc)

    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    
    similarity = cos(img_enc, similar_enc)
    
    img2 = cv2.imread('train/' + res_name)
    img2 = cv2.resize(img2, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
    img2 = torch.from_numpy(img2).permute(2, 0, 1)/255.0
    
    result_label = f'Predicted Condition: {y_pred_label[0]}, Similarity: {(similarity.item())*100:.4f}%'
   
    img2 = img2.numpy()
    plt.imsave('result.png', np.transpose(img2, (1, 2, 0)))
    
    canvas.create_text(
    600.0,
    70.0,
    anchor="nw",
    text="Similar Image",
    fill="#FCFCFC",
    font=("Roboto Bold", 18 * -1))
    
    
    title = canvas.create_text(
    224.0,
    570.0,
    anchor="nw",
    text=result_label,
    fill="#FCFCFC",
    font=("Roboto Bold", 24 * -1))

    
    frame2 = Frame(window, width=327, height=327, bg="white", colormap="new")
    frame2.pack()
    frame2.place(x=600, y=100)
    
    
    img=PIL.Image.open(image)
    img=img.resize((327, 327))
    img=ImageTk.PhotoImage(img)
    label = Label(frame, image = img)
    label.pack()
    
    sim_img=PIL.Image.open('2.png')
    sim_img=sim_img.resize((327, 327))
    sim_img=ImageTk.PhotoImage(sim_img)
    label2 = Label(frame2, image = sim_img)
    label2.pack()
    
    window.mainloop()


feature_extract = True

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

class DenseNet121(nn.Module):
    def __init__(self, emb_size=512):
        super(DenseNet121, self).__init__()
        self.network = models.densenet121(pretrained=True, progress = True)
        set_parameter_requires_grad(self.network, feature_extract)
        self.network.classifier = nn.Linear(self.network.classifier.in_features, out_features=emb_size)
    
    def forward(self, images):
        embeddings = self.network(images)
        return embeddings
    
##########################################################################################################################
class SwinTransformer(nn.Module):
    def __init__(self, emb_size=512):
        super(SwinTransformer, self).__init__()
        self.network = timm.create_model('swin_base_patch4_window7_224_in22k', pretrained=True, progress=True)
        set_parameter_requires_grad(self.network, feature_extract)
        self.network.head = nn.Linear(self.network.head.in_features, out_features=emb_size)
        
    def forward(self, images):
        embeddings = self.network(images)
        return embeddings


##########################################################################################################################
class MobileNetV2(nn.Module):
    def __init__(self, emb_size=512):
        super(MobileNetV2, self).__init__()
        self.network = models.mobilenet_v2(pretrained=True)
        set_parameter_requires_grad(self.network, feature_extract)
        self.network.classifier[1] = nn.Linear(self.network.classifier[1].in_features, out_features=emb_size)
        
    def forward(self, images):
        embeddings = self.network(images)
        return embeddings
    
##########################################################################################################################
class EfficientNetB0(nn.Module):
    def __init__(self, emb_size=512):
        super(EfficientNetB0, self).__init__()
        self.network = models.efficientnet_b0(pretrained = True)
        set_parameter_requires_grad(self.network, feature_extract)
        self.network.classifier[1] = nn.Linear(in_features=self.network.classifier[1].in_features, out_features = emb_size)
        
    def forward(self, images):
        embeddings = self.network(images)
        return embeddings    


##########################################################################################################################    
class Resnext101_32x8d(nn.Module):
    def __init__(self, emb_size=512):
        super(Resnext101_32x8d, self).__init__()
        self.network =  models.resnext101_32x8d(pretrained = True, progress = True)
        set_parameter_requires_grad(self.network, feature_extract)
        self.network.fc = nn.Linear(in_features=self.network.fc.in_features, out_features = emb_size)
        
    def forward(self, images):
        embeddings = self.network(images)
        return embeddings

class Bit(nn.Module):
    def __init__(self, emb_size=512):
        super(Bit, self).__init__()
        self.network =  timm.create_model('resnetv2_101x1_bitm', pretrained=True)
        set_parameter_requires_grad(self.network, feature_extract)
        self.network.head.fc = nn.Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, images):
        embeddings = self.network(images)
        return embeddings    

##########################################################################################################################
class Ensemble(nn.Module):
    def __init__(self, Bit, VGG16, SwinTransformer, MobileNetV2, EfficientNetB0, Resnext101_32x8d):
        super(Ensemble, self).__init__()
        
        self.Bit = Bit
        self.VGG16 = VGG16
        self.SwinTransformer = SwinTransformer
        self.MobileNetV2 = MobileNetV2
        self.EfficientNetB0 = EfficientNetB0
        self.Resnext101_32x8d = Resnext101_32x8d
        self.classifier = nn.Linear(512*6, 512)
        
    def forward(self, y):
        x0 = self.Bit(y.clone())
        x0 = x0.view(x0.size(0), -1)
        
        x1 = self.VGG16(y)
        x1 = x1.view(x1.size(0), -1)
        
        x2 = self.SwinTransformer(y)
        x2 = x2.view(x2.size(0), -1)
        
        x4 = self.MobileNetV2(y)
        x4 = x4.view(x4.size(0), -1)
        
        x5 = self.EfficientNetB0(y)
        x5 = x5.view(x5.size(0), -1)

        x8 = self.Resnext101_32x8d(y)
        x8 = x8.view(x8.size(0), -1)
        
        x = torch.cat((x0, x1, x2, x4, x5, x8), dim=1)

        x = F.relu(self.classifier(x))
        return x

SwinTransformer = SwinTransformer().to(device) 
MobileNetV2 = MobileNetV2().to(device)
EfficientNetB0 = EfficientNetB0().to(device) 
Resnext101_32x8d = Resnext101_32x8d().to(device)
Bit = Bit().to(device)
DenseNet121 = DenseNet121().to(device)

model = Ensemble(Bit, DenseNet121, SwinTransformer, MobileNetV2, EfficientNetB0, Resnext101_32x8d)
model.to(device)
model.load_state_dict(torch.load('few-shot-final_all.pt'))

df_enc = pd.read_csv('proposed_model_encoding.csv')

whole_df = pd.read_csv('full_dataset-2.csv')


window = Tk()

window.geometry("1396x818")
window.configure(bg = "#3A7FF6")

canvas = Canvas(
    window,
    bg = "#3A7FF6",
    height = 818,
    width = 1396,
    bd = 0,
    highlightthickness = 0,
    relief = "ridge"
)

result_label = ''

canvas.place(x = 0, y = 0)

canvas.create_rectangle(
    1047.0,
    0.0,
    1396.0,
    818.0,
    fill="#FCFCFC",
    outline="")

canvas.create_text(
    45.0,
    16.0,
    anchor="nw",
    text="Covid-19, CAP and NonCOVID classification",
    fill="#FCFCFC",
    font=("Roboto Bold", 24 * -1)
)

canvas.create_text(
    1070.0,
    37.0,
    anchor="nw",
    text="Upload the image to classify",
    fill="#505485",
    font=("Roboto Bold", 24 * -1)
)

canvas.create_text(
    45.0,
    70.0,
    anchor="nw",
    text="Input Image",
    fill="#FCFCFC",
    font=("Roboto Bold", 18 * -1)
)

frame = Frame(window, width=327, height=327, bg="white", colormap="new")
frame.pack()
frame.place(x=45, y=100)

title = canvas.create_text(
    224.0,
    570.0,
    anchor="nw",
    text=result_label,
    fill="#FCFCFC",
    font=("Roboto Bold", 24 * -1))

button_image_1 = PhotoImage(file='./assets/button_1.png')
button_1 = Button(
    image=button_image_1,
    borderwidth=0,
    highlightthickness=0,
    command=lambda: upload_file(),
    relief="flat"
)
button_1.place(
    x=1104.0,
    y=248.0,
    width=236.0,
    height=77.0
)

window.mainloop()

