import os
from PIL import Image
import torch
import gradio as gr
os.system("pip install gradio==2.5.3")

model2 = torch.hub.load(
    "AK391/animegan2-pytorch:main",
    "generator",
    pretrained=True,
    device="cuda",
    progress=False
)


model1 = torch.hub.load("AK391/animegan2-pytorch:main", "generator", pretrained="face_paint_512_v1",  device="cuda")
face2paint = torch.hub.load(
    'AK391/animegan2-pytorch:main', 'face2paint', 
    size=512, device="cuda",side_by_side=False
)
def inference(img, ver):
    if ver == 'version 2 (🔺 robustness,🔻 stylization)':
        out = face2paint(model2, img)
    else:
        out = face2paint(model1, img)
    return out
  
title = "AnimeGANv2"
description = "Gradio Demo for AnimeGanv2 Face Portrait. To use it, simply upload your image, or click one of the examples to load them. Read more at the links below. Please use a cropped portrait picture for best results similar to the examples below."
article = "<p style='text-align: center'><a href='https://github.com/bryandlee/animegan2-pytorch' target='_blank'>Github Repo Pytorch</a></p> <center><img src='https://visitor-badge.glitch.me/badge?page_id=akhaliq_animegan' alt='visitor badge'></center> <p style='text-align: center'>samples from repo: <img src='https://user-images.githubusercontent.com/26464535/129888683-98bb6283-7bb8-4d1a-a04a-e795f5858dcf.gif' alt='animation'/> <img src='https://user-images.githubusercontent.com/26464535/137619176-59620b59-4e20-4d98-9559-a424f86b7f24.jpg' alt='animation'/><img src='https://user-images.githubusercontent.com/26464535/127134790-93595da2-4f8b-4aca-a9d7-98699c5e6914.jpg' alt='animation'/></p>"
examples=[['groot.jpeg','version 2 (🔺 robustness,🔻 stylization)'],['bill.png','version 1 (🔺 stylization, 🔻 robustness)'],['tony.png','version 1 (🔺 stylization, 🔻 robustness)'],['elon.png','version 2 (🔺 robustness,🔻 stylization)'],['IU.png','version 1 (🔺 stylization, 🔻 robustness)'],['billie.png','version 2 (🔺 robustness,🔻 stylization)'],['will.png','version 2 (🔺 robustness,🔻 stylization)'],['beyonce.png','version 1 (🔺 stylization, 🔻 robustness)'],['gongyoo.jpeg','version 1 (🔺 stylization, 🔻 robustness)']]
gr.Interface(inference, [gr.inputs.Image(type="pil"),gr.inputs.Radio(['version 1 (🔺 stylization, 🔻 robustness)','version 2 (🔺 robustness,🔻 stylization)'], type="value", default='version 2 (🔺 robustness,🔻 stylization)', label='version')
], gr.outputs.Image(type="pil"),title=title,description=description,article=article,examples=examples,allow_flagging=False,enable_queue=True).launch()