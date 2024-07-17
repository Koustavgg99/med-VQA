# -*- coding: utf-8 -*-
"""
Created on Thu Jun  9 13:30:55 2022

@author: vasud
"""
import torch
import torch.nn as nn
import torchvision.models as models
from transformers import BertModel
import initialization as init
import torch.utils.model_zoo as model_zoo
from resnet import resnet_encoders

encoders = {}
encoders.update(resnet_encoders)

def get_img_encoder(name, args, in_channels=3, depth=5):
    print("Image Encoder called")
    weights = args.weights # None by default which will load DiRA

    try:
        Encoder = encoders[name]["encoder"]
    except KeyError:
        raise KeyError("Wrong encoder name `{}`, supported encoders: {}".format(name, list(encoders.keys())))

    params = encoders[name]["params"]
    params.update(depth=depth)
    encoder = Encoder(**params)

    if weights is not None:
        print("Loading weights")
        if weights.lower() == 'random_init':
            print("Randomly initializing encoder {}.".format(name))
            init.initialize_encoder(encoder)

        elif weights.lower() != "imagenet":
            weight = args.weight #path to DiRA pre-trained model
            state_dict = torch.load(weight, map_location="cpu")

            if "state_dict" in state_dict:
                 state_dict = state_dict["state_dict"]
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
            state_dict = {k.replace("encoder.", ""): v for k, v in state_dict.items()}
            for k in list(state_dict.keys()):
                 if k.startswith('fc') or k.startswith('segmentation_head') or k.startswith('decoder') :
                      del state_dict[k]

            #msg = encoder.load_state_dict(state_dict, strict=False)
            print("=> loaded pre-trained model '{}'".format(weight))
            #print("missing keys:", msg.missing_keys)
            encoder.load_state_dict(state_dict, strict=False)
        else:
            print("Loading ImageNet weights")
            try:
                settings = encoders[name]["pretrained_settings"][weights.lower()]
            except KeyError:
                raise KeyError("Wrong pretrained weights `{}` for encoder `{}`. Available options are: {}".format(
                    weights, name, list(encoders[name]["pretrained_settings"].keys()),
                ))
            print ("settings url",settings["url"])
            if settings["url"].startswith("http"):
                encoder.load_state_dict(model_zoo.load_url(settings["url"]))
            else:
                encoder.load_state_dict(torch.load(settings["url"], map_location='cpu'))
            print("=> loaded supervised ImageNet pre-trained model")

    # encoder.set_in_channels(in_channels) implement this if channel size is 3 then kaiming initialization
    # code available at:
    # Kaiming initialization: https://github.com/fhaghighi/DiRA/blob/main/segmentation_models_pytorch/encoders/_utils.py
    # https://github.com/fhaghighi/DiRA/blob/main/segmentation_models_pytorch/encoders/_base.py

    return encoder
    


def get_img_encoder_names():
    return list(encoders.keys())


def get_img_preprocessing_params(encoder_name, pretrained="imagenet"):
    settings = encoders[encoder_name]["pretrained_settings"]

    if pretrained not in settings.keys():
        raise ValueError("Available pretrained options {}".format(settings.keys()))

    formatted_settings = {}
    formatted_settings["input_space"] = settings[pretrained].get("input_space")
    formatted_settings["input_range"] = settings[pretrained].get("input_range")
    formatted_settings["mean"] = settings[pretrained].get("mean")
    formatted_settings["std"] = settings[pretrained].get("std")
    return formatted_settings

    '''

def get_img_encoder(name, args):
    model = models.__dict__['resnet50']
    #weight = 'unet.pth'
    weight = args.weight
    state_dict = torch.load(weight, map_location='cpu')
    #print(state_dict.keys())
    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
    state_dict = {k.replace("encoder.", ""): v for k, v in state_dict.items()}

    for k in list(state_dict.keys()):
        if k.startswith('fc') or k.startswith('segmentation_head') or k.startswith('decoder'):
            print("deleting k: {}".format(k))
            del state_dict[k]

    msg = model.load_state_dict(state_dict, strict=False)
    #print(model)
    print("loaded pretrained model '{}'".format(weight))
    print("missing keys: ", msg.missing_keys)
    return model

'''

class GetBERTFeatures(nn.Module):
    def __init__(self, args):
        super(GetBERTFeatures, self).__init__()
        #self.bert_model = BertModel.from_pretrained('/scratch/19cs91r12/BERT_base_uncased/')
        self.bert_model = BertModel.from_pretrained(args.text_encoder_path)
        #self.bert_model = DistilBertModel.from_pretrained('D:\PhD\Project\Code\BERT_base_uncased\\')
        self.bert_model.eval()
        
    def forward(self, input_id, attention_mask):
        with torch.no_grad():
            hidden_layers = self.bert_model(input_id, attention_mask)
            layers = hidden_layers[2]
            question_embedding = (layers[-1]+layers[-2])/2
        return question_embedding
        


#model = models.__dict__['resnet50']
#x = torch.randn(1,3,224,224)
#print(model(x).shape)
#encoders = {}
        
        