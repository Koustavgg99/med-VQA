
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class MutanFusion(nn.Module):
    def __init__(self, input_dim=1024, out_dim=5000, num_layers=5):
        super(MutanFusion, self).__init__()
        self.input_dim = input_dim
        self.out_dim = out_dim
        self.num_layers = num_layers

        hv = []
        for i in range(self.num_layers):
            do = nn.Dropout(p=0.5)
            lin = nn.Linear(input_dim, out_dim)
            hv.append(nn.Sequential(do, lin, nn.Tanh()))

        self.image_transformation_layers = nn.ModuleList(hv)

        hq = []
        for i in range(self.num_layers):
            do = nn.Dropout(p=0.5)
            lin = nn.Linear(input_dim, out_dim)
            hq.append(nn.Sequential(do, lin, nn.Tanh()))

        self.ques_transformation_layers = nn.ModuleList(hq)

    def forward(self, ques_emb, img_emb):
        batch_size = img_emb.size()[0]
        x_mm = []
        for i in range(self.num_layers):
            x_hv = img_emb
            x_hv = self.image_transformation_layers[i](x_hv)
            
            x_hq = ques_emb
            x_hq = self.ques_transformation_layers[i](x_hq)
            x_mm.append(torch.mul(x_hq, x_hv))

        x_mm = torch.stack(x_mm, dim=1)
        x_mm = x_mm.sum(1).view(batch_size, self.out_dim)
        x_mm = torch.tanh(x_mm)
        return x_mm

class MUTAN_baseline(nn.Module):
    def __init__(self, embedding_size, LSTM_units, LSTM_layers, feat_size,
                 batch_size, ans_vocab_size, global_avg_pool_size, 
                 dropout=0.3, mutan_output_dim=5000):
        super(MUTAN_baseline, self).__init__()
        self.batch_size = batch_size
        self.ans_vocab_size = ans_vocab_size
        self.mutan_output_dim = mutan_output_dim
        self.feat_size = feat_size
        self.mutan_out = 1000

        # Load pretrained ResNet model
        self.image_encoder = models.resnet50(pretrained=True)
        self.image_encoder = nn.Sequential(*list(self.image_encoder.children())[:-2]) # Remove the last two layers
        
        self.mutan = MutanFusion(LSTM_units, self.mutan_out)
        self.LSTM = nn.LSTM(input_size=embedding_size, hidden_size=LSTM_units, 
                            num_layers=LSTM_layers, batch_first=False)
        self.pool2d = nn.AdaptiveAvgPool2d((1, 1))
        self.Dropout = nn.Dropout(p=dropout)
        self.Linear_predict = nn.Linear(self.mutan_out, ans_vocab_size)
        self.Softmax = nn.Softmax(dim=1)

    def forward(self, ques_embed, img):
        # Extract image features using pretrained ResNet
        img_feat = self.image_encoder(img)
        img_feat_pooled = self.pool2d(img_feat)
        img_feat_sq = img_feat_pooled.view(img_feat_pooled.size(0), -1) # Flatten

        # ques_embed                                         N x T x embedding_size
        ques_embed_resh = ques_embed.permute(1, 0, 2)       # T x N x embedding_size
        lstm_out, (hn, cn) = self.LSTM(ques_embed_resh)
        ques_lstm = lstm_out[-1]                            # N x lstm_units
        ques_lstm = self.Dropout(ques_lstm)

        iq_feat = self.mutan(img_feat_sq, ques_lstm)

        iq_sqrt = torch.sqrt(F.relu(iq_feat)) - torch.sqrt(F.relu(-iq_feat))
        iq_norm = F.normalize(iq_sqrt)

        pred = self.Linear_predict(iq_norm)
        pred = self.Softmax(pred)

        return pred
