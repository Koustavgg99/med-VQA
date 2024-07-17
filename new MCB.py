import torch
import torch.nn as nn
from torch.nn import AvgPool2d
import torch.nn.functional as F
import torchvision.models as models
class MCB_CoAtt(nn.Module):
    def __init__(self, embedding_size, LSTM_units, LSTM_layers, feat_size,
                 batch_size, ans_vocab_size, global_avg_pool_size, 
                 dropout=0.3, MCB_output_dim=5000):
        super(MCB_CoAtt, self).__init__()
        self.batch_size = batch_size
        self.ans_vocab_size = ans_vocab_size
        self.MCB_output_dim = MCB_output_dim
        self.feat_size = feat_size
        self.MCB_out = 1000
        self.MCB_factor = 5
        self.channel_size = global_avg_pool_size
        self.num_ques_glimpse = 2
        self.num_img_glimpse = 2
        
        # Initialize ResNet as the image encoder
        resnet = models.resnet50(pretrained=True)
        self.resnet_features = nn.Sequential(*list(resnet.children())[:-2])
        
        self.LSTM = nn.LSTM(input_size=embedding_size, hidden_size=LSTM_units, 
                            num_layers=LSTM_layers, batch_first=False)
        self.pool2d = AvgPool2d(global_avg_pool_size, stride=1)
        self.Dropout_L = nn.Dropout(p=dropout)
        self.Dropout_M = nn.Dropout(p=dropout)
        self.Softmax = nn.Softmax(dim=1)
        
        # Linear layers for feature projection
        self.Linear1_q_proj = nn.Linear(LSTM_units*self.num_ques_glimpse, self.MCB_output_dim)
        self.Conv_i_proj = nn.Conv2d(2048, self.MCB_output_dim, 1)
        self.Linear_img_proj = nn.Linear(self.feat_size*self.num_img_glimpse, self.MCB_output_dim)
        
        self.Conv1_Qatt = nn.Conv2d(LSTM_units, 512, 1)
        self.Conv2_Qatt = nn.Conv2d(512, self.num_ques_glimpse, 1)
        self.Conv1_Iatt = nn.Conv2d(1000, 512, 1)
        self.Conv2_Iatt = nn.Conv2d(512, self.num_img_glimpse, 1)

        self.Linear_predict = nn.Linear(self.MCB_out, self.ans_vocab_size)
        
        self.qatt_maps = None
        self.iatt_maps = None
        
    def forward(self, ques_embed, img_feat):
        # Extract image features using ResNet
        img_feat = self.resnet_features(img_feat)
        img_feat_resh = img_feat.permute(0, 2, 3, 1).contiguous()  # N x w x w x 2048
        img_feat_resh = img_feat_resh.view(img_feat_resh.size(0), -1, img_feat_resh.size(3))  # N x w*w x 2048
        
        # Question LSTM processing
        ques_embed_resh = ques_embed.permute(1, 0, 2).contiguous()  # T x N x embedding_size
        lstm_out, (hn, cn) = self.LSTM(ques_embed_resh)
        lstm1_droped = self.Dropout_L(lstm_out)
        lstm1_resh = lstm1_droped.permute(1, 2, 0).contiguous()  # N x 1024 x T
        lstm1_resh2 = torch.unsqueeze(lstm1_resh, 3)  # N x 1024 x T x 1
        
        '''
        Question Attention
        '''        
        qatt_conv1 = self.Conv1_Qatt(lstm1_resh2)  # N x 512 x T x 1
        qatt_relu = F.relu(qatt_conv1)
        qatt_conv2 = self.Conv2_Qatt(qatt_relu)  # N x 2 x T x 1
        qatt_conv2 = qatt_conv2.reshape(qatt_conv2.shape[0]*self.num_ques_glimpse, -1)
        qatt_softmax = self.Softmax(qatt_conv2)
        qatt_softmax = qatt_softmax.view(qatt_conv1.shape[0], self.num_ques_glimpse, -1, 1)
        self.qatt_maps = qatt_softmax
        qatt_feature_list = []
        for i in range(self.num_ques_glimpse):
            t_qatt_mask = qatt_softmax.narrow(1, i, 1)  # N x 1 x T x 1
            t_qatt_mask = t_qatt_mask * lstm1_resh2  # N x 1024 x T x 1
            t_qatt_mask = torch.sum(t_qatt_mask, 2, keepdim=True)  # N x 1024 x 1 x 1
            qatt_feature_list.append(t_qatt_mask)
        qatt_feature_concat = torch.cat(qatt_feature_list, 1)  # N x 2048 x 1 x 1
        
        '''
        Image Attention
        '''
        q_feat_resh = torch.squeeze(qatt_feature_concat)  # N x 2048
        i_feat_resh = img_feat_resh.permute(0, 2, 1).contiguous()  # N x 2048 x (w*w)
        i_feat_resh2 = torch.unsqueeze(i_feat_resh, 3)  # N x 2048 x (w*w) x 1
        iq_feat_resh = torch.unsqueeze(q_feat_resh, 2)  # N x 2048 x 1
        iq_feat_resh2 = torch.unsqueeze(iq_feat_resh, 3)  # N x 2048 x 1 x 1
        iq_iatt_mul = torch.matmul(i_feat_resh2, iq_feat_resh2.permute(0, 1, 3, 2))  # N x 2048 x (w*w) x 2048
        iq_iatt_relu = F.relu(iq_iatt_mul)
        iq_iatt_conv1 = self.Conv1_Iatt(iq_iatt_relu)  # N x 512 x (w*w) x 2048
        iq_iatt_conv2 = self.Conv2_Iatt(iq_iatt_conv1)  # N x 2 x (w*w) x 2048
        iq_iatt_conv2 = iq_iatt_conv2.reshape(iq_iatt_conv2.shape[0]*self.num_img_glimpse, -1)
        iq_iatt_softmax = self.Softmax(iq_iatt_conv2)
        iq_iatt_softmax = iq_iatt_softmax.view(iq_iatt_conv1.shape[0], self.num_img_glimpse, -1, 2048)
        self.iatt_maps = iq_iatt_softmax
        iatt_feature_list = []
        for i in range(self.num_img_glimpse):
            t_iatt_mask = iq_iatt_softmax.narrow(1, i, 1)  # N x 1 x (w*w) x 2048
            t_iatt_mask = t_iatt_mask * i_feat_resh  # N x 2048 x (w*w)
            t_iatt_mask = torch.sum(t_iatt_mask, 2, keepdim=True)  # N x 2048 x 1
            iatt_feature_list.append(t_iatt_mask)
        iatt_feature_concat = torch.cat(iatt_feature_list, 1)  # N x 4096 x 1
        iatt_feature_concat = iatt_feature_concat.view(-1, self.feat_size*self.num_img_glimpse)  # N x (feat_size*num_img_glimpse)
        img_proj = self.Linear_img_proj(iatt_feature_concat)  # N x 5000
        
        '''
        Bilinear pooling
        '''
        iq_feat = img_proj * self.Linear1_q_proj(torch.squeeze(qatt_feature_concat))  # N x 5000
        iq_resh = iq_feat.view(-1, 1, self.MCB_out, self.MCB_factor)  # N x 1 x 1000 x 5
        iq_sumpool = torch.sum(iq_resh, 3)  # N x 1 x 1000 x 1
        iq_sumpool = torch.squeeze(iq_sumpool)  # N x 1000
        
        iq_sqrt = torch.sqrt(F.relu(iq_sumpool)) - torch.sqrt(F.relu(-iq_sumpool))
        iq_norm = F.normalize(iq_sqrt)
        pred = self.Linear_predict(iq_norm)
        pred = self.Softmax(pred)
        return pred

def build_mcb(dataset, args):
    # Extract dataset parameters
    embedding_size = args.embedding_size
    LSTM_units = args.LSTM_units
    LSTM_layers = args.LSTM_layers
    feat_size = args.feat_size
    batch_size = args.batch_size
    ans_vocab_size = dataset.num_ans_candidates
    global_avg_pool_size = args.global_avg_pool_size
    dropout = args.dropout
    MCB_output_dim = args.MCB_output_dim
    
    # Determine which model to use
  
    if args.model_type == 'MCB_CoAtt':
        model = MCB_CoAtt(embedding_size, LSTM_units, LSTM_layers, feat_size,
                          batch_size, ans_vocab_size, global_avg_pool_size, 
                          dropout, MCB_output_dim)
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")
    
    return model