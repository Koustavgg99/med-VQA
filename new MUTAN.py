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

class Mutan_CoAtt(nn.Module):
    def __init__(self, embedding_size, LSTM_units, LSTM_layers, feat_size,
                 batch_size, ans_vocab_size, global_avg_pool_size, 
                 dropout=0.3, Mutan_output_dim=5000):
        super(Mutan_CoAtt, self).__init__()
        self.batch_size = batch_size
        self.ans_vocab_size = ans_vocab_size
        self.Mutan_output_dim = Mutan_output_dim
        self.feat_size = feat_size
        self.Mutan_out = 1000
        self.Mutan_factor = 5
        self.channel_size = global_avg_pool_size
        self.num_ques_glimpse = 2
        self.num_img_glimpse = 2

        # Load pretrained ResNet model
        self.image_encoder = models.resnet50(pretrained=True)
        self.image_encoder = nn.Sequential(*list(self.image_encoder.children())[:-2]) # Remove the last two layers

        self.LSTM = nn.LSTM(input_size=embedding_size, hidden_size=LSTM_units, 
                            num_layers=LSTM_layers, batch_first=False)
        self.pool2d = nn.AdaptiveAvgPool2d((global_avg_pool_size, global_avg_pool_size))
        self.Dropout = nn.Dropout(p=dropout)
        self.Softmax = nn.Softmax(dim=1)

        self.mutan1 = MutanFusion(LSTM_units*self.num_ques_glimpse, self.Mutan_out)
        self.mutan2 = MutanFusion(LSTM_units*self.num_ques_glimpse, self.Mutan_out)
        
        self.Linear1_q_proj = nn.Linear(LSTM_units*self.num_ques_glimpse, self.Mutan_output_dim)
        self.Conv_i_proj = nn.Conv2d(self.feat_size, self.Mutan_output_dim, 1)

        self.Dropout_L = nn.Dropout(p=0.2)
        self.Dropout_M = nn.Dropout(p=0.2)
        self.Conv1_Qatt = nn.Conv2d(LSTM_units, 512, 1)
        self.Conv2_Qatt = nn.Conv2d(512, self.num_ques_glimpse, 1)
        self.Conv1_Iatt = nn.Conv2d(1000, 512, 1)
        self.Conv2_Iatt = nn.Conv2d(512, self.num_img_glimpse, 1)

        self.Linear_predict = nn.Linear(self.Mutan_out, self.ans_vocab_size)
        
        self.qatt_maps = None
        self.iatt_maps = None
        
    def forward(self, ques_embed, img):
        # Extract image features using pretrained ResNet
        img_feat = self.image_encoder(img)
        img_feat_pooled = self.pool2d(img_feat)
        img_feat_sq = img_feat_pooled.view(img_feat_pooled.size(0), -1) # Flatten
        
        # Preparing image features
        img_feat_resh = img_feat_sq.view(img_feat_sq.size(0), -1, self.channel_size * self.channel_size)
        
        # ques_embed                                         N x T x embedding_size
        ques_embed_resh = ques_embed.permute(1, 0, 2).contiguous()        # T x N x embedding_size
        lstm_out, (hn, cn) = self.LSTM(ques_embed_resh)
        lstm1_droped = self.Dropout_L(lstm_out)
        lstm1_resh = lstm1_droped.permute(1, 2, 0).contiguous()                   # N x 1024 x T
        lstm1_resh2 = torch.unsqueeze(lstm1_resh, 3)                # N x 1024 x T x 1

        '''
        Question Attention
        '''
        qatt_conv1 = self.Conv1_Qatt(lstm1_resh2)                   # N x 512 x T x 1
        qatt_relu = F.relu(qatt_conv1)
        qatt_conv2 = self.Conv2_Qatt(qatt_relu)                     # N x 2 x T x 1
        qatt_conv2 = qatt_conv2.reshape(qatt_conv2.shape[0]*self.num_ques_glimpse,-1)
        qatt_softmax = self.Softmax(qatt_conv2)
        qatt_softmax = qatt_softmax.view(qatt_conv1.shape[0], self.num_ques_glimpse, -1, 1)
        self.qatt_maps = qatt_softmax
        qatt_feature_list = []
        for i in range(self.num_ques_glimpse):
            t_qatt_mask = qatt_softmax.narrow(1, i, 1)              # N x 1 x T x 1
            t_qatt_mask = t_qatt_mask * lstm1_resh2                 # N x 1024 x T x 1
            t_qatt_mask = torch.sum(t_qatt_mask, 2, keepdim=True)   # N x 1024 x 1 x 1
            qatt_feature_list.append(t_qatt_mask)
        qatt_feature_concat = torch.cat(qatt_feature_list, 1)       # N x 2048 x 1 x 1

        '''
        Image Attention with Mutan
        '''
        q_feat_resh = torch.squeeze(qatt_feature_concat)                                # N x 2048
        i_feat_resh = torch.unsqueeze(img_feat_resh, 3)                                   # N x 2048 x w*w x 1

        i_feat_mutan_input = i_feat_resh.reshape(img_feat_resh.shape[0], 
                                                img_feat_resh.shape[1]*self.channel_size*self.channel_size
                                            )      # N x 2048*w*w

        iq_mutan_output = self.mutan1(i_feat_mutan_input, q_feat_resh)                        # N x 5000*w*w
        iq_mutan_resh = iq_mutan_output.view(
            iq_mutan_output.shape[0],
            self.Mutan_output_dim,
            self.channel_size*self.channel_size,
            1
        )                                                                               # N x 5000 x w*w x 1        
        
        iatt_iq_droped = self.Dropout_M(iq_mutan_resh)                                # N x 5000 x w*w x 1
        
        iatt_iq_permute1 = iatt_iq_droped.permute(0,2,1,3).contiguous()                              # N x w*w x 5000 x 1
        iatt_iq_resh = iatt_iq_permute1.view(iatt_iq_permute1.shape[0], self.channel_size*self.channel_size, 
                                             self.Mutan_out, self.Mutan_factor)
        iatt_iq_sumpool = torch.sum(iatt_iq_resh, 3, keepdim=True)                      # N x w*w x 1000 x 1 
        iatt_iq_permute2 = iatt_iq_sumpool.permute(0,2,1,3).contiguous()                            # N x 1000 x w*w x 1

        iatt_iq_sqrt = torch.sqrt(F.relu(iatt_iq_permute2)) - torch.sqrt(F.relu(-iatt_iq_permute2))
        iatt_iq_sqrt = torch.squeeze(iatt_iq_sqrt)
        iatt_iq_sqrt = iatt_iq_sqrt.reshape(iatt_iq_sqrt.shape[0], -1)                           # N x 1000*w*w
        iatt_iq_l2 = F.normalize(iatt_iq_sqrt)
        iatt_iq_l2 = iatt_iq_l2.view(iatt_iq_l2.shape[0], self.Mutan_out, self.channel_size*self.channel_size, 1)  # N x 1000 x w*w x 1
        
        ## 2 conv layers 1000 -> 512 -> 2
        iatt_conv1 = self.Conv1_Iatt(iatt_iq_l2)                    # N x 512 x w*w x 1
        iatt_relu = F.relu(iatt_conv1)
        iatt_conv2 = self.Conv2_Iatt(iatt_relu)                     # N x 2 x w*w x 1
        iatt_conv2 = iatt_conv2.view(iatt_conv2.shape[0]*self.num_img_glimpse, -1)
        iatt_softmax = self.Softmax(iatt_conv2)
        iatt_softmax = iatt_softmax.view(iatt_conv1.shape[0], self.num_img_glimpse, -1, 1)
        self.iatt_maps = iatt_softmax.view(iatt_conv1.shape[0], self.num_img_glimpse, self.channel_size, self.channel_size)
        iatt_feature_list = []
        for i in range(self.num_img_glimpse):
            t_iatt_mask = iatt_softmax.narrow(1, i, 1)              # N x 1 x w*w x 1
            t_iatt_mask = t_iatt_mask * i_feat_resh                 # N x 2048 x w*w x 1
            t_iatt_mask = torch.sum(t_iatt_mask, 2, keepdim=True)   # N x 2048 x 1 x 1
            iatt_feature_list.append(t_iatt_mask)
        iatt_feature_concat = torch.cat(iatt_feature_list, 1)       # N x 4096 x 1 x 1
        iatt_feature_concat = torch.squeeze(iatt_feature_concat)    # N x 4096

        '''
        Fine-grained Image-Question Mutan fusion
        '''
        iq_feat = self.mutan2(iatt_feature_concat, q_feat_resh)          # N x 5000
        Mutan_iq_drop = self.Dropout_M(iq_feat)
        Mutan_iq_resh = Mutan_iq_drop.view(Mutan_iq_drop.shape[0], 1, self.Mutan_out, self.Mutan_factor)   # N x 1 x 1000 x 5
        Mutan_iq_sumpool = torch.sum(Mutan_iq_resh, 3, keepdim=True)    # N x 1 x 1000 x 1
        Mutan_out = torch.squeeze(Mutan_iq_sumpool)                     # N x 1000
        Mutan_sign_sqrt = torch.sqrt(F.relu(Mutan_out)) - torch.sqrt(F.relu(-Mutan_out))
        Mutan_l2 = F.normalize(Mutan_sign_sqrt)
        prediction = self.Linear_predict(Mutan_l2)
        prediction = self.Softmax(prediction)

        return prediction


def build_MUTAN(dataset, args):
    """
    Function to build the MUTAN model for Visual Question Answering.

    Args:
        dataset (Dataset): Dataset object containing the VQA dataset.
        args (Namespace): Arguments containing hyperparameters and configurations.

    Returns:
        Mutan_CoAtt: Initialized MUTAN co-attention model.
    """
    
    # Define embedding size, LSTM units, layers, etc.
    embedding_size = args.embedding_size
    LSTM_units = args.LSTM_units
    LSTM_layers = args.LSTM_layers
    feat_size = args.feat_size
    batch_size = args.batch_size
    ans_vocab_size = dataset.num_ans_candidates
    global_avg_pool_size = args.global_avg_pool_size
    dropout = args.dropout
    Mutan_output_dim = args.Mutan_output_dim

    # Construct and return the MUTAN model
    model = Mutan_CoAtt(embedding_size=embedding_size, 
                        LSTM_units=LSTM_units, 
                        LSTM_layers=LSTM_layers, 
                        feat_size=feat_size, 
                        batch_size=batch_size, 
                        ans_vocab_size=ans_vocab_size, 
                        global_avg_pool_size=global_avg_pool_size, 
                        dropout=dropout, 
                        Mutan_output_dim=Mutan_output_dim)
    return model
