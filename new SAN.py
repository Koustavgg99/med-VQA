import torch
import torch.nn as nn
from torchvision import models
from attention import BiAttention, StackedAttention
from language_model import WordEmbedding, QuestionEmbedding
from classifier import SimpleClassifier
from utils import tfidf_loading

class SAN_Model(nn.Module):
    def __init__(self, w_emb, q_emb, v_att, classifier, args):
        super(SAN_Model, self).__init__()
        self.args = args
        self.w_emb = w_emb
        self.q_emb = q_emb
        self.v_att = v_att
        self.classifier = classifier

        # Use a pretrained ResNet for image encoding
        self.img_encoder = models.resnet50(pretrained=True)
        self.img_encoder.fc = nn.Identity()  # Remove the final classification layer

    def forward(self, v, q):
        """Forward

        v: [batch, num_objs, obj_dim]
        q: [batch_size, seq_length]

        return: logits, not probs
        """
        # get visual feature
        v = v.squeeze(1)  # Adjust the dimensions if necessary
        v_emb = self.img_encoder(v)

        # get textual feature
        w_emb = self.w_emb(q)
        q_emb = self.q_emb(w_emb)  # [batch, q_dim], return final hidden state

        # Attention
        att = self.v_att(v_emb, q_emb)
        return att

# Build SAN model
def build_SAN(dataset, args):
    # init word embedding module, question embedding module, and Attention network
    w_emb = WordEmbedding(dataset.dictionary.ntoken, 300, 0.0, args.op)
    q_emb = QuestionEmbedding(300 if 'c' not in args.op else 600, args.num_hid, 1, False, 0.0, args.rnn)
    v_att = StackedAttention(args.num_stacks, 2048, args.num_hid, args.num_hid, dataset.num_ans_candidates, args.dropout)  # 2048 is the output dim of ResNet

    # Loading tfidf weighted embedding
    if hasattr(args, 'tfidf'):
        w_emb = tfidf_loading(args.tfidf, w_emb, args)

    # init classifier
    classifier = SimpleClassifier(
        args.num_hid, 2 * args.num_hid, dataset.num_ans_candidates, args)
    
    # construct VQA model and return
    return SAN_Model(w_emb, q_emb, v_att, classifier, args)
