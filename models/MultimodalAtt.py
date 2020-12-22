import numpy as np
import torch
import torch.nn as nn 
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from .Attention import Attention
from .AttentionFlow import AttentionFlow
from .TemporalAttFlow import TemporalAttentionOverFlow
from torch.autograd import Variable
from functools import cmp_to_key


def cmp(x, y):
    if x == y:
        return 0
    if x > y:
        return 1
    if x < y:
        return -1


def KeyL(x, y):
    return cmp(x[1], y[1])


class MultimodalAtt(nn.Module):

    def __init__(self, vocab_size, max_len, dim_hidden, dim_word, dim_vid=2048, 
        dim_audio=20, sos_id=1, eos_id=0, n_layers=1, rnn_dropout_p=0):
        super(MultimodalAtt, self).__init__()

        self.rnn_cell = nn.LSTM
        
        # dimension of the word embedding layer
        self.dim_word = dim_word
        # Output size would be a one-dimensional vector of the vocab base
        self.dim_output = vocab_size
        # The hidden size of the LSTM cells output. DEFAULT TO 256 
        self.dim_hidden = dim_hidden
        # Define the max length of either the generated sentence or training caption
        self.max_len = max_len
        # Number of layers for the LSTM cells. Default to 1
        self.n_layers = n_layers
        # Dimension of the video feature. Default to 2048
        self.dim_vid = dim_vid
        # Dimension of the audio feature. Default to 20 (extract 20 MFCCs for each time step)
        self.dim_audio = dim_audio
        # The ix in the vocab base for the <SOS> signal
        self.sos_id = sos_id
        # Same as above for <EOS>
        self.eos_id = eos_id

        # Define LSTM encoders
        self.fc2_encoder = nn.LSTM(
                input_size=128,
                hidden_size=self.dim_hidden,
                bidirectional=True,
                batch_first=True,
                )

        self.conv4_encoder = nn.LSTM(
                input_size=512,
                hidden_size=self.dim_hidden,
                bidirectional=True,
                batch_first=True,
                )

        self.word_enc = nn.LSTM(
                input_size=self.dim_word,
                hidden_size=self.dim_hidden,
                bidirectional=True,
                batch_first=True,
                )
        # DEFINE LINEAR LAYER TO PROJECT FC2 HIDDENSTATE TO C4 DIMENSIONS
        #self.linear_fc2 = nn.Linear(256, 512)
        self.w_dec = nn.Linear(10*self.dim_hidden, 2*self.dim_hidden)

        # DEFINE ATTENTION LAYERS
        self.TemporalAttention_aud = Attention(2*self.dim_hidden)
        self.TemporalAttention_flow = TemporalAttentionOverFlow(2*self.dim_hidden)
        self.FlowMatrix = AttentionFlow(2*self.dim_hidden)

        # DEFINE DECODER TO GENERATE CAPTION
        self.decoder = nn.LSTM(
                input_size=self.dim_word+2*dim_hidden,
                hidden_size=2*self.dim_hidden,
                batch_first=True,
                )

        # LOADING THE PRE TRAINED EMBEDDINGS FROM FASTTEXT
        if vocab_size == 16860: # MSRVTT
            pretrained = torch.Tensor(np.load('/home/cxu-serve/p1/rohan27/'\
                'research/audiocaps/code2/helpers/fasttext_msrvtt.npy'))
        elif vocab_size == 4506:
            pretrained = torch.Tensor(np.load('/home/cxu-serve/p1/rohan27/'\
                'research/audiocaps/acaps_pytorch/data/fasttext_acaps.npy'))
        self.embedding = nn.Embedding.from_pretrained(pretrained)

        # OUTPUT LAYER
        self.out = nn.Linear(2*self.dim_hidden, self.dim_output)

    def forward(self, ac4, afc2, sem_feats, target_variable=None, mode='train', opt={}):
        # ENCODE THE WORD SEMANTIC FEATURES
        words = self.embedding(sem_feats)
        w_encoder_output, (w_encoder_state, w_cell_state) = self.word_enc(words)

        # GET THE INPUT SHAPES
        bs, seq_len, _, _, c4_dim = ac4.shape 

        # ENCODE THE SOUND INPUTS
        fc2_encoder_output, (fc2_hidden_state, fc2_cell_state) = self.fc2_encoder(afc2)
        fc2_hidden_state = fc2_hidden_state.view(bs, -1) # bs, 512
        fc2_hidden_state.unsqueeze_(1) # bs, 1, 512

        # Pre-Process the conv4 features
        ac4 = ac4.view(bs, seq_len, -1, c4_dim)
        ac4 = torch.sum(ac4, dim=2) # bs, seq_len, c4_dim=512

        # DO ELEMENT WISE ADDITION BEFORE USING BI-LSTM
        ac4 = ac4 + fc2_hidden_state # Why don't we add the output tensor of fc2 instead of the hidden one
        encoder_output, (c4_hidden_state, c4_cell_state) = self.conv4_encoder(ac4)

        c4_hidden_state = c4_hidden_state.view(1, bs, -1) # 1, bs, 512
        c4_cell_state = c4_cell_state.view(1, bs, -1) # 1, bs, 512
        decoder_hidden = c4_hidden_state
        decoder_cell = c4_cell_state
        decoder_input = self.TemporalAttention_aud(c4_hidden_state, encoder_output) # bs, 1, 512

        # GET THE ALIGNED EMBEDDING MATRIX
        aligned_embds = self.FlowMatrix(encoder_output, w_encoder_output)

        seq_probs = list()
        seq_preds = list()
        if mode == 'train':
            for i in range(self.max_len - 1):
                # <eos> doesn't input to the network
                current_words = self.embedding(target_variable[:, i])
                self.decoder.flatten_parameters()
                decoder_input = torch.cat((decoder_input, current_words.unsqueeze(1)), dim=2)
                #decoder_input = current_words.unsqueeze(1) # USING TEACHER FORCING (COMMENT IT OUT FOR OTHER MODELS)
                decoder_output, (decoder_hidden, decoder_cell) = \
                        self.decoder(decoder_input, (decoder_hidden, decoder_cell))
                context = self.TemporalAttention_flow(decoder_hidden, aligned_embds) # bs, 1, 4*512
                output = torch.cat((context, torch.transpose(decoder_hidden, 0, 1)), dim=2)
                output = F.tanh(self.w_dec(output))
                decoder_input = output
                
                output = self.out(output)
                seq_probs.append(output)
            seq_probs = torch.cat(seq_probs, 1)

        elif mode == 'inference' and opt['beam']:
            beam_size = opt['beam_size']
            start = [Variable(torch.cuda.LongTensor([self.sos_id] * bs))]
            current_words = [[start, 0.0, (decoder_hidden, decoder_cell), decoder_input]]

            for j in range(self.max_len-1):
                temp = []
                for i in range(len(current_words)):
                    s = current_words[i]
                    emb_words = self.embedding(s[0][-1]) 
                    self.decoder.flatten_parameters()

                    decoder_input = torch.cat((s[-1], emb_words.unsqueeze(1)), dim=2)
                    decoder_output, s[2] = self.decoder(decoder_input, s[2])
                    context = self.TemporalAttention_flow(s[2][0], aligned_embds)    
                    output = torch.cat((context, s[2][0]), dim=2)
                    output = F.tanh(self.w_dec(output))
                    s[-1] = output
                    logits = F.log_softmax(self.out(output).squeeze(1), dim=1)

                    seq_probs.append(logits.unsqueeze(1))
                    #_, preds = torch.max(logits, 1)

                    # GET THE TOP K PREDICTIONS FOR BEAM SEARCH
                    topk_probs, topk_words = torch.topk(logits, k=beam_size, dim=1)
                    # batch*beam -> beam*batch
                    topk_probs = topk_probs.permute(1, 0)
                    topk_words = topk_words.permute(1, 0)

                    for prob, word in zip(topk_probs, topk_words):
                        next_cap = s[0][:]
                        next_cap.append(word)
                        temp.append([next_cap, s[1]+prob, (s[2][0].clone(), s[2][1].clone()), s[-1].clone()])

                current_words = temp
                current_words = sorted(current_words, reverse=False,\
                        key=cmp_to_key(KeyL))

                current_words = current_words[-beam_size:]
            seq_preds = torch.cat(current_words[-1][0][1:], 0).unsqueeze(0)

        elif mode == 'inference' and not opt['beam']:
            current_words = self.embedding(torch.cuda.LongTensor([self.sos_id] * bs))

            for i in range(self.max_len-1):
                self.decoder.flatten_parameters()
                #decoder_input = current_words.unsqueeze(1) # USING TEACHER FORCING (COMMENT IT OUT FOR OTHER MODELS)
                decoder_input = torch.cat((decoder_input, current_words.unsqueeze(1)), dim=2)
                decoder_output, (decoder_hidden, decoder_cell) =\
                        self.decoder(decoder_input, (decoder_hidden, decoder_cell))
                context = self.TemporalAttention_flow(decoder_hidden, aligned_embds) # bs, 1, 4*512
                output = torch.cat((context, torch.transpose(decoder_hidden, 0, 1)), dim=2)
                output = F.tanh(self.w_dec(output))
                decoder_input = output
                
                logits = F.log_softmax(self.out(output).squeeze(1), dim=1)
                seq_probs.append(logits.unsqueeze(1))
                _, preds = torch.max(logits, 1)
                current_words = self.embedding(preds)
                seq_preds.append(preds.unsqueeze(1))
            seq_probs = torch.cat(seq_probs, 1)
            seq_preds = torch.cat(seq_preds, 1)
 
        return seq_probs, seq_preds


