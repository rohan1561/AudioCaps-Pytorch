import numpy as np
import torch
import torch.nn as nn 
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from .Attention import Attention
from .ChildSum import ChildSum
from queue import PriorityQueue
import operator

class BeamSearchNode(object):
    def __init__(self, hiddenstate, previousNode, wordId, logProb, length):
        '''
        :param hiddenstate:
        :param previousNode:
        :param wordId:
        :param logProb:
        :param length:
        '''
        self.h = hiddenstate
        self.prevNode = previousNode
        self.wordid = wordId
        self.logp = logProb
        self.leng = length

    def eval(self, alpha=1.0):
        reward = 0
        # Add here a function for shaping a reward

        return self.logp / float(self.leng - 1 + 1e-6) + alpha * reward


#decoder = DecoderRNN()


def beam_decode(batch_size, SOS_token, EOS_token, decoder_hiddens, decoder, attention_mod, encoder_outputs, decoder_inputs, embedding, output_layer, max_len):
    '''
    :param batch_size: target indexes tensor of shape [B, T] where B is the batch size and T is the maximum length of the output sentence (REPLACED WITH BATCH SIZE)
    :param decoder_hidden: input tensor of shape [1, B, H] for start of the decoding
    :param encoder_outputs: if you are using attention mechanism you can pass encoder outputs, [T, B, H] where T is the maximum length of input sentence
    :return: decoded_batch
    '''

    beam_width = 10
    topk = 1  # how many sentence do you want to generate
    decoded_batch = []

    # decoding goes sentence by sentence
    for idx in range(batch_size):
        if isinstance(decoder_hiddens, tuple):  # LSTM case
            decoder_hidden = (decoder_hiddens[0][:,idx, :].unsqueeze(0),decoder_hiddens[1][:,idx, :].unsqueeze(0))
        else:
            decoder_hidden = decoder_hiddens[:, idx, :].unsqueeze(0)
        encoder_output = encoder_outputs[idx, :, :].unsqueeze(0)

        # Start with the start of the sentence token
        #decoder_input = torch.LongTensor([[SOS_token]], device="cpu")
        decoder_input = torch.LongTensor([[SOS_token]]).cuda()


        # Number of sentence to generate
        endnodes = []
        number_required = min((topk + 1), topk - len(endnodes))

        # starting node -  hidden vector, previous node, word id, logp, length
        node = BeamSearchNode(decoder_hidden, None, decoder_input, 0, 1)
        nodes = PriorityQueue()

        # start the queue
        nodes.put((-node.eval(), node))
        qsize = 1

        # start beam search
        while True:
            # give up when decoding takes too long
            if qsize > 2000: break

            # fetch the best node
            score, n = nodes.get()
            decoder_input = n.wordid
            decoder_hidden = n.h
            if qsize > 1:
                aud_context = attention_mod(decoder_hidden[0], encoder_output)
                decoder_input = torch.cat((aud_context, embedding(decoder_input)), dim=2) 
            elif qsize == 1:
                decoder_input = torch.cat((decoder_inputs[idx, ...].unsqueeze(0),
                    embedding(decoder_input)), dim=2)

            if n.wordid.item() == EOS_token and n.prevNode != None:
                endnodes.append((score, n))
                # if we reached maximum # of sentences required
                if len(endnodes) >= number_required:
                    break
                else:
                    continue

            # decode for one step using decoder
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            decoder_output = F.log_softmax(output_layer(decoder_output.squeeze(0)), dim=1)
            

            # PUT HERE REAL BEAM SEARCH OF TOP
            log_prob, indexes = torch.topk(decoder_output, beam_width)
            nextnodes = []

            for new_k in range(beam_width):
                decoded_t = indexes[0][new_k].view(1, -1)
                log_p = log_prob[0][new_k].item()

                node = BeamSearchNode(decoder_hidden, n, decoded_t, n.logp + log_p, n.leng + 1)
                score = -node.eval()
                nextnodes.append((score, node))

            # put them into queue
            for i in range(len(nextnodes)):
                score, nn = nextnodes[i]
                nodes.put((score, nn))
                # increase qsize
            qsize += len(nextnodes) - 1

        # choose nbest paths, back trace them
        if len(endnodes) == 0:
            endnodes = [nodes.get() for _ in range(topk)]

        utterances = []
        for score, n in sorted(endnodes, key=operator.itemgetter(0)):
            utterance = []
            utterance.append(n.wordid)
            # back trace
            while n.prevNode != None:
                n = n.prevNode
                utterance.append(n.wordid)

            utterance = utterance[::-1]
            if len(utterance) < max_len:
                utterance += [torch.LongTensor([[0]]).cuda()]*(max_len-len(utterance))
            if len(utterance) > max_len:
                utterance = utterance[:max_len-1] + [torch.LongTensor([[0]]).cuda()]
            utterances.append(utterance)

        decoded_batch.append(utterances[0])

    return decoded_batch


class MultimodalAtt3(nn.Module):

    def __init__(self, vocab_size, max_len, dim_hidden, dim_word, dim_vid=2048, 
        dim_audio=20, sos_id=1, eos_id=0, n_layers=1, rnn_dropout_p=0):
        super(MultimodalAtt3, self).__init__()

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

        # DEFINE ATTENTION LAYERS
        self.TemporalAttention_aud = Attention(512)

        # DEFINE DECODER TO GENERATE CAPTION
        self.decoder = nn.LSTM(
                input_size=self.dim_word + 512,
                hidden_size=2*self.dim_hidden,
                batch_first=True,
                )

        # LOADING THE PRE TRAINED EMBEDDINGS FROM FASTTEXT
        pretrained = torch.Tensor(np.load('/home/cxu-serve/p1/rohan27/'\
            'research/audiocaps/code2/helpers/fasttext_msrvtt.npy'))
        self.embedding = nn.Embedding.from_pretrained(pretrained)

        # OUTPUT LAYER
        self.out = nn.Linear(512, self.dim_output)

    def forward(self, ac4, afc2, target_variable=None, mode='train', opt={}):
        # GET THE INPUT SHAPES
        bs, seq_len, _, _, c4_dim = ac4.shape 
        _, _, fc2_dim = afc2.shape

        '''
        # ENCODE THE WORD EMBEDDING INPUTS
        words = self.embedding(target_variable)
        print(words.shape)
        w_encoder_output, (w_encoder_state, w_cell_state) = self.word_enc(words)
        print(w_encoder_output.shape)
        '''

        # ENCODE THE SOUND INPUTS
        fc2_encoder_output, (fc2_hidden_state, fc2_cell_state) = self.fc2_encoder(afc2)
        fc2_hidden_state = fc2_hidden_state.view(bs, -1) # bs, 512
        fc2_hidden_state.unsqueeze_(1) # bs, 1, 512

        if ac4 is not None:
            # Pre-Process the conv4 features
            ac4 = ac4.view(bs, seq_len, -1, c4_dim)
            ac4 = torch.sum(ac4, dim=2) # bs, seq_len, c4_dim=512

            # DO ELEMENT WISE ADDITION BEFORE USING BI-LSTM
            ac4 = ac4 + fc2_hidden_state # Why don't we add the output tensor of fc2 instead of the hidden one
            c4_encoder_output, (c4_hidden_state, c4_cell_state) = self.conv4_encoder(ac4)

            c4_hidden_state = c4_hidden_state.view(1, bs, -1) # 1, bs, 512
            c4_cell_state = c4_cell_state.view(1, bs, -1) # 1, bs, 512
            decoder_hidden = c4_hidden_state
            decoder_cell = c4_cell_state
            decoder_input = self.TemporalAttention_aud(c4_hidden_state, c4_encoder_output) # bs, 1, 512
        elif ac4 is None:
            # USE FC2 DIRECTLY FOR ABLATION
            decoder_hidden = fc2_hidden_state.transpose(0, 1) # 1, bs, 512
            decoder_cell = fc2_cell_state.view(1, bs, -1) # 1, bs, 512
            decoder_input = self.TemporalAttention_aud(fc2_hidden_state, fc2_encoder_output) # bs, 1, 512

        seq_probs = list()
        seq_preds = list()
        if mode == 'train':
            for i in range(self.max_len - 1):
                # <eos> doesn't input to the network
                current_words = self.embedding(target_variable[:, i])
                self.decoder.flatten_parameters()
                decoder_input = torch.cat((decoder_input, current_words.unsqueeze(1)), dim=2)
                decoder_output, (decoder_hidden, decoder_cell) = \
                        self.decoder(decoder_input, (decoder_hidden, decoder_cell))
                aud_context = self.TemporalAttention_aud(decoder_hidden, c4_encoder_output)
                decoder_input = aud_context
                
                output = self.out(decoder_output)
                seq_probs.append(output)
            seq_probs = torch.cat(seq_probs, 1)

        elif mode == 'inference':
            result = beam_decode(decoder_input.shape[0], self.sos_id, self.eos_id, (decoder_hidden,\
                    decoder_cell), self.decoder, self.TemporalAttention_aud,\
                    c4_encoder_output, decoder_input, self.embedding, self.out, self.max_len-1)
            seq_preds = []
            for r in result:
                sent = torch.stack(r, dim=0)
                seq_preds.append(sent)
            seq_preds = torch.stack(seq_preds, dim=0).squeeze()
            print(seq_preds.shape)
            seq_probs = 0


            '''
            for i in range(self.max_len-1):
                self.decoder.flatten_parameters()
                decoder_input = torch.cat((decoder_input, current_words.unsqueeze(1)), dim=2)
                decoder_output, (decoder_hidden, decoder_cell) =\
                        self.decoder(decoder_input, (decoder_hidden, decoder_cell))
                aud_context = self.TemporalAttention_aud(decoder_hidden, c4_encoder_output)    
                decoder_input = aud_context
                
                logits = F.log_softmax(self.out(decoder_output).squeeze(1), dim=1)
                seq_probs.append(logits.unsqueeze(1))
                _, preds = torch.max(logits, 1)
                current_words = self.embedding(preds)
                seq_preds.append(preds.unsqueeze(1))
            seq_probs = torch.cat(seq_probs, 1)
            seq_preds = torch.cat(seq_preds, 1)
            '''
        return seq_probs, seq_preds


