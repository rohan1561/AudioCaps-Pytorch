import json
import os
import random
import numpy as np

import torch
from torch.utils.data import Dataset
from moviepy.video.io.VideoFileClip import VideoFileClip
import h5py

class VideoAudioDataset(Dataset):
    def __init__(self, opt, mode):
        super(VideoAudioDataset, self).__init__()
        self.mode = mode

        info = json.load(open(opt["info_json"].split('.')[0] + f'_{self.mode}.json'))
        self.captions = info['all_info']
        self.indices = info['indices']
        self.ix_to_word = info['ix_to_word']
        self.word_to_ix = info['word_to_ix']
        self.feats_dir = opt['output_dir']
        self.max_len = opt['max_len']
        self.max_semfeat_len = opt['max_semfeat_len']
        self.max_video_duration = 10
        filename_fc2 = os.path.join(self.feats_dir, "vggish_last_pad.hdf5")
        filename_conv4 = os.path.join(self.feats_dir, "vggish_conv4_pad.hdf5")
 
        self.feat_fc2 = h5py.File(filename_fc2, 'r')
        self.feat_conv4 = h5py.File(filename_conv4, 'r')
       
        print('vocab size is ', len(self.ix_to_word))
        print('number of videos: ', len(self.indices))

        print('load features from %s' % (self.feats_dir))
        print('max sequence length in data is', self.max_len)


    def __getitem__(self, ix):
        '''
        if self.mode == 'val':
            ix += len(self.splits['train'])
        elif self.mode == 'test':
            ix = ix + len(self.splits['train']) + len(self.splits['val'])
        '''

        vid_id = self.indices[str(ix)]
        audio_conv4 = self.feat_conv4[vid_id].value.astype(np.float32)
        audio_fc2 = self.feat_fc2[vid_id].value.astype(np.float32)

        # GET THE CAPTIONS
        mask = np.zeros(self.max_len)
        caption = self.captions[vid_id + ' caption']
        gts = np.zeros(self.max_len)

        if len(caption) > self.max_len:
            caption = caption[:self.max_len]
            caption[-1] = '2'
        for j, w in enumerate(caption):
            gts[j] = int(w)

        label = gts
        try:
            non_zero = (label == 0).nonzero()
            mask[:int(non_zero[0][0])]=1
        except IndexError:
            print(vid_id)
            print(label)
            print(caption)

        # GET THE SEMANTIC FEATURES 
        sem_feats = self.captions[vid_id + ' sem_feats']
        gts2 = np.zeros(self.max_semfeat_len)

        if len(sem_feats) > self.max_semfeat_len:
            sem_feats = sem_feats[:self.max_semfeat_len]
        for j, w in enumerate(sem_feats):
            gts2[j] = int(w)

        data = dict()
        data['audio_conv4'] = torch.from_numpy(audio_conv4).type(torch.FloatTensor)
        data['audio_fc2'] = torch.from_numpy(audio_fc2).type(torch.FloatTensor)
        data['labels'] = torch.from_numpy(label).type(torch.LongTensor)
        data['masks'] = torch.from_numpy(mask).type(torch.FloatTensor)
        data['video_ids'] = vid_id
        data['gts'] = torch.from_numpy(gts).long()
        data['sem_feats'] = torch.from_numpy(gts2).long()
        return data

    def __len__(self):
        return len(self.indices)
    
    def get_vocab_size(self):
        return len(self.ix_to_word)
    
    def get_vocab(self):
        return self.ix_to_word
    
        
