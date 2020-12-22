import torch
import torch.nn as nn 
from torch.utils.data import DataLoader
from torch.utils.data import Subset

import os
import argparse
import json
from pandas.io.json import json_normalize

import NLUtils
from cocoeval import suppress_stdout_stderr, COCOScorer
from dataloader import VideoAudioDataset
from models import MultimodalAtt
from tqdm import tqdm
import opts


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def convert_data_to_coco_scorer_format(data_frame):
    gts = {}
    for row in zip(data_frame["caption"], data_frame["video_id"]):
        if row[1] in gts:
            gts[row[1]].append(
                {'image_id': row[1], 'cap_id': len(gts[row[1]]), 'caption': row[0]})
        else:
            gts[row[1]] = []
            gts[row[1]].append(
                {'image_id': row[1], 'cap_id': len(gts[row[1]]), 'caption': row[0]})
    return gts


def eval(model, crit, loader, vocab, opt):
    model.eval()
    '''
    if opt['beam']:
        bs = 1
    else:
        bs = opt['batch_size']
    loader = DataLoader(dataset, batch_size=bs, shuffle=True)
    '''
    scorer = COCOScorer()
    gt_dataframe = json_normalize(
        json.load(open(opt["input_json"]))['sentences'])
    gts = convert_data_to_coco_scorer_format(gt_dataframe)
    results = []
    samples = {}
    for data in tqdm(loader):
        # forward the model to get loss
        video_ids = data['video_ids']
        audio_conv4 = data['audio_conv4'].cuda()
        audio_fc2 = data['audio_fc2'].cuda()
        sem_feats = data['sem_feats'].cuda()
       
        # forward the model to also get generated samples for each image
        with torch.no_grad():
            seq_probs, seq_preds = model(audio_conv4, audio_fc2, sem_feats, mode='inference', opt=opt)

        sents = NLUtils.decode_sequence(vocab, seq_preds)

        for k, sent in enumerate(sents):
            video_id = video_ids[k]
            samples[video_id] = [{'image_id': video_id, 'caption': sent}]

    with suppress_stdout_stderr():
        valid_score = scorer.score(gts, samples, samples.keys())
    results.append(valid_score)
    print(valid_score)

    if not os.path.exists(opt["results_path"]):
        os.makedirs(opt["results_path"])

    '''
    with open(os.path.join(opt["results_path"], "scores.txt"), 'a') as scores_table:
        scores_table.write(json.dumps(results[0]) + "\n")
    with open(os.path.join(opt["results_path"],
                           'vanilla' + ".json"), 'w') as prediction_results:
        json.dump({"predictions": samples, "scores": valid_score},
                  prediction_results)
    '''
    return valid_score

def main(loader, vocab, opt, model=None):
    data_source = opt['Source']

    '''
    if data_source == 'acaps': 
        from dataloader_acaps import VideoAudioDataset
    elif data_source == 'msr':
        from dataloader import VideoAudioDataset
    '''

    if model is None:
        vocab_size = len(vocab)
        model = MultimodalAtt(vocab_size, opt['max_len'],
                opt['dim_hidden'], opt['dim_word'])

        model = nn.DataParallel(model)

        if opt['beam']:
            bw = opt['beam_size']
            print(f'Using beam search with beam width = {bw}')
        model_path = opt['checkpoint_path']
        for i in os.listdir(model_path):
            if i.endswith('.pth'):
                print(i)
                path = os.path.join(model_path, i)
                model.load_state_dict(torch.load(path))
                crit = NLUtils.LanguageModelCriterion()

                eval(model, crit, loader, vocab, opt)
    else:
        '''
        Running from inside train.py
        '''
        if opt['beam']:
            bw = opt['beam_size']
            print(f'Using beam search with beam width = {bw}')

        crit = NLUtils.LanguageModelCriterion()
        scores = eval(model, crit, loader, vocab, opt)
        return scores


if __name__ == '__main__':

    data_source = input('Source? (msr/acaps):')
    print('Dont forget to change the folder in the line below')
    folder = './save_msr_max_len_44/'
    print(f'MODELS in the folder: {folder}')
    opt = json.load(open(os.path.join(folder, 'opt_info.json')))
    os.environ['CUDA_VISIBLE_DEVICES'] = opt["gpu"]
    beam = input('beam? (y/n):')
    if beam == 'y':
        beam_size = int(input('beam size?:'))
        opt["beam"] = True
        opt["beam_size"] = beam_size
        batch_size = 1

    elif beam == 'n':
        opt['beam'] = False
        batch_size = opt['batch_size']

    if data_source == 'acaps': 
        from dataloader_acaps import VideoAudioDataset
    elif data_source == 'msr':
        from dataloader import VideoAudioDataset

    mode = 'test'
    dataset = VideoAudioDataset(opt, mode)
    vocab = dataset.get_vocab()
    print(batch_size, 'xxxxxxxxxxxxxxx')
    loader_val = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    main(loader_val, vocab, opt)

