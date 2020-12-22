import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Subset
import torch.optim as optim
from torch.nn.utils import clip_grad_value_
import os
import argparse
import json 
from models import MultimodalAtt
from NLUtils import LanguageModelCriterion
import evaluate

def train(loader, loader_val, val_vocab, model, crit, optimizer, lr_scheduler, opt):
    model = nn.DataParallel(model)
    max_scores = 0
    for epoch in range(opt['epochs']):
        model.train()
        save_flag=True
        lr_scheduler.step()
        iteration = 0

        for data in loader:
            audio_conv4 = data['audio_conv4'].cuda()
            audio_fc2 = data['audio_fc2'].cuda()
            labels = data['labels'].cuda()
            masks = data['masks'].cuda()
            sem_feats = data['sem_feats'].cuda()

            torch.cuda.synchronize()
            optimizer.zero_grad()
            
            seq_probs, _ = model(audio_conv4, audio_fc2, sem_feats, labels, 'train', opt=opt)

            loss = crit(seq_probs, labels[:, 1:], masks[:, 1:])
            loss.backward()
            clip_grad_value_(model.parameters(), opt['grad_clip'])
            optimizer.step()
            train_loss = loss.item()
            torch.cuda.synchronize()
            iteration += 1

            print("iter %d (epoch %d), train_loss = %.6f" % (iteration, epoch, train_loss))

            '''
            if epoch % opt["save_checkpoint_every"] == 0 and save_flag:
                model_path = os.path.join(opt["checkpoint_path"], 'model_%d.pth' % (epoch))
                model_info_path = os.path.join(opt["checkpoint_path"], 'model_score.txt')
                torch.save(model.state_dict(), model_path)
                print("model saved to %s" % (model_path))
                with open(model_info_path, 'a') as f:
                    f.write("model_%d, loss: %.6f\n" % (epoch, train_loss))
                save_flag=False
            '''
        scores = evaluate.main(loader_val, val_vocab, opt, model)
        sc = scores['Bleu_4']
        scores = sum([scores['Bleu_1'], scores['Bleu_2'], scores['Bleu_3'], scores['Bleu_4']])
        if scores > max_scores:
            max_scores = scores
            print(scores)
            model_path = os.path.join(opt["checkpoint_path"], 'model_%d.pth' % (epoch))
            model_info_path = os.path.join(opt["checkpoint_path"], 'model_score.txt')
            torch.save(model.state_dict(), model_path)
            print("model saved to %s" % (model_path))
            with open(model_info_path, 'a') as f:
                f.write(f"model_%{epoch}, bleu4: {sc}\n")


def main(opt):
    dataset = VideoAudioDataset(opt, 'train')
    opt['vocab_size'] = dataset.get_vocab_size()
    loader = DataLoader(dataset, batch_size=opt['batch_size'], shuffle=True)

    dataset_val = VideoAudioDataset(opt, 'test')
    loader_val = DataLoader(dataset_val, batch_size=opt['batch_size'], shuffle=True)
    val_vocab = dataset_val.get_vocab()

    model = MultimodalAtt(opt['vocab_size'], opt['max_len'], opt['dim_hidden'], opt['dim_word'],
                            n_layers=opt['num_layers'], rnn_dropout_p=opt['rnn_dropout_p'])
    model = model.cuda()
    crit = LanguageModelCriterion()
    optimizer = optim.Adam(
        model.parameters(),
        lr=opt["learning_rate"],
        weight_decay=opt["weight_decay"],
        amsgrad=True)
    exp_lr_scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=opt["learning_rate_decay_every"],
        gamma=opt["learning_rate_decay_rate"])

    train(loader, loader_val, val_vocab, model, crit, optimizer, exp_lr_scheduler, opt)

if __name__ == '__main__':
    data_source = input('Source? (msr/acaps):')
 
    if data_source == 'acaps': 
        import opts_acaps
        from dataloader_acaps import VideoAudioDataset
        opt = opts_acaps.parse_opt()
    elif data_source == 'msr':
        import opts
        from dataloader import VideoAudioDataset
        opt = opts.parse_opt()
    opt = vars(opt)
    opt['Source'] = data_source
    os.environ['CUDA_VISIBLE_DEVICES'] = opt["gpu"]
    opt_json = os.path.join(opt["checkpoint_path"], 'opt_info.json')
    if not os.path.isdir(opt["checkpoint_path"]):
        os.mkdir(opt["checkpoint_path"])
    with open(opt_json, 'w') as f:
        json.dump(opt, f)
    print('save opt details to %s' % (opt_json))
    main(opt)


