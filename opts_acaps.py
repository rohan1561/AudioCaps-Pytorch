import argparse


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_opt():
    parser = argparse.ArgumentParser()
    # Data input settings
    parser.add_argument(
        '--input_json',
        type=str,
        default='data/capinfo_acaps.json',
        help='path to the json file containing video info')

    parser.add_argument(
        '--info_json',
        type=str,
        default='data/info_acaps.json',
        help='path to the json file containing additional info and vocab')

    parser.add_argument(
        '--dim_hidden',
        type=int,
        default=256,
        help='size of the rnn hidden layer')

    parser.add_argument(
        '--num_layers', type=int, default=1, help='number of layers in the RNN')

    parser.add_argument(
        "--max_len",
        type=int,
        default=52, # Actual max = 52
        help='max length of captions(containing <sos>,<eos>)')

    parser.add_argument(
        "--max_semfeat_len",
        type=int,
        default=22,
        help='max length of captions(containing <sos>,<eos>)')

    parser.add_argument(
        '--input_dropout_p',
        type=float,
        default=0.2,
        help='strength of dropout in the Language Model RNN')

    parser.add_argument(
        '--rnn_dropout_p',
        type=float,
        default=0.0,
        help='strength of dropout in the Language Model RNN')

    parser.add_argument(
        '--dim_word',
        type=int,
        default=300,
        help='the encoding size of each token in the vocabulary, and the video.'
    )

    # Optimization: General

    parser.add_argument(
        '--epochs', type=int, default=3001, help='number of epochs')

    parser.add_argument(
        '--batch_size', type=int, default=128, help='minibatch size')
    parser.add_argument(
        '--grad_clip',
        type=float,
        default=5,  # 5.,
        help='clip gradients at this value')

    parser.add_argument(
        '--learning_rate', type=float, default=10e-4, help='learning rate')

    parser.add_argument(
        '--learning_rate_decay_every',
        type=int,
        default=200,
        help='every how many iterations thereafter to drop LR?(in epoch)')
    parser.add_argument('--learning_rate_decay_rate', type=float, default=0.8)

    parser.add_argument(
        '--optim_alpha', type=float, default=0.9, help='alpha for adam')

    parser.add_argument(
        '--optim_beta', type=float, default=0.999, help='beta used for adam')

    parser.add_argument(
        '--optim_epsilon',
        type=float,
        default=1e-8,
        help='epsilon that goes into denominator for smoothing')

    parser.add_argument(
        '--weight_decay',
        type=float,
        default=5e-4,
        help='weight_decay. strength of weight regularization')

    parser.add_argument(
        '--checkpoint_path',
        type=str,
        default='save_acaps',
        help='directory to store checkpointed models')

    parser.add_argument(
        '--gpu', type=str, default='0', help='gpu device number')
    
    parser.add_argument(
        '--output_dir', type=str, help='output directory of npy',
        default='/home/cxu-serve/p1/rohan27/research/audiocaps/OriginalACAPS/code/data/audiocaps/features/'
    )

    # --------------eval arguments-------------------
    model_folder = './save_acaps/'

    parser.add_argument('--recover_opt', type=str, default=f'{model_folder}/opt_info.json',
                        help='recover train opts from saved opt_json')
    parser.add_argument('--dump_json', type=int, default=1,
                        help='Dump json with predictions into vis folder? (1=yes,0=no)')
    parser.add_argument('--results_path', type=str, default='results_acaps/')
    parser.add_argument('--dump_path', type=int, default=0,
                        help='Write image paths along with predictions into vis json? (1=yes,0=no)')
    parser.add_argument('--batch_size_eval', type=int, default=128,
                        help='minibatch size')
    parser.add_argument('--beam_size', type=int, default=5,
                        help='beam size to use for beam search')
    parser.add_argument('--beam', type=str2bool, default=False,
                        help='Use beam search?')

    args = parser.parse_args()

    return args
