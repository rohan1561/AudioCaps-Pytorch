from preprocess import vToA
import argparse
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_dir', type=str, default='./rough/missing_aud/',
    help='The video dir that one would like to extract audio file from')
    parser.add_argument('--output_dir', type=str, default='./rough/aud',
    help='The file output directory')
    parser.add_argument('--output_channels', type=int, default=1, 
    help='The number of output audio channels, default to 1')
    parser.add_argument('--output_frequency', type=int, default=16000, 
    help='The output audio frequency in Hz, default to 16000')
    parser.add_argument('--band_width', type=int, default=160, 
    help='Bandwidth specified to sample the audio (unit in kbps), default to 160')
    parser.add_argument('--model', type=str, default='resnet152', 
    help='The pretrained model to use for extracting image features, default to resnet152')
    parser.add_argument('--gpu', type=str, default='0', 
    help='The CUDA_VISIBLE_DEVICES argument, default to 0')
    parser.add_argument('--n_frame_steps', type=int, default=80,
    help='The number of frames to extract from a single video')
    opt = parser.parse_args()
    opt=vars(opt)

    if not os.path.exists(opt['output_dir']):
        os.mkdir(opt['output_dir'])
    vToA(opt)

if __name__ == '__main__':
    main()

