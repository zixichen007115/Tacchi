import os
import argparse
from solver import Solver
from data_loader import get_loader
from torch.backends import cudnn

def str2bool(v):
    return v.lower() in ('true')

def main(config):
    # For fast training.
    cudnn.benchmark = True

    # Create directories if not exist.
    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir)
    if not os.path.exists(config.model_save_dir):
        os.makedirs(config.model_save_dir)

    image_dir = "data_g/"+config.img_kind

    # Data loader.
    loader = get_loader(image_dir, 
        config.crop_size, config.image_size, config.batch_size,
        config.mode, config.num_workers)



    # Solver for training and testing StarGAN.
    solver = Solver(loader, config)

    if config.mode == 'train':
        solver.train()
    elif config.mode == 'test':
        solver.test()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Model configuration.

    parser.add_argument('--dim', type=int, default=21, help='dimension of domain labels')
    parser.add_argument('--crop_size', type=int, default=440, help='crop size for the dataset')
    parser.add_argument('--image_size', type=int, default=256, help='image resolution')

    
    # Training configuration.
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--num_iters', type=int, default=1000, help='number of total iterations for training D')
    parser.add_argument('--r_lr', type=float, default=0.001, help='learning rate for ResNet')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')
    parser.add_argument('--resume_iters', type=int, default=None, help='resume training from this step')
    parser.add_argument('--model_name', type=str, default="", help=' the name of model')

    # Test configuration.
    parser.add_argument('--test_iters', type=int, default=1000, help='test model from this step')

    # Miscellaneous.
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--use_tensorboard', type=str2bool, default=False)

    # Directories.
    parser.add_argument('--img_kind', type=str, default='real')
    # , choices=['real', 'sim_1', 'sim_10', 'sim_100', 'sim_DG21'])
    parser.add_argument('--log_dir', type=str, default='result/logs')
    parser.add_argument('--model_save_dir', type=str, default='result/models')

    # Step size.
    parser.add_argument('--log_step', type=int, default=10)
    parser.add_argument('--model_save_step', type=int, default=1000)

    config = parser.parse_args()
    print(config)
    main(config)

