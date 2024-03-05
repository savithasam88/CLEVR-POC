import os
import argparse

import numpy as np
import torch


class BaseOptions():
    """Base option class"""

    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('--run_dir', default='_scratch/test_run', type=str, help='experiment directory')
        self.parser.add_argument('--dataset', default='clevr', type=str, help='select dataset, options: clevr, clevr-humans')
        # Dataloader
        self.parser.add_argument('--shuffle', default=1, type=int, help='shuffle dataset')
        self.parser.add_argument('--num_workers', default=8, type=int, help='number of workers for loading data')
        # Run
        self.parser.add_argument('--manual_seed', default=None, type=int, help='manual seed')
        self.parser.add_argument('--gpu_ids', default='0', type=str, help='ids of gpu to be used')
        self.parser.add_argument('--visualize', default=0, type=int, help='visualize experiment')
        # Dataset catalog
        # - CLEVR
        self.parser.add_argument('--main_root',default = None, type=str, help='main root')
        self.parser.add_argument('--ns_vqa_root',default=None, type=str, help='ns-vqa root')
        self.parser.add_argument('--clevr_train_scene_path', default=None,
                                 type=str, help='path to clevr train scenes')
        
        self.parser.add_argument('--clevr_val_scene_path', default=None,
                                 type=str, help='path to clevr val scenes')
                                 
        self.parser.add_argument('--clevr_constraint_scene_path', default=None,
                                 type=str, help='path to constraints')
        
        
        self.parser.add_argument('--clevr_train_question_path', default=None,
                                 type=str, help='path to clevr train questions')
        self.parser.add_argument('--clevr_val_question_path', default=None,
                                 type=str, help='path to clevr val questions')
        self.parser.add_argument('--train_image_path', default=None,
                                 type=str, help='path to train images clevr ')
        self.parser.add_argument('--val_image_path', default=None,
                                 type=str, help='path to val images clevr ')
        self.parser.add_argument('--clevr_vocab_path', default=None,
                                 type=str, help='path to clevr vocab')
        self.parser.add_argument('--data_folder_name', default=None,
                                 type=str, help='data folder')
        #self.parser.add_argument('--clevr_vocab_path', default='../data/reason/clevr_h5_500/clevr_vocab.json',
        #                         type=str, help='path to clevr vocab')
        

    def parse(self):
        # Instantiate option
        self.opt = self.parser.parse_args()

        # Parse gpu id list
        str_gpu_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_gpu_ids:
            if str_id.isdigit() and int(str_id) >= 0:
                self.opt.gpu_ids.append(int(str_id))
        if len(self.opt.gpu_ids) > 0 and torch.cuda.is_available():
            torch.cuda.set_device(self.opt.gpu_ids[0])
        else:
            print('| using cpu')
            self.opt.gpu_ids = []

        # Set manual seed
        if self.opt.manual_seed is not None:
            torch.manual_seed(self.opt.manual_seed)
            if len(self.opt.gpu_ids) > 0 and torch.cuda.is_available():
                torch.cuda.manual_seed(self.opt.manual_seed)

        # Print and save options
        args = vars(self.opt)
        print('| options')
        for k, v in args.items():
            print('%s: %s' % (str(k), str(v)))
        if not os.path.isdir(self.opt.run_dir):
            os.makedirs(self.opt.run_dir)
        if self.is_train:
            file_path = os.path.join(self.opt.run_dir, 'train_opt.txt')
        else:
            file_path = os.path.join(self.opt.run_dir, 'test_opt.txt')
        with open(file_path, 'wt') as fout:
            fout.write('| options\n')
            for k, v in args.items():
                fout.write('%s: %s\n' % (str(k), str(v)))

        return self.opt
