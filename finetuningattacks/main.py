import torch
import argparse
from .config import *

if __name__ == '__main__':

    torch.cuda.empty_cache()

    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--attack')
    parser.add_argument('-d', '--dataset')
    args = parser.parse_args()


    if args.attack == 'tgda':
        # import here so that we don't load extra datasets
        config = TGDA_CONFIG[args.dataset]

        attack_tgda(**config)
        test_tgda(**config)
    
    elif args.attack == 'pc-gc':
        config = GC_CONFIG[args.dataset]

        attack_pc(**config)
        
        attack_gc(**config)
        test_gc(**config)
    
    elif args.attack == 'pc':
        config = GC_CONFIG[args.dataset]

        attack_pc(**config)
    
    elif args.attack == 'gc':
        config = GC_CONFIG[args.dataset]

        attack_gc(**config)
        test_gc(**config)
