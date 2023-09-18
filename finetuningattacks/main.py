import torch
import argparse
from .config import *

if __name__ == '__main__':
    TQDM_DISABLE=1

    torch.cuda.empty_cache()

    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--attack')
    parser.add_argument('-d', '--dataset')
    args = parser.parse_args()

    # ----------------------------------------------------------------------------------
    # ------------------------------------- TGDA ---------------------------------------
    # ----------------------------------------------------------------------------------

    if args.attack == 'tgda':
        # import here so that we don't load extra datasets
        config = TGDA_CONFIG[args.dataset]

        attack_tgda(**config)
        test_tgda(**config)

    # ----------------------------------------------------------------------------------
    # ----------------------------------- PC + GC --------------------------------------
    # ----------------------------------------------------------------------------------

    elif args.attack == 'pc-gc':
        pc_config = PC_CONFIG[args.dataset]
        attack_pc(**pc_config)

        gc_config = GC_CONFIG[args.dataset]
        attack_gc(**gc_config)
        test_gc(**gc_config)
    
    elif args.attack == 'pc':
        config = PC_CONFIG[args.dataset]

        attack_pc(**config)
    
    elif args.attack == 'gc':
        config = GC_CONFIG[args.dataset]

        attack_gc(**config)
        test_gc(**config)

    elif args.attack == 'gc-test':
        config = GC_CONFIG[args.dataset]

        test_gc(**config)

    # ----------------------------------------------------------------------------------
    # -------------------------------------- DM ----------------------------------------
    # ----------------------------------------------------------------------------------

    elif args.attack == 'dm':
        config = DM_CONFIG[args.dataset]

        attack_dm(**config)
        test_dm(**config)
    
    elif args.attack == 'dm-test':
        config = DM_CONFIG[args.dataset]

        test_dm(**config)


    # ----------------------------------------------------------------------------------
    # -------------------------------------- LD ----------------------------------------
    # ----------------------------------------------------------------------------------

    elif args.attack == 'ld':
        config = LD_CONFIG[args.dataset]
        # for alpha in [0, 1, 10, 100, 1000, 10000, 100000, 1000000, 10000000]:
        #     print(f"\n\n -------------------------------------------------------------------------------------")
        #     print(f"-------------------------------------- {alpha} --------------------------------------")
        #     print(f"-------------------------------------------------------------------------------------\n\n")
        #     if alpha == 0:
        #         config['poisoned_batch_size'] = 0
        #     else:
        #         config['poisoned_batch_size'] = 1
        #         config['magnitude'] = alpha

        #     attack_ld(**config)

        attack_ld(**config)
    
    elif args.attack == 'ri':
        config = RI_CONFIG[args.dataset]
        attack_ri(**config)

