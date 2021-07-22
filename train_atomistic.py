"""Trains an atomistic model on a dataset."""
import os
import shutil
from chemprop.parsing import parse_train_args
from chemprop.train import cross_validate_atomistic
from chemprop.utils import create_logger

if __name__ == '__main__':
    args = parse_train_args()
    args.atomistic=True
    logger = create_logger(name='train', save_dir=args.save_dir, quiet=args.quiet)

    # Copy db file to LOCAL storage
    clean_up  = lambda : None
    if args.slurm_job and os.environ.get("TMPDIR") is not None: 
        tmp_dir = os.environ.get("TMPDIR")
        _, file_name = os.path.split(args.data_path)
        old_loc = args.data_path
        new_loc = os.path.join(tmp_dir, file_name)
        shutil.copy2(args.data_path, new_loc)
        args.data_path = new_loc
        clean_up= lambda : os.remove(new_loc)

    cross_validate_atomistic(args, logger)
    clean_up()

        
