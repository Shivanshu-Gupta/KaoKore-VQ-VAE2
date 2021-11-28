import os
import json
import shutil
import subprocess
import torch
import logging.config

from pathlib import Path
from datetime import datetime
from threading import Thread, Event
from itertools import cycle

def create_sprite_image(images):
    import numpy as np
    """Returns a sprite image consisting of images passed as argument. Images should be count x width x height"""
    if isinstance(images, list):
        images = np.array(images)
    img_h = images.shape[1]
    img_w = images.shape[2]
    n_plots = int(np.ceil(np.sqrt(images.shape[0])))

    spriteimage = np.ones((img_h * n_plots, img_w * n_plots))

    for i in range(n_plots):
        for j in range(n_plots):
            this_filter = i * n_plots + j
            if this_filter < images.shape[0]:
                this_img = images[this_filter]
                spriteimage[i * img_h:(i + 1) * img_h,
                j * img_w:(j + 1) * img_w] = this_img

    return spriteimage

def vector_to_matrix_mnist(mnist_digits):
    import numpy as np
    """Reshapes normal mnist digit (batch,28*28) to matrix (batch,28,28)"""
    return np.reshape(mnist_digits, (-1, 28, 28))


def invert_grayscale(mnist_digits):
    """ Makes black white, and white black """
    return 255 - mnist_digits


def get_sprite_image(to_visualise, do_invert=True):
    to_visualise = vector_to_matrix_mnist(to_visualise)
    if do_invert:
        to_visualise = invert_grayscale(to_visualise)
    return create_sprite_image(to_visualise)

def setup_logging_from_args(args):
    """
    Calls setup_logging, exports args and creates a ResultsLog class.
    Can resume training/logging if args.resume is set
    """
    def set_args_default(field_name, value):
        if hasattr(args, field_name):
            return eval('args.' + field_name)
        else:
            return value

    # Set default args in case they don't exist in args
    resume = set_args_default('resume', False)
    save_name = set_args_default('save_name', '')
    results_dir = Path(set_args_default('results_dir', './results'))

    if save_name is '':
        save_name = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    save_path = results_dir / save_name
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    os.makedirs(save_path, exist_ok=True)
    log_file = save_path / 'log.txt'

    setup_logging(log_file, resume)
    export_args(args, save_path)
    return save_path


def setup_logging(log_file='log.txt', resume=False):
    """
    Setup logging configuration
    """
    if os.path.isfile(log_file) and resume:
        file_mode = 'a'
    else:
        file_mode = 'w'

    root_logger = logging.getLogger()
    if root_logger.handlers:
        root_logger.removeHandler(root_logger.handlers[0])
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S",
                        filename=log_file,
                        filemode=file_mode)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


def export_args(args, save_path):
    """
    args: argparse.Namespace
        arguments to save
    save_path: string
        path to directory to save at
    """
    os.makedirs(save_path, exist_ok=True)
    json_file_name = save_path / 'args.json'
    with open(json_file_name, 'w') as fp:
        json.dump(dict(args._get_kwargs()), fp, sort_keys=True, indent=4)


def save_checkpoint(state, is_best, path='.', filename='checkpoint.pth.tar', save_all=False):
    path = Path(path)
    filename = path / filename
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, path / 'model_best.pth.tar')
    if save_all:
        shutil.copyfile(filename, path / f"checkpoint_epoch_{state['epoch']}.pth.tar")
