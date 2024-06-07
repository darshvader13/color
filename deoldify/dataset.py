import sys
sys.path.insert(0, '../.')

from fastai import *
from fastai.vision import *
from DeOldify.fastai.vision.data import ImageDataBunch
from DeOldify.fastai.vision.transform import get_transforms
from fastai.vision.data import ImageDataBunch
from fastai.vision.transform import get_transforms
from pathlib import Path

def get_colorize_data(
    sz: int,
    bs: int,
    path: Path,  # Now only one path is used for both crappy and good images
    random_seed: int = None,
    keep_pct: float = 1.0,
    num_workers: int = 8,
    stats: tuple = imagenet_stats,
    xtra_tfms=[],
) -> ImageDataBunch:
    
    src = (
        ImageImageList.from_folder(path, convert_mode='RGB')
        .use_partial_data(sample_pct=keep_pct, seed=random_seed)
        .split_by_rand_pct(0.1, seed=random_seed)
    )
    data = (
        src.label_from_func(lambda x: x)  # The label is the same file in this setup
        .transform(
            get_transforms(
                use_grayscale=True, xtra_tfms=xtra_tfms
            ), 
            size=sz, 
            tfm_y=False
        )
        .databunch(bs=bs, num_workers=num_workers, no_check=True)
        .normalize(stats, do_y=True)
    )
    data.c = 3
    return data
def get_dummy_databunch() -> ImageDataBunch:
    path = Path('./dummy/')
    return get_colorize_data(
        sz=1, bs=1, path=path, keep_pct=0.001
    )