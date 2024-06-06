import sys
sys.path.insert(0, '../.')

from fastai import *
from fastai.core import *
from fastai.vision.transform import get_transforms
from fastai.vision.data import ImageImageList, ImageDataBunch, imagenet_stats
from PIL import Image

def convert_to_greyscale(img: Image) -> Image:
    return img.convert("L").convert("RGB")

def get_colorize_data(
    sz: int,
    bs: int,
    colored_path: Path,
    random_seed: int = None,
    keep_pct: float = 1.0,
    num_workers: int = 8,
    stats: tuple = imagenet_stats,
    xtra_tfms=[],
) -> ImageDataBunch:

    src = (
        ImageImageList.from_folder(colored_path, convert_mode='RGB')
        .use_partial_data(sample_pct=keep_pct, seed=random_seed)
        .split_by_rand_pct(0.1, seed=random_seed)
    )

    data = (
        src.label_from_func(lambda x: x)
        .transform(
            get_transforms(
                max_zoom=1.2, max_lighting=0.5, max_warp=0.25, xtra_tfms=xtra_tfms
            ),
            size=sz,
            tfm_y=True,
        )
        .databunch(bs=bs, num_workers=num_workers, no_check=True)
        .normalize(stats, do_y=True)
    )

    # Apply greyscale conversion to inputs only for training set
    data.train_ds.x.add_tfm(convert_to_greyscale)

    data.c = 3
    return data

def get_dummy_databunch() -> ImageDataBunch:
    path = Path('./dummy/')
    return get_colorize_data(
        sz=1, bs=1, colored_path=path, keep_pct=0.001
    )
