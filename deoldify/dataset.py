from fastai import *
from fastai.core import *
from fastai.vision.transform import get_transforms
from fastai.vision.data import ImageImageList, ImageDataBunch, imagenet_stats


def get_colorize_data(
    sz: int,
    bs: int,
    bw_path: Path,
    color_path: Path,
    random_seed: int = None,
    keep_pct: float = 1.0,
    num_workers: int = 8,
    stats: tuple = imagenet_stats,
    xtra_tfms=[],
) -> ImageDataBunch:
    
    src = (
        ImageImageList.from_folder(bw_path, convert_mode='RGB')
        .use_partial_data(sample_pct=keep_pct, seed=random_seed)
        .split_by_rand_pct(0.1, seed=random_seed)
    )

    data = (
        src.label_from_func(lambda x: color_path / x.relative_to(bw_path))
        .transform(
            get_transforms(),
            size=sz,
            tfm_y=False,
        )
        .databunch(bs=bs, num_workers=num_workers, no_check=False)
        .normalize(stats, do_y=False)
    )

    data.c = 3
    return data


def get_dummy_databunch() -> ImageDataBunch:
    path = Path('./dummy/')
    return get_colorize_data(
        sz=1, bs=1, bw_path=path, color_path=path, keep_pct=0.001
    )