from dataset.imagenet import build_imagenet, build_imagenet_code
from dataset.coco import build_coco
from dataset.openimage import build_openimage
from dataset.pexels import build_pexels
from dataset.t2i import build_t2i, build_t2i_code, build_t2i_image
from dataset.cmip6 import build_cmip6_nc
from torch.utils.data import DistributedSampler


def build_dataset(args, **kwargs):
    # images
    if args.dataset == 'imagenet':
        return build_imagenet(args, **kwargs)
    if args.dataset == 'imagenet_code':
        return build_imagenet_code(args, **kwargs)
    if args.dataset == 'coco':
        return build_coco(args, **kwargs)
    if args.dataset == 'openimage':
        return build_openimage(args, **kwargs)
    if args.dataset == 'pexels':
        return build_pexels(args, **kwargs)
    if args.dataset == 't2i_image':
        return build_t2i_image(args, **kwargs)
    if args.dataset == 't2i':
        return build_t2i(args, **kwargs)
    if args.dataset == 't2i_code':
        return build_t2i_code(args, **kwargs)
    
    raise ValueError(f'dataset {args.dataset} is not supported')

def build_dataset_sampler(args, *, 
                          world_size, 
                          rank,
                          shuffle,
                          seed,
                          **kwargs):
    if args.dataset not in ["cmip6_nc"]:
        dataset = build_dataset(args, **kwargs)
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle,
            seed=seed
        )
        return dataset, sampler
    else:
        dataset, sampler = build_cmip6_nc(args, 
                                          world_size=world_size,
                                          rank=rank, 
                                          shuffle=shuffle,
                                          seed=seed, 
                                          **kwargs)
