import os

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from rich.progress import track

from fid_metrics import (
    ImageDataset,
    ImageSequenceDataset,
    VideoDataset,
    build_inception,
    build_inception3d,
    calculate_fid,
    calculate_fid_per_frame,
    is_image_dir_path,
    is_video_path,
    postprocess_i2d_pred,
)


def build_loaders(type, paths, cfg):
    dls = []
    for path in paths:
        bs = cfg.batch_size
        dataset_cfgs = cfg.get('dataset')

        if is_video_path(path):
            if type == 'fid':
                if dataset_cfgs:
                    dataset_cfgs = dict(dataset_cfgs)
                    dataset_cfgs['sequence_length'] = bs
                else:
                    dataset_cfgs = {'sequence_length': bs}
                bs = 1
            elif type == 'fid_per_frame':
                # One item == one video's full [start_frame, end_frame) window,
                # so position t in every clip is the same absolute frame index.
                dataset_cfgs = dict(dataset_cfgs) if dataset_cfgs else {}
                start = int(dataset_cfgs.get('start_frame', 0))
                end = dataset_cfgs.get('end_frame')
                assert end is not None, 'fid_per_frame requires data.dataset.end_frame'
                dataset_cfgs['sequence_length'] = int(end) - start
                dataset_cfgs['no_overlap'] = True
                bs = 1
            C = VideoDataset
        elif is_image_dir_path(path):
            if type == 'fid_per_frame':
                raise NotImplementedError('fid_per_frame supports video paths only')
            C = ImageDataset if type == 'fid' else ImageSequenceDataset
        else:
            raise NotImplementedError

        dataset = C(path, **dataset_cfgs) if dataset_cfgs else C(path)
        # Distance metrics are invariant to sample order. Sequential FVD access
        # lets each worker reuse a decoded video window instead of seeking into
        # the same mp4 once per short clip.
        dl = torch.utils.data.DataLoader(
            dataset, bs, shuffle=type == 'fid', num_workers=cfg.num_workers
        )
        dls.append(dl)
    return dls


def build_model(type, cfg):
    if type in ('fid', 'fid_per_frame'):
        return build_inception(cfg.dims)
    elif type == 'fvd':
        return build_inception3d(cfg.type, cfg.path)
    else:
        raise NotImplementedError


@hydra.main(config_path='../configs', config_name='config', version_base=None)
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    for metric_cfgs in cfg.metrics:
        type = metric_cfgs.type
        dls = build_loaders(type, cfg.paths, metric_cfgs.data)
        model = build_model(type, metric_cfgs.model).to(device).eval()

        feats = [[], []]
        for i, dl in enumerate(dls):
            if cfg.get('num_iters'):
                seq = range(cfg.num_iters // metric_cfgs.data.batch_size)
            else:
                seq = range(len(dl))
            dl = iter(dl)

            for _ in track(seq, description=f'{type}_{i}'):
                x = next(dl).to(device)
                if type in ('fid', 'fid_per_frame') and x.dim() == 5:
                    x = x.squeeze(0).transpose(0, 1)
                elif type == 'fvd':
                    x = x * 2 - 1  # [-1, 1]
                with torch.no_grad():
                    if type == 'fid':
                        pred = model(x)
                        pred = postprocess_i2d_pred(pred)
                    elif type == 'fid_per_frame':
                        # forward the window's frames in batches of batch_size
                        chunk = metric_cfgs.data.batch_size
                        preds = [postprocess_i2d_pred(model(x[s:s + chunk]))
                                 for s in range(0, x.shape[0], chunk)]
                        preds = [p if p.dim() > 1 else p.unsqueeze(0) for p in preds]
                        pred = torch.cat(preds, dim=0)
                    elif type == 'fvd':
                        if metric_cfgs.model.type == 'styleganv':
                            pred = model(x, return_features=True)
                        else:
                            pred = model(x)

                feats[i].append(pred.cpu().numpy())
            feats[i] = (
                np.stack(feats[i], axis=0)
                if type == 'fid_per_frame'
                else np.concatenate(feats[i], axis=0)
            )
        if type == 'fid_per_frame':
            calculate_fid_per_frame(
                *feats,
                start_frame=int(metric_cfgs.data.dataset.get('start_frame', 0)),
                output_csv=os.path.abspath(metric_cfgs.get('output_csv', 'fid_per_frame.csv')),
            )
        else:
            fid = calculate_fid(*feats)
            print(f'{type.upper()}: {fid}')


if __name__ == '__main__':
    main()
