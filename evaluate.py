"""Evaluate one ExoDINO checkpoint on its configured held-out fold."""
from __future__ import annotations
import argparse, csv, os, sys
from pathlib import Path
import torch, yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from exodino.data import ExoDiscSemanticDataset
from exodino.model import DinoSurgicalSegmenter
from exodino2.model import ExoDINO2FT3
from exodino3.data import ExoDINO3Dataset
from exodino3.model import ExoDINO3
from exodino4.model import ExoDINO4


def model_from_config(config):
    model_cfg=config['model']; variant=model_cfg.get('model_variant','exodino')
    kwargs={k:v for k,v in model_cfg.items() if k!='model_variant'}
    if variant=='exodino': return DinoSurgicalSegmenter(**kwargs)
    if variant=='exodino2_ft3': return ExoDINO2FT3(**kwargs)
    if variant=='exodino3': return ExoDINO3(**kwargs)
    if variant=='exodino4': return ExoDINO4(**kwargs,backbone=config['backbone'])
    raise ValueError(f'Unsupported model variant: {variant}')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,default=Path('outputs/evaluation.csv'))
    parser.add_argument('--batch-size',type=int,default=4)
    args=parser.parse_args()
    config=yaml.safe_load(args.config.read_text()); data=config['data']; model_cfg=config['model']
    data['root_dir']=os.path.expandvars(data['root_dir'])
    if '$' in data['root_dir']: raise ValueError('Set EXODISC_DATA_ROOT or configure data.root_dir.')
    cls=ExoDINO3Dataset if model_cfg.get('model_variant') in {'exodino3','exodino4'} else ExoDiscSemanticDataset
    kw=dict(root_dir=data['root_dir'],operation_ids=data['val_operations'],image_size=data['image_size'],
            num_classes=model_cfg['num_classes'],operation_prefix=data['operation_prefix'],
            images_dir_name=data['images_dir_name'],masks_dir_name=data['masks_dir_name'],
            transform=None,cutmix_probability=0.0,include_empty_masks=True)
    if cls is ExoDINO3Dataset: kw['overlay_mask']=config.get('overlay_mask')
    dataset=cls(**kw); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loader=DataLoader(dataset,batch_size=args.batch_size,shuffle=False,num_workers=data['num_workers'],pin_memory=device.type=='cuda')
    model=model_from_config(config).to(device)
    checkpoint=torch.load(args.checkpoint,map_location=device,weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'],strict=True); model.eval()
    n=model_cfg['num_classes']; confusion=torch.zeros((n,n),dtype=torch.int64)
    with torch.inference_mode():
        for batch in loader:
            images=batch['image'].to(device); target=batch['mask'].long().to(device)
            pred=model(images).argmax(1)
            confusion += torch.bincount((target*n+pred).reshape(-1).cpu(),minlength=n*n).reshape(n,n)
    tp=confusion.diag().double(); gt=confusion.sum(1).double(); predicted=confusion.sum(0).double()
    union=gt+predicted-tp; present=gt>0; iou=torch.full((n,),float('nan'),dtype=torch.double)
    iou[present]=tp[present]/union[present].clamp_min(1)
    foreground=present.clone(); foreground[0]=False
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('w',newline='',encoding='utf-8') as f:
        writer=csv.writer(f); writer.writerow(['class_id','class_name','ground_truth_present','iou'])
        for i,name in enumerate(data['class_names']): writer.writerow([i,name,bool(present[i]),'' if not present[i] else f'{iou[i].item():.8f}'])
        writer.writerow(['macro_foreground_present_classes','','',float(iou[foreground].mean()) if foreground.any() else ''])
    print(f'Wrote {args.output}; foreground macro mIoU={iou[foreground].mean().item() if foreground.any() else float("nan"):.6f}')

if __name__=='__main__': main()
