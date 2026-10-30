"""Measure ExoDINO3 full encoder+decoder inference latency."""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
import numpy as np
import torch, yaml
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
from exodino3.model import ExoDINO3


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,default=Path('benchmark_fps.txt'))
    parser.add_argument('--device',choices=['auto','cuda','cpu'],default='auto')
    parser.add_argument('--precision',choices=['fp32','fp16','bf16'],default='fp32')
    args=parser.parse_args()
    if args.device=='cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA was requested but is unavailable.')
    device=torch.device('cuda' if args.device=='cuda' or (args.device=='auto' and torch.cuda.is_available()) else 'cpu')
    config=yaml.safe_load(args.config.read_text()); model_cfg=config['model']
    variant=model_cfg.get('model_variant','exodino3')
    if variant!='exodino3': raise ValueError('benchmark_fps.py is defined for ExoDINO3 configurations.')
    kwargs={k:v for k,v in model_cfg.items() if k!='model_variant'}
    if device.type!='cuda' and args.precision!='fp32': raise ValueError('fp16 and bf16 benchmark precision require CUDA.')
    model=ExoDINO3(**kwargs).to(device)
    state=torch.load(args.checkpoint,map_location=device,weights_only=False)
    model.load_state_dict(state['model_state_dict'],strict=True); model.eval()
    input_tensor=torch.randn(1,3,504,504,device=device)
    amp_dtype={'fp16':torch.float16,'bf16':torch.bfloat16}.get(args.precision)
    def forward():
        if amp_dtype is None: return model(input_tensor)
        with torch.autocast(device_type=device.type,dtype=amp_dtype,enabled=True): return model(input_tensor)
    timings=[]
    with torch.inference_mode():
        for _ in range(50): forward()
        if device.type=='cuda': torch.cuda.synchronize()
        if device.type=='cuda':
            events=[]
            for _ in range(200):
                begin=torch.cuda.Event(enable_timing=True); end=torch.cuda.Event(enable_timing=True)
                begin.record(); forward(); end.record(); events.append((begin,end))
            torch.cuda.synchronize()
            timings=[begin.elapsed_time(end) for begin,end in events]
        else:
            for _ in range(200):
                begin=time.perf_counter(); forward(); timings.append((time.perf_counter()-begin)*1000)
    info=f'''GPU: {torch.cuda.get_device_name(device) if device.type=='cuda' else 'not used (CPU)'}\nPyTorch: {torch.__version__}\nCUDA: {torch.version.cuda or 'not available'}\nPrecision: {args.precision}\nInput shape: [1, 3, 504, 504]\nWarm-up iterations: 50\nTimed iterations: 200\nMean latency: {float(np.mean(timings)):.4f} ms\nMedian latency: {float(np.median(timings)):.4f} ms\nFPS (from mean latency): {1000/float(np.mean(timings)):.3f}\n'''
    args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(info,encoding='utf-8'); print(info,end=''); print(f'Saved report: {args.output}')

if __name__=='__main__': main()
