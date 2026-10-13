"""Plot historical v9 logged training loss against independent validation metrics."""
import json
import re
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
LOG=ROOT/'results/vla/training_logs/smolvla_execution_aligned_full_lora16_20261011_truth_v9.log'
OUT=ROOT/'results/vla/v9_loss_review_20261012_v17'
raw=LOG.read_bytes()
text=raw.decode('utf-16' if raw.startswith((b'\xff\xfe',b'\xfe\xff')) else 'utf-8-sig',errors='replace')
matches=list(re.finditer(r'step:(\S+)\s+smpl:.*?loss:([\d.]+).*?lr:([\deE.+-]+)',text))
config=json.loads((ROOT/'results/vla/training/smolvla_execution_aligned_full_lora16_20261011_truth_v9/checkpoints/006000/pretrained_model/train_config.json').read_text(encoding='utf-8'))
freq=config['log_freq']
if len(matches)*freq != config['steps']:
    raise ValueError('Missing/extra log intervals: cannot reconstruct exact steps from rounded K labels')
records=[]
for i,m in enumerate(matches):
    step=(i+1)*freq
    progress=list(re.finditer(r'(\d+)/6000',text[max(0,m.start()-1000):m.start()]))
    if progress and int(progress[-1].group(1)) != step:
        raise ValueError('Progress bar and log interval disagree')
    records.append({'step':step,'loss':float(m[2]),'lr':float(m[3])})
windows=[]
for end in range(1000,6001,1000):
    loss=[r['loss'] for r in records if end-1000 < r['step'] <= end]
    windows.append({'steps':[end-999,end],'mean_loss':float(np.mean(loss)),
                    'min_logged_loss':min(loss),'max_logged_loss':max(loss)})
scan=json.loads((ROOT/'results/vla/fresh_lora_comparison_20261011_v10/continued_validation240.json').read_text(encoding='utf-8-sig'))
validation=[{k:r[k] for k in ('step','position_mae_cm','gripper_sign_accuracy','orientation_mae_deg')} for r in scan['results']]
tail=[r for r in records if r['step']>3000]
slope=float(np.polyfit([r['step'] for r in tail],[r['loss'] for r in tail],1)[0]*1000)
payload={'source_log':str(LOG),'log_intervals':len(records),'log_frequency':freq,
         'records':records,'windows_1000_steps':windows,'tail_3000_steps_slope_per_1000_steps':slope,
         'fixed_validation_240_frames':validation,'updates':6000,'batch_size':1,
         'training_frames':27338,'nominal_frame_epochs':6000/27338,
         'caveats':['Logged losses are rounded 100-update averages, not each-step loss.',
                    'Flow-matching training loss is not position error or closed-loop success.',
                    'Weighted replacement sampling means nominal epochs do not measure unique frame coverage.',
                    'v9 started from an already fine-tuned v7 adapter; 6000 is only this continuation stage.']}
OUT.mkdir(parents=True,exist_ok=True)
(OUT/'loss_review.json').write_text(json.dumps(payload,indent=2),encoding='utf-8')
fig,axes=plt.subplots(1,2,figsize=(12,4.2),layout='constrained')
axes[0].plot([r['step'] for r in records],[r['loss'] for r in records],color='#83a7c9',lw=1,label='Logged 100-update mean')
axes[0].plot([500,1500,2500,3500,4500,5500],[w['mean_loss'] for w in windows],'o-',color='#174c83',lw=2,label='1000-update mean')
axes[0].set(xlabel='Optimizer updates in v9 stage',ylabel='Weighted flow-matching loss',title='v9 training: early drop, noisy plateau')
axes[0].legend(fontsize=8)
axes[1].plot([v['step'] for v in validation],[v['position_mae_cm'] for v in validation],'o-',color='#207d5a')
for v in validation:
    axes[1].annotate(f"{v['position_mae_cm']:.3f} cm",(v['step'],v['position_mae_cm']),xytext=(0,10),textcoords='offset points',ha='center')
axes[1].set(xlabel='Optimizer updates in v9 stage',ylabel='Position action error (cm)',title='Fixed validation: 240 frames',ylim=(1.35,1.82))
axes[1].set_xlim(0,6600)
for ax in axes: ax.grid(alpha=.2)
fig.savefig(OUT/'v9_loss_curve.png',dpi=160)
plt.close(fig)
print(json.dumps({k:v for k,v in payload.items() if k!='records'},indent=2))
