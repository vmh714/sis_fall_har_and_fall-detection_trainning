# -*- coding: utf-8 -*-
"""So sanh 'free-fall pre-dip' giua Fall vs cac su kien va-dap KHONG-nga (D18 vap, D19 nhay, Run).
Muc tieu: free-fall (SVM tut ~0g truoc impact) co tach Fall khoi va-dap khong,
va no co nam trong pha PRE cua w128 (58 mau truoc dinh, left_ratio=0.45) khong?"""
import glob, numpy as np, pandas as pd
from pathlib import Path
RAW=Path("SisFall_dataset"); ACC=32.0/8192
def load(p):
    rows=[]
    for ln in open(p,errors="ignore"):
        ln=ln.strip().rstrip(";")
        if not ln: continue
        q=ln.split(",")
        if len(q)>=3:
            try: rows.append([float(q[0]),float(q[1]),float(q[2])])
            except: pass
    return (np.asarray(rows)[::2])*ACC

PRE=58   # so mau truoc dinh giu lai trong w128 (left_ratio 0.45)
def stats(patterns):
    premins=[]; peaks=[]
    files=[]
    for pat in patterns: files+=glob.glob(str(RAW/"*"/pat))
    for f in files:
        a=load(f)
        if len(a)<PRE+5: continue
        svm=np.sqrt((a**2).sum(1)); pk=int(svm.argmax())
        lo=max(0,pk-PRE)
        premins.append(float(svm[lo:pk+1].min()))   # do sau free-fall trong pha PRE cua w128
        peaks.append(float(svm[pk]))
    premins=np.array(premins); peaks=np.array(peaks)
    pc=lambda a,p:round(float(np.percentile(a,p)),2)
    return dict(n=len(premins),
        peak_p50=pc(peaks,50), peak_p95=pc(peaks,95),
        premin_p10=pc(premins,10), premin_p50=pc(premins,50), premin_p90=pc(premins,90),
        frac_freefall=round(float((premins<0.5).mean())*100,1))  # % co dip <0.5g (dau hieu free-fall ro)

groups={
 "FALL      (F01-15)":["F*.txt"],
 "STUMBLE   (D18)"   :["D18*.txt"],
 "JUMP      (D19)"   :["D19*.txt"],
 "RUN       (D03/04)":["D03*.txt","D04*.txt"],
 "WALK      (D01/02)":["D01*.txt","D02*.txt"],
}
print(f"{'NHOM':20s} {'n':>5s} {'peakP50':>8s} {'peakP95':>8s} | {'preMinP10':>9s} {'preMinP50':>9s} {'preMinP90':>9s} | %free-fall(<0.5g)")
for name,pats in groups.items():
    s=stats(pats)
    print(f"{name:20s} {s['n']:5d} {s['peak_p50']:8.2f} {s['peak_p95']:8.2f} | "
          f"{s['premin_p10']:9.2f} {s['premin_p50']:9.2f} {s['premin_p90']:9.2f} | {s['frac_freefall']:5.1f}%")
print("\nPRE =",PRE,"mau (~0.58s) -> phan FREE-FALL nay NAM TRONG w128.")
print("Doc: Fall co preMin thap (free-fall ~0g) & %free-fall cao; va-dap khong-nga thi preMin ~1g & %free-fall thap.")
