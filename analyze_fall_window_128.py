# -*- coding: utf-8 -*-
"""
Phan tich do rong su kien Fall bang max-SVM (accel) de tra loi:
    Window 128 mau (~1.28s @100Hz) co om tron pha free-fall + impact + settle khong?

Quy uoc (theo Readme.txt + pipeline step1):
  - Dung cot 1-3 (ADXL345, 13-bit, +-16g):  g = (2*16)/2^13 * AD = (32/8192)*AD
  - Gyro cot 4-6 (ITG3200) khong dung cho SVM impact; neu can dps = (4000/65536)*RD.
  - Downsample 200Hz -> 100Hz bang [::2] (giong df.iloc[::2]).
  - SVM = sqrt(ax^2 + ay^2 + az^2)  (don vi g; nghi 1g luc dung yen).
"""
import os, glob, numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
DATASET = os.path.join(ROOT, "SisFall_dataset")
OUTDIR = os.path.join(ROOT, "window128_analysis")
os.makedirs(OUTDIR, exist_ok=True)

FS = 100                 # Hz sau downsample
ACC_SCALE = 32.0 / 8192  # ADXL345 -> g
DEV_THR = 0.3            # |SVM-1g| > 0.3g coi la "dong" (active)
SETTLE_HOLD = 30         # so mau (0.3s) SVM phai on dinh moi coi la "settled"

def load_acc_g(path):
    """Doc file SisFall -> accel ADXL345 (g), downsample 100Hz. Tra ve (N,3)."""
    rows = []
    with open(path, "r", errors="ignore") as f:
        for line in f:
            line = line.strip().rstrip(";").strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 3:
                continue
            try:
                rows.append([float(parts[0]), float(parts[1]), float(parts[2])])
            except ValueError:
                continue
    a = np.asarray(rows, dtype=np.float64)[::2]   # downsample -> 100Hz
    return a * ACC_SCALE

def analyze(path):
    acc = load_acc_g(path)
    n = len(acc)
    if n < 64:
        return None
    svm = np.sqrt((acc ** 2).sum(axis=1))
    peak = int(np.argmax(svm))
    peak_val = float(svm[peak])
    dev = np.abs(svm - 1.0)

    # free-fall onset: diem SVM nho nhat trong 80 mau (0.8s) truoc peak (pha roi tu do ~0g)
    lo = max(0, peak - 80)
    ff_min_idx = lo + int(np.argmin(svm[lo:peak + 1])) if peak > lo else peak
    pre_samples = peak - ff_min_idx               # peak <- freefall onset

    # settle: tu peak di toi, diem dau tien ma dev < THR lien tuc SETTLE_HOLD mau
    settle_idx = n - 1
    i = peak
    while i < n:
        if dev[i] < DEV_THR:
            j = i
            while j < n and dev[j] < DEV_THR:
                j += 1
            if j - i >= SETTLE_HOLD:
                settle_idx = i
                break
            i = j
        else:
            i += 1
    post_samples = settle_idx - peak              # peak -> settled

    # "impact complex" = free-fall onset -> settled
    complex_span = settle_idx - ff_min_idx

    # Voi cua so 128 peak-centered (+-64): bao nhieu % nang luong impact (dev^2) nam trong?
    w = 128
    half = w // 2
    s, e = peak - half, peak + half
    seg = slice(max(0, s), min(n, e))
    # nang luong tren toan vung impact complex
    cs, ce = max(0, ff_min_idx), min(n, settle_idx + 1)
    energy_total = float((dev[cs:ce] ** 2).sum()) + 1e-9
    energy_in_win = float((dev[max(seg.start, cs):min(seg.stop, ce)] ** 2).sum())
    energy_frac = energy_in_win / energy_total

    return dict(file=os.path.basename(path), n=n, peak=peak, peak_val=peak_val,
                pre=pre_samples, post=post_samples, complex_span=complex_span,
                fits128=(pre_samples <= half and post_samples <= half),
                energy_frac128=energy_frac)

def main():
    files = sorted(glob.glob(os.path.join(DATASET, "*", "F*.txt")))
    res = [r for r in (analyze(f) for f in files) if r]
    print(f"Da phan tich {len(res)}/{len(files)} file Fall\n")

    pre = np.array([r["pre"] for r in res])
    post = np.array([r["post"] for r in res])
    span = np.array([r["complex_span"] for r in res])
    pk = np.array([r["peak_val"] for r in res])
    ef = np.array([r["energy_frac128"] for r in res])
    fits = np.array([r["fits128"] for r in res])

    def pct(a):
        return {p: round(float(np.percentile(a, p)), 1) for p in (50, 90, 95, 99, 99.9)}

    print("=== Don vi: SO MAU @100Hz (1 mau = 10ms). 128 win => +-64 mau quanh peak ===")
    print(f"Peak SVM (g):           p50={np.percentile(pk,50):.2f}  p95={np.percentile(pk,95):.2f}  max={pk.max():.2f}")
    print(f"PRE  (peak<-freefall):  {pct(pre)}   max={pre.max()}")
    print(f"POST (peak->settled):   {pct(post)}   max={post.max()}")
    print(f"COMPLEX span (ff->settle):{pct(span)} max={span.max()}")
    print()
    print(f"% Fall co impact-complex FIT trong win128 peak-centered: {100*fits.mean():.1f}%")
    print(f"Nang luong impact nam trong win128 (dev^2): p50={np.percentile(ef,50)*100:.1f}%  "
          f"p05={np.percentile(ef,5)*100:.1f}%  min={ef.min()*100:.1f}%")
    print(f"So Fall co >=95%% nang luong trong win128: {100*np.mean(ef>=0.95):.1f}%")
    print(f"So Fall co >=99%% nang luong trong win128: {100*np.mean(ef>=0.99):.1f}%")

    # So sanh nguong window khac nhau (theo COMPLEX span)
    print("\n=== % Fall co impact-complex <= W mau ===")
    for W in (96, 128, 160, 200, 256):
        print(f"  W={W:3d} ({W/FS:.2f}s): {100*np.mean(span<=W):.1f}%")

    # Luu CSV chi tiet
    csv = os.path.join(OUTDIR, "fall_window_stats.csv")
    with open(csv, "w") as f:
        f.write("file,n,peak,peak_val_g,pre,post,complex_span,fits128,energy_frac128\n")
        for r in res:
            f.write(f"{r['file']},{r['n']},{r['peak']},{r['peak_val']:.3f},"
                    f"{r['pre']},{r['post']},{r['complex_span']},"
                    f"{int(r['fits128'])},{r['energy_frac128']:.4f}\n")
    print(f"\n[CSV] {csv}")

    # === Cat thu window 128 peak-centered cho vai vi du dai dien (moi loai F muot) ===
    sample_files = []
    for code in ("F01", "F06", "F08", "F11", "F15"):  # walk/faint/getup/sitdown/sitting
        m = sorted(glob.glob(os.path.join(DATASET, "SA06", f"{code}_SA06_R01.txt")))
        sample_files += m
    cut_dir = os.path.join(OUTDIR, "cut_windows")
    os.makedirs(cut_dir, exist_ok=True)
    for path in sample_files:
        acc = load_acc_g(path)
        svm = np.sqrt((acc ** 2).sum(axis=1))
        peak = int(np.argmax(svm))
        s = max(0, peak - 64); e = s + 128
        if e > len(acc):
            e = len(acc); s = e - 128
        win = acc[s:e]                      # (128,3) accel g, peak-centered
        name = os.path.splitext(os.path.basename(path))[0]
        np.save(os.path.join(cut_dir, f"{name}_win128.npy"), win.astype(np.float32))
    print(f"[CUT] {len(sample_files)} window 128x3 (g) luu o {cut_dir}")

    # Plot SVM + vung win128 cho cac vi du
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(len(sample_files), 1, figsize=(11, 2.2 * len(sample_files)))
        if len(sample_files) == 1:
            axes = [axes]
        for ax, path in zip(axes, sample_files):
            acc = load_acc_g(path)
            svm = np.sqrt((acc ** 2).sum(axis=1))
            t = np.arange(len(svm)) / FS
            peak = int(np.argmax(svm))
            s = max(0, peak - 64); e = min(len(svm), s + 128)
            ax.plot(t, svm, lw=0.8)
            ax.axvspan(s / FS, e / FS, color="orange", alpha=0.25, label="win128")
            ax.axvline(peak / FS, color="r", ls="--", lw=0.8, label="peak SVM")
            ax.axhline(1.0, color="k", ls=":", lw=0.6)
            ax.set_title(f"{os.path.basename(path)}  peak={svm[peak]:.2f}g", fontsize=9)
            ax.set_ylabel("SVM (g)")
        axes[-1].set_xlabel("time (s)")
        axes[0].legend(fontsize=8, loc="upper right")
        plt.tight_layout()
        png = os.path.join(OUTDIR, "fall_svm_win128_examples.png")
        plt.savefig(png, dpi=110)
        print(f"[PNG] {png}")
    except Exception as ex:
        print("plot skip:", ex)

if __name__ == "__main__":
    main()
