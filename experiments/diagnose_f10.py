"""Round-5 diagnosis part 2: F10 D30 per-run errors + convergence traces."""
import json
import numpy as np

BASE = r"E:/A_Works/paper/Evolutionary Algorithm/experiments/results/"
D30 = json.load(open(BASE + "cec_results_D30.json", encoding="utf-8"))
D30_OLD = json.load(open(BASE + "cec_results_D30_v3.3_10runs_backup.json", encoding="utf-8"))

F10 = "F10: Composition Function (Rastrigin+Griewank+Schwefel)"

print("=== F10 D30, 30-run per-run errors ===")
for algo in D30["algorithms"]:
    v = np.array(D30["all_errors"][algo][F10])
    print(f"\n{algo}: mean={v.mean():.4e} med={np.median(v):.4e} min={v.min():.4e} max={v.max():.4e}")
    print("  runs:", np.array2string(v, precision=3, max_line_width=250))
    print(f"  n_zero(<1e-8): {(v < 1e-8).sum()}/30")

print("\n=== F10 D30, v3.3 10-run backup ===")
for algo in D30_OLD["algorithms"]:
    v = np.array(D30_OLD["all_errors"][algo][F10])
    print(f"{algo}: mean={v.mean():.4e}  runs:", np.array2string(v, precision=3, max_line_width=250))

# --- convergence traces ---
ct = D30["convergence_traces"][F10]
print("\n=== F10 D30 convergence trace structure ===")
for algo, tr in ct.items():
    a = np.array(tr)
    print(f"\n{algo}: trace shape {a.shape}")
    # a might be (n_runs, n_checkpoints) or (n_checkpoints,)
    if a.ndim == 1:
        idx = np.arange(len(a)) * (1500 // len(a)) if len(a) > 1 else [0]
        print("  checkpoints:", list(zip(idx[::max(1, len(a)//10)], a[::max(1, len(a)//10)])))
    else:
        # median and best across runs at each checkpoint
        med = np.median(a, axis=0)
        print(f"  median across runs at {len(med)} checkpoints:")
        step = max(1, len(med) // 12)
        for i in range(0, len(med), step):
            print(f"    ckpt {i:4d}: median={med[i]:.3e}  best={a[:, i].min():.3e}  worst={a[:, i].max():.3e}")

# --- ruggedness traces ---
rt = D30.get("ruggedness_traces", {})
if F10 in rt:
    print("\n=== F10 ruggedness traces ===")
    for algo, tr in rt[F10].items() if isinstance(rt[F10], dict) else []:
        a = np.array(tr)
        print(f"{algo}: shape={a.shape}", np.array2string(np.asarray(a).flatten(), precision=3, max_line_width=250))
