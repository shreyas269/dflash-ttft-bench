"""Tabulate a comprehensive TTFT sweep (no_spec / eagle3 / dflash_optimized /
dflash_original).

Expects <out-dir> to contain {no_spec,eagle3,dflash_optimized,dflash_original}.json
as written by ttft_comprehensive.py --json-out.
"""
import argparse
import json
import os

MODES = ["no_spec", "eagle3", "dflash_optimized", "dflash_original"]
SERIAL_SCENARIOS = [
    "SERIAL_COLD",
    "CACHE_HIT_MISS",
    "CACHE_HIT_HIT",
    "SHARED_PREFIX_FIRST",
    "SHARED_PREFIX_LATER",
]


def load(out_dir, mode):
    p = os.path.join(out_dir, f"{mode}.json")
    if not os.path.isfile(p):
        return None
    with open(p) as f:
        return json.load(f)


def pct_delta(a, b):
    if b == 0:
        return float("nan")
    return 100.0 * (a - b) / b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    data = {m: load(args.out_dir, m) for m in MODES}

    print("\n=== modes ===")
    for m in MODES:
        d = data[m]
        if d is None:
            print(f"  {m:<18} MISSING")
            continue
        ns = d.get("num_spec", "?")
        tgt = d.get("target", "?")
        print(f"  {m:<18} num_spec={ns}  target={tgt}")

    for scenario in SERIAL_SCENARIOS:
        print()
        print(f"=== {scenario} ===")
        print(
            f"  {'mode':<18} {'n':>3} {'min':>8} {'p50':>8} {'mean':>8} "
            f"{'p90':>8} {'p95':>8} {'p99':>8} {'max':>8} {'max/p50':>8}  (ms)"
        )
        for mode in MODES:
            d = data[mode]
            if d is None:
                print(f"  {mode:<18}  MISSING")
                continue
            s = d["scenarios"].get(scenario)
            if s is None or s.get("n", 0) == 0:
                print(f"  {mode:<18}  no data")
                continue
            print(
                f"  {mode:<18} {s['n']:>3} "
                f"{s['min_ms']:>8.1f} {s['p50_ms']:>8.1f} {s['mean_ms']:>8.1f} "
                f"{s['p90_ms']:>8.1f} {s['p95_ms']:>8.1f} {s['p99_ms']:>8.1f} "
                f"{s['max_ms']:>8.1f} {s['ratio_max_over_p50']:>7.2f}x"
            )

        e = (data.get("eagle3") or {}).get("scenarios", {}).get(scenario)
        do = (data.get("dflash_original") or {}).get("scenarios", {}).get(scenario)
        dn = (data.get("dflash_optimized") or {}).get("scenarios", {}).get(scenario)

        def row(label, a, b):
            if not a or not b or not a.get("n") or not b.get("n"):
                return
            p50 = pct_delta(a["p50_ms"], b["p50_ms"])
            p90 = pct_delta(a["p90_ms"], b["p90_ms"])
            p99 = pct_delta(a["p99_ms"], b["p99_ms"])
            mx = pct_delta(a["max_ms"], b["max_ms"])
            print(
                f"  Δ {label:<38} p50={p50:+5.1f}%  p90={p90:+5.1f}%  "
                f"p99={p99:+5.1f}%  max={mx:+5.1f}%"
            )

        row("dflash_original vs eagle3", do, e)
        row("dflash_optimized vs eagle3", dn, e)
        row("dflash_optimized vs dflash_original", dn, do)

    print()
    print("=== CONCURRENT BATCH (total wall, per-req avg) ===")
    for b in ("BATCH_4", "BATCH_16"):
        print(f"  {b}:")
        print(
            f"    {'mode':<18} {'n':>3} {'total_ms':>10} {'per_req_ms':>11}"
        )
        for mode in MODES:
            d = data[mode]
            if d is None:
                continue
            s = d["scenarios"].get(b, {})
            if not s:
                continue
            print(
                f"    {mode:<18} {s.get('n', 0):>3} "
                f"{s.get('total_ms', 0):>10.1f} {s.get('per_req_ms', 0):>11.1f}"
            )
        e = (data.get("eagle3") or {}).get("scenarios", {}).get(b)
        do = (data.get("dflash_original") or {}).get("scenarios", {}).get(b)
        dn = (data.get("dflash_optimized") or {}).get("scenarios", {}).get(b)
        if e and do:
            print(f"    Δ dflash_original  vs eagle3: total "
                  f"{pct_delta(do['total_ms'], e['total_ms']):+.1f}%")
        if e and dn:
            print(f"    Δ dflash_optimized vs eagle3: total "
                  f"{pct_delta(dn['total_ms'], e['total_ms']):+.1f}%")
        if do and dn:
            print(f"    Δ dflash_optimized vs dflash_original: total "
                  f"{pct_delta(dn['total_ms'], do['total_ms']):+.1f}%")


if __name__ == "__main__":
    main()
