#!/usr/bin/env python3
"""Autonomous coverage test for neuralset tool (PR #10)."""
import sys
import json
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# Import directly without triggering devduck full init
import importlib.util
spec = importlib.util.spec_from_file_location("neuralset", "devduck/tools/neuralset.py")
ns = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ns)

neuralset = ns.neuralset

# Coverage matrix: (test_id, action, kwargs, expected_status_or_substring)
TESTS = [
    # ---- help ----
    ("T01_help", {"action": "help"}, "success"),

    # ---- list ----
    ("T02_list_all", {"action": "list"}, "success"),
    ("T03_list_eeg", {"action": "list", "modality": "eeg"}, "success"),
    ("T04_list_meg", {"action": "list", "modality": "meg"}, "success"),
    ("T05_list_fmri", {"action": "list", "modality": "fmri"}, "success"),
    ("T06_list_ieeg", {"action": "list", "modality": "ieeg"}, "success"),
    ("T07_list_fnirs", {"action": "list", "modality": "fnirs"}, "success"),
    ("T08_list_spikes", {"action": "list", "modality": "spikes"}, "success"),
    ("T09_list_bad_modality", {"action": "list", "modality": "xray"}, "error"),

    # ---- info ----
    ("T10_info_known", {"action": "info", "dataset": "things-eeg2"}, "success"),
    ("T11_info_missing", {"action": "info"}, "error"),
    ("T12_info_unknown", {"action": "info", "dataset": "zzz-nonexistent"}, "success"),  # returns hint, not error

    # ---- setup (dry-run) ----
    ("T13_setup_dry", {"action": "setup", "dry_run": True}, "success"),
    ("T14_setup_force_dry", {"action": "setup", "force": True, "dry_run": True}, "success"),

    # ---- download ----
    ("T15_download_dry", {"action": "download", "dataset": "things-eeg2", "dry_run": True}, "success"),
    ("T16_download_subjects_dry", {"action": "download", "dataset": "things-eeg2", "subjects": [1, 2], "dry_run": True}, "success"),
    ("T17_download_missing", {"action": "download"}, "error"),

    # ---- preprocess ----
    ("T18_preprocess_dry", {"action": "preprocess", "dataset": "things-eeg2", "dry_run": True}, "success"),
    ("T19_preprocess_config_dry", {"action": "preprocess", "dataset": "things-eeg2", "config": "default", "dry_run": True}, "success"),
    ("T20_preprocess_missing", {"action": "preprocess"}, "error"),

    # ---- features ----
    ("T21_features_dry", {"action": "features", "dataset": "things-eeg2", "backbone": "clip", "dry_run": True}, "success"),
    ("T22_features_layer_dry", {"action": "features", "dataset": "things-eeg2", "backbone": "clip", "layer": "visual.transformer.resblocks.11", "dry_run": True}, "success"),
    ("T23_features_missing_dataset", {"action": "features", "backbone": "clip"}, "error"),
    ("T24_features_missing_backbone", {"action": "features", "dataset": "things-eeg2"}, "error"),

    # ---- align ----
    ("T25_align_dry", {"action": "align", "dataset": "things-eeg2", "backbone": "clip", "dry_run": True}, "success"),
    ("T26_align_missing", {"action": "align"}, "error"),

    # ---- encode ----
    ("T27_encode_dry", {"action": "encode", "dataset": "things-eeg2", "backbone": "clip", "dry_run": True}, "success"),
    ("T28_encode_model_dry", {"action": "encode", "dataset": "things-eeg2", "backbone": "clip", "model": "mlp", "dry_run": True}, "success"),
    ("T29_encode_missing", {"action": "encode"}, "error"),

    # ---- decode ----
    ("T30_decode_dry", {"action": "decode", "dataset": "things-eeg2", "dry_run": True}, "success"),
    ("T31_decode_target_dry", {"action": "decode", "dataset": "things-eeg2", "target": "category", "dry_run": True}, "success"),
    ("T32_decode_embedding_dry", {"action": "decode", "dataset": "things-eeg2", "target": "embedding", "dry_run": True}, "success"),
    ("T33_decode_missing", {"action": "decode"}, "error"),

    # ---- evaluate ----
    ("T34_evaluate_dry", {"action": "evaluate", "dataset": "things-eeg2", "dry_run": True}, "success"),
    ("T35_evaluate_r2_dry", {"action": "evaluate", "dataset": "things-eeg2", "metric": "r2", "dry_run": True}, "success"),
    ("T36_evaluate_acc_dry", {"action": "evaluate", "dataset": "things-eeg2", "metric": "accuracy", "dry_run": True}, "success"),
    ("T37_evaluate_missing", {"action": "evaluate"}, "error"),

    # ---- leaderboard ----
    ("T38_leaderboard_view", {"action": "leaderboard", "sub_action": "view"}, "success"),
    ("T39_leaderboard_submit_missing", {"action": "leaderboard", "sub_action": "submit"}, "error"),

    # ---- cache ----
    ("T40_cache_stats", {"action": "cache", "sub_action": "stats"}, "success"),
    ("T41_cache_list", {"action": "cache", "sub_action": "list"}, "success"),

    # ---- bad action ----
    ("T42_bad_action", {"action": "invalid_xyz"}, "error"),
]

results = []
passed = 0
failed = 0

for test_id, kwargs, expected in TESTS:
    try:
        res = neuralset(**kwargs)
        status = res.get("status", "?")
        text = ""
        if res.get("content") and isinstance(res["content"], list):
            text = res["content"][0].get("text", "")[:200] if res["content"] else ""
        ok = (status == expected)
        results.append({
            "id": test_id,
            "kwargs": kwargs,
            "expected": expected,
            "got_status": status,
            "preview": text[:120],
            "pass": ok,
        })
        if ok:
            passed += 1
        else:
            failed += 1
    except Exception as e:
        failed += 1
        results.append({
            "id": test_id,
            "kwargs": kwargs,
            "expected": expected,
            "got_status": "EXCEPTION",
            "preview": f"{type(e).__name__}: {e}",
            "pass": False,
            "trace": traceback.format_exc()[-500:],
        })

print(f"\n{'='*70}")
print(f"COVERAGE MATRIX - PR #10 (neuralset)")
print(f"{'='*70}")
print(f"{'ID':<30} {'EXPECT':<10} {'GOT':<12} {'PASS'}")
print(f"{'-'*70}")
for r in results:
    mark = "✅" if r["pass"] else "❌"
    print(f"{r['id']:<30} {r['expected']:<10} {r['got_status']:<12} {mark}")

print(f"\n{'='*70}")
print(f"TOTAL: {passed}/{len(TESTS)} passed, {failed} failed")
print(f"{'='*70}")

# Save JSON
with open("/tmp/neuralset_coverage.json", "w") as f:
    json.dump({
        "total": len(TESTS),
        "passed": passed,
        "failed": failed,
        "pass_rate": round(passed/len(TESTS)*100, 1),
        "results": results,
    }, f, indent=2)

print(f"\nDetailed results: /tmp/neuralset_coverage.json")

# Print failures in detail
if failed > 0:
    print(f"\n{'='*70}")
    print("FAILURES DETAIL:")
    print(f"{'='*70}")
    for r in results:
        if not r["pass"]:
            print(f"\n❌ {r['id']}: {r['kwargs']}")
            print(f"   Expected: {r['expected']} | Got: {r['got_status']}")
            print(f"   Preview: {r['preview']}")
            if "trace" in r:
                print(f"   Trace: {r['trace']}")

sys.exit(0 if failed == 0 else 1)
