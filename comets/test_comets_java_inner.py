"""COMETS Java test with AGORA-enriched medium."""
import sys, os, pathlib
sys.path.insert(0, "/home/nishioka/IKM_Hiwi")

import cometspy as c
from nife.comets.oral_biofilm import OralBiofilmComets, AGORA_TRACE_METS

m = OralBiofilmComets(
    # comets_home auto-detected: ~/comets_linux
    agora_dir="/home/nishioka/IKM_Hiwi/nife/comets/agora_gems",
)

for cond in ("healthy", "diseased"):
    print(f"\n--- {cond} ---", flush=True)
    layout = m.build_layout(cond)
    params = m.build_params(max_cycles=100, write_biomass_log=True, biomass_log_rate=10)
    out = f"/home/nishioka/IKM_Hiwi/nife/comets/comets_runs/java_test_{cond}"
    pathlib.Path(out).mkdir(parents=True, exist_ok=True)
    # relative_dir must be relative to cwd (not absolute) to avoid double-prefix
    rel_out = os.path.relpath(out) + "/"

    exp = c.comets(layout, params, relative_dir=rel_out)
    print("CLASSPATH:", exp.JAVA_CLASSPATH[:200], flush=True)
    try:
        exp.run(delete_files=False)
        bm = exp.total_biomass
        print("Columns:", bm.columns.tolist()[:8])
        sp_cols = [col for col in bm.columns if col not in ("Cycle", "cycle", "x", "y", "z")]
        if sp_cols:
            t0 = bm.iloc[0][sp_cols].sum()
            tf = bm.iloc[-1][sp_cols].sum()
            print(f"  Biomass: {t0:.3e} -> {tf:.3e}  (ratio {tf/max(t0,1e-20):.2f})")
        else:
            print("  No species columns found")
            print(bm.head(3))
    except Exception as e:
        print(f"  ERROR: {e}")
        print("=== run_output (Java trace) ===")
        print(getattr(exp, "run_output", "(no run_output attr)"))
        break  # only try healthy; diseased will have same issue

print("\n[DONE]")
