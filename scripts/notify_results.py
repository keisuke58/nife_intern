#!/usr/bin/env python3
"""
notify_results.py — 新しい fit JSON を検知して paper_data.py と比較。
HPC ジョブ完了通知にも使う。Slack DM は Claude セッション経由。

Usage:
    python scripts/notify_results.py              # 新規 JSON チェック
    python scripts/notify_results.py --hpc        # PBS ジョブ状況チェック
    python scripts/notify_results.py --watch      # 30秒ポーリング
"""
import sys, os, json, time, subprocess
from pathlib import Path

sys.path.insert(0, '/home/nishioka/IKM_Hiwi/nife')
NIFE    = Path('/home/nishioka/IKM_Hiwi/nife')
NOTIFY  = NIFE / '.new_results_notify'
SEEN    = NIFE / '.seen_results'

def load_seen():
    if not SEEN.exists(): return set()
    return set(SEEN.read_text().splitlines())

def save_seen(s):
    SEEN.write_text('\n'.join(sorted(s)))

def compare_with_baseline(json_path: Path) -> str:
    """新しい fit JSON を paper baseline と比較"""
    try:
        from paper_data import paper_5sp_theta
        import numpy as np
        d = json.loads(json_path.read_text())
        lines = [f"新規結果: {json_path.name}"]
        A_IDX = {(0,0):0,(0,1):1,(1,1):2,(2,2):5,(2,3):6,(3,3):7,
                 (0,2):10,(0,3):11,(1,2):12,(1,3):13,
                 (4,4):14,(0,4):16,(1,4):17,(2,4):18,(3,4):19}
        for cond in ['CH','DH']:
            if cond in d and 'theta' in d[cond]:
                th_new  = np.array(d[cond]['theta'])
                th_base = paper_5sp_theta(cond)
                diff    = np.abs(th_new - th_base).mean()
                lines.append(f"  {cond}: baseline との平均差 = {diff:.4f}")
        return '\n'.join(lines)
    except Exception as e:
        return f"比較失敗: {e}"

def check_hpc() -> str:
    """PBS ジョブ状況を SSH で確認"""
    try:
        r = subprocess.run(
            ['ssh', '-o', 'ConnectTimeout=5', 'fifawc', 'qstat -u $USER 2>/dev/null | tail -20'],
            capture_output=True, text=True, timeout=10)
        if r.returncode == 0 and r.stdout.strip():
            return f"PBS jobs:\n{r.stdout.strip()}"
        return "PBS: ジョブなし or 接続失敗"
    except Exception as e:
        return f"PBS 確認失敗: {e}"

def scan_new_jsons(days: int = 1) -> list:
    """直近N日の新規 results JSON"""
    import os, time
    cutoff = time.time() - days * 86400
    seen   = load_seen()
    new    = []
    for f in NIFE.glob('results/**/*.json'):
        if f.stat().st_mtime > cutoff and str(f) not in seen:
            new.append(f)
    return new

def main():
    hpc   = '--hpc'   in sys.argv
    watch = '--watch' in sys.argv

    if hpc:
        print(check_hpc())
        return

    def run_once():
        new = scan_new_jsons(days=1)
        if not new:
            print("新規 result JSON なし")
            return []
        seen = load_seen()
        msgs = []
        for f in new:
            msg = compare_with_baseline(f)
            print(msg)
            msgs.append(msg)
            seen.add(str(f))
        save_seen(seen)
        return msgs

    if watch:
        print("ポーリング開始 (30秒間隔, Ctrl+C で停止)")
        while True:
            run_once()
            time.sleep(30)
    else:
        run_once()

if __name__ == '__main__':
    main()
