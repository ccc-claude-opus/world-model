"""實驗 H：用 YOLO11n-pose + ByteTrack 追蹤雙人舞影片，看它能不能當 Duolando 的輸入端。

量三件事：
  1. 兩位舞者的 ID 會不會互換（沒有標註，用「關節位置連續性」推估可疑互換）
  2. 閉式舞姿時漏抓多少：每格偵測到幾個人、關節信心 < 0.5 的比例
  3. 速度：每格推論時間（CPU / MPS）

影片（Pexels 免費授權，data/ 不進 git）：
  data/pose/tango.mp4  https://www.pexels.com/video/8281169/
  data/pose/waltz.mp4  https://www.pexels.com/video/30755431/
模型：models/yolo11n-pose.pt（Ultralytics，AGPL-3.0）

用法：uv run python exp_h/pose_track.py data/pose/tango.mp4 data/pose/waltz.mp4 --device mps
"""
import argparse, json, os, time

import cv2
import numpy as np
from ultralytics import YOLO

L_WRIST, R_WRIST, L_SH, R_SH, L_HIP, R_HIP = 9, 10, 5, 6, 11, 12


def torso_gray(img, kp):
    """肩到髖的方框平均亮度：用來分辨真實身分（兩段影片都是男方淺色西裝、女方深色洋裝）。"""
    pts = kp[[L_SH, R_SH, L_HIP, R_HIP]]
    if (pts == 0).any():
        return np.nan
    x0, y0 = pts.min(0).astype(int)
    x1, y1 = pts.max(0).astype(int)
    patch = img[max(y0, 0):y1, max(x0, 0):x1]
    return float(cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY).mean()) if patch.size else np.nan


def run(path, model, device, out_dir, tracker, width=540):
    name = os.path.splitext(os.path.basename(path))[0]
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    size = (width, int(H * width / W))
    vw = cv2.VideoWriter(f"{out_dir}/{name}.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    frames, times = [], []
    while True:
        ok, img = cap.read()
        if not ok:
            break
        img = cv2.resize(img, size)                          # 先縮小：4K 直接丟進去只會更慢，640 輸入一樣
        t0 = time.perf_counter()
        r = model.track(img, persist=True, tracker=tracker, device=device, verbose=False, conf=0.3)[0]
        times.append(time.perf_counter() - t0)
        ids = r.boxes.id.int().tolist() if r.boxes.id is not None else [-1] * len(r.boxes)
        kps = r.keypoints.xy.cpu().numpy()
        frames.append({"ids": ids, "area": (r.boxes.xywh[:, 2] * r.boxes.xywh[:, 3]).tolist(),
                       "gray": [torso_gray(img, k) for k in kps],
                       "kp": r.keypoints.xy.cpu().numpy(), "conf": r.keypoints.conf.cpu().numpy() if r.keypoints.conf is not None else None})
        vw.write(r.plot(boxes=True, labels=True, conf=False))
    vw.release()
    return name, fps, size, frames, np.array(times)


def analyze(frames, fps):
    # 兩位舞者 = 整段影片中出現最多格的兩個 track ID
    count = {}
    for f in frames:
        for i in f["ids"]:
            count[i] = count.get(i, 0) + 1
    main = [i for i, _ in sorted(count.items(), key=lambda x: -x[1])[:2]]
    n_people = np.array([len(f["ids"]) for f in frames])
    both = np.array([all(m in f["ids"] for m in main) for f in frames])
    # 用服裝亮度給兩位舞者「真實身分」：每格最大的兩人中，較亮的是男方（L）、較暗的是女方（F）
    owner, id_changes = {}, {"L": [], "F": []}
    last = {"L": None, "F": None}
    for k, f in enumerate(frames):
        big = sorted(range(len(f["ids"])), key=lambda j: -f["area"][j])[:2]
        if len(big) < 2 or any(np.isnan(f["gray"][j]) for j in big) or abs(f["gray"][big[0]] - f["gray"][big[1]]) < 15:
            continue                                         # 只剩一人、關節缺、或亮度分不開就跳過
        light, dark = sorted(big, key=lambda j: -f["gray"][j])
        for who, j in (("L", light), ("F", dark)):
            tid = f["ids"][j]
            if last[who] is not None and tid != last[who]:
                id_changes[who].append(round(k / fps, 2))
            last[who] = tid
            owner.setdefault(tid, set()).add(who)
    mixed = sorted(t for t, w in owner.items() if len(w) == 2)   # 同一個 ID 先後給了男方和女方 = 互換
    low_conf, hands, swaps, prev = [], [], [], None
    for k, f in enumerate(frames):
        if f["conf"] is not None:
            for m in main:
                if m in f["ids"]:
                    low_conf.append((f["conf"][f["ids"].index(m)] < 0.5).mean())
        if not all(m in f["ids"] for m in main):
            prev = None
            continue
        P = [f["kp"][f["ids"].index(m)] for m in main]
        C = [f["conf"][f["ids"].index(m)] > 0.5 for m in main]
        # 握手：兩人最近一對手腕距離 / 平均軀幹長
        torso = np.mean([np.linalg.norm((p[L_SH] + p[R_SH]) / 2 - (p[L_HIP] + p[R_HIP]) / 2) for p in P])
        pairs = [np.linalg.norm(P[0][a] - P[1][b]) for a in (L_WRIST, R_WRIST) for b in (L_WRIST, R_WRIST) if C[0][a] and C[1][b]]
        if pairs and torso > 1:
            hands.append(min(pairs) / torso)
        # 可疑互換：把這格的兩人對調後，和上一格的關節距離反而小很多
        if prev is not None:
            def d(a, b):
                v = np.linalg.norm(a - b, axis=1)
                return np.median(v)
            keep = d(P[0], prev[0]) + d(P[1], prev[1])
            swap = d(P[0], prev[1]) + d(P[1], prev[0])
            if swap < 0.5 * keep:
                swaps.append(round(k / fps, 2))
        prev = P
    return {"frames": len(frames), "track_ids_used": len(count), "main_ids": main,
            "frames_with_both_pct": round(100 * both.mean(), 1),
            "people_per_frame": {str(n): int((n_people == n).sum()) for n in sorted(set(n_people))},
            "low_conf_keypoints_pct": round(100 * float(np.mean(low_conf)), 1),
            "hand_dist_over_torso_median": round(float(np.median(hands)), 2) if hands else None,
            "suspected_swaps_at_s": swaps,
            "id_changes_leader_at_s": id_changes["L"], "id_changes_follower_at_s": id_changes["F"],
            "ids_given_to_both_dancers": mixed}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", nargs="+")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--tracker", default="bytetrack.yaml", help="bytetrack.yaml 或 exp_h/botsort_reid.yaml")
    ap.add_argument("--out", default="out/exp_h")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    report = {}
    for v in args.videos:
        model = YOLO("models/yolo11n-pose.pt")             # 每段影片重新載入，追蹤器狀態不會延續
        name, fps, size, frames, t = run(v, model, args.device, args.out, args.tracker)
        rep = analyze(frames, fps)
        rep.update({"input_size": size, "video_fps": round(fps, 2), "device": args.device, "tracker": args.tracker,
                    "ms_per_frame_median": round(1000 * float(np.median(t[5:])), 1)})   # 前幾格是暖機
        report[name] = rep
        print(name, json.dumps(rep), flush=True)
    json.dump(report, open(f"{args.out}/metrics_{args.device}.json", "w"), indent=2)


if __name__ == "__main__":
    main()
