"""實驗 H 延伸：整段影片跑完後，離線把破碎的追蹤 ID 合併成「男方 L、女方 F」兩條軌跡。

輸入 pose_track.py 存的 out/exp_h/<name>_dets.npz（每格的人框與 17 個關節）。
做法：
  1. 外觀：沿著軀幹與腿的骨段取像素，算 HSV 色彩直方圖；全片做 2-means 分成兩群，衣服較亮的那群叫 L。
     （兩段測試影片都是男方淺色、女方深色；換影片時這個命名規則要改。）
  2. 每格最多取兩個主要偵測（濾掉面積太小的背景人物），列出所有「誰是誰」的假設。
  3. Viterbi：成本 = 外觀距離 + 關節位移（和這個身分上次出現的位置比，除以軀幹長），
     找整段影片總成本最小的指派。被擋住而沒有框的人就留空（NaN）。
評估：exp_h/gt_labels.json 是人工看影格標的真實身分（探戈每 0.5 秒、華爾滋每 1 秒）。

用法：uv run python exp_h/merge_ids.py tango waltz
"""
import argparse, json, os

import cv2
import numpy as np

SEGS = [(5, 11), (6, 12), (5, 12), (6, 11), (11, 13), (12, 14), (13, 15), (14, 16)]   # 軀幹交叉線與腿
SK = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (0, 5), (0, 6)]
COLOR = {"L": (230, 120, 30), "F": (40, 40, 220)}                                     # BGR：男藍、女紅


def appearance(hsv, kp, conf):
    """沿骨段取 5×5 像素塊，算 H（以飽和度加權）12 格 + S 4 格 + V 8 格的直方圖。"""
    pix = []
    for a, b in SEGS:
        if conf[a] > 0.3 and conf[b] > 0.3:
            for s in np.linspace(0.15, 0.85, 8):
                x, y = (kp[a] * (1 - s) + kp[b] * s).astype(int)
                pix.append(hsv[max(y - 2, 0):y + 3, max(x - 2, 0):x + 3].reshape(-1, 3))
    if not pix:
        return None
    p = np.concatenate(pix).astype(float)
    h = np.histogram(p[:, 0], 12, (0, 180), weights=p[:, 1] / 255)[0]
    s = np.histogram(p[:, 1], 4, (0, 256))[0]
    v = np.histogram(p[:, 2], 8, (0, 256))[0]
    return np.concatenate([h / max(h.sum(), 1e-6) * 0.5, s / s.sum(), v / v.sum()])


def load(name, dets_dir, video_dir):
    d = dict(np.load(f"{dets_dir}/{name}_dets.npz"))
    size = tuple(int(v) for v in d["size"])
    cap, k, feats = cv2.VideoCapture(f"{video_dir}/{name}.mp4"), 0, [None] * len(d["frame"])
    while True:
        ok, img = cap.read()
        if not ok:
            break
        hsv = cv2.cvtColor(cv2.resize(img, size), cv2.COLOR_BGR2HSV)
        for r in np.where(d["frame"] == k)[0]:
            feats[r] = appearance(hsv, d["kp"][r], d["kp_conf"][r])
        k += 1
    d["feat"], d["n_frames"] = feats, k
    return d


def candidates(d):
    """每格最多兩個主要偵測：面積至少是「每格最大人框」第 90 百分位的 30%，再取信心最高的兩個。"""
    area = (d["box"][:, 2] - d["box"][:, 0]) * (d["box"][:, 3] - d["box"][:, 1])
    big = np.percentile([area[d["frame"] == k].max() for k in np.unique(d["frame"])], 90)
    out = []
    for k in range(d["n_frames"]):
        rows = [r for r in np.where(d["frame"] == k)[0] if area[r] > 0.3 * big and d["feat"][r] is not None]
        out.append(sorted(rows, key=lambda r: -d["score"][r])[:2])
    return out


def cluster(d, cand, iters=20):
    X = np.array([d["feat"][r] for rows in cand for r in rows])
    bright = X[:, -8:] @ np.arange(8)                                                 # V 直方圖的平均格位
    mu = np.stack([X[bright >= np.median(bright)].mean(0), X[bright < np.median(bright)].mean(0)])
    for _ in range(iters):
        lab = np.abs(X[:, None] - mu[None]).sum(2).argmin(1)
        mu = np.stack([X[lab == c].mean(0) for c in (0, 1)])
    return {"L": mu[0], "F": mu[1]} if mu[0][-8:] @ np.arange(8) > mu[1][-8:] @ np.arange(8) else {"L": mu[1], "F": mu[0]}


def motion(kp, conf, last, torso):
    if last is None:
        return 0.0
    ok = (conf > 0.5) & (last[1] > 0.5)
    if ok.sum() < 3:
        return 0.0
    return min(float(np.median(np.linalg.norm(kp[ok] - last[0][ok], axis=1))) / torso, 2.0)


def viterbi(d, cand, mu, w_app=1.0, w_mot=1.0):
    kp, conf = d["kp"], d["kp_conf"]
    torso = np.median([np.linalg.norm((kp[r][5] + kp[r][6]) / 2 - (kp[r][11] + kp[r][12]) / 2)
                       for rows in cand for r in rows if (conf[r][[5, 6, 11, 12]] > 0.5).all()])
    # 每個狀態：(累積成本, 指派 {身分: 偵測列}, 上一狀態索引, 兩人最後位置)
    prev = [(0.0, {}, -1, {"L": None, "F": None})]
    hist = []
    for rows in cand:
        hyps = [{}] if not rows else ([{"L": rows[0]}, {"F": rows[0]}] if len(rows) == 1 else
                                      [{"L": rows[0], "F": rows[1]}, {"L": rows[1], "F": rows[0]}])
        cur = []
        for h in hyps:
            app = sum(np.abs(d["feat"][r] - mu[who]).sum() for who, r in h.items())
            best = None
            for i, (c, _, _, last) in enumerate(prev):
                cost = c + w_app * app + w_mot * sum(motion(kp[r], conf[r], last[who], torso) for who, r in h.items())
                if best is None or cost < best[0]:
                    best = (cost, i, last)
            new_last = dict(best[2])
            for who, r in h.items():
                new_last[who] = (kp[r], conf[r])
            cur.append((best[0], h, best[1], new_last))
        hist.append(cur)
        prev = cur
    # 回溯
    i, path = int(np.argmin([s[0] for s in prev])), []
    for cur in reversed(hist):
        path.append(cur[i][1])
        i = cur[i][2]
    return path[::-1]


def score(d, assign_of, gt):
    """assign_of(frame, box_cx) -> 'L' / 'F' / None；比對人工標註。"""
    hit = 0
    for g in gt:
        rows = np.where(d["frame"] == g["frame"])[0]
        cx = (d["box"][rows, 0] + d["box"][rows, 2]) / 2
        r = rows[np.abs(cx - g["box_cx"]).argmin()]
        hit += assign_of(g["frame"], r) == g["who"]
    return round(100 * hit / len(gt), 1)


def render(name, d, path, video_dir, out):
    size = tuple(int(v) for v in d["size"])
    cap = cv2.VideoCapture(f"{video_dir}/{name}.mp4")
    vw = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), float(d["fps"]), size)
    for h in path:
        ok, img = cap.read()
        img = cv2.resize(img, size)
        for who, r in h.items():
            p, c = d["kp"][r].astype(int), d["kp_conf"][r] > 0.5
            for a, b in SK:
                if c[a] and c[b]:
                    cv2.line(img, tuple(p[a]), tuple(p[b]), COLOR[who], 3)
            x0, y0 = d["box"][r][:2].astype(int)
            cv2.putText(img, "leader" if who == "L" else "follower", (x0, max(y0 - 6, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR[who], 2)
        vw.write(img)
    vw.release()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="+")
    ap.add_argument("--dets", default="out/exp_h")
    ap.add_argument("--videos", default="data/pose")
    args = ap.parse_args()
    gt_all = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "gt_labels.json")))
    report = {}
    for name in args.names:
        d = load(name, args.dets, args.videos)
        gt, cand = gt_all[name], candidates(d)
        mu = cluster(d, cand)
        # 基準：原始追蹤 ID。就算事後知道「每個 ID 多數時候是誰」（用標註作弊），正確率上限是多少
        tid = d["track_id"]
        vote = {}
        for g in gt:
            rows = np.where(d["frame"] == g["frame"])[0]
            r = rows[np.abs((d["box"][rows, 0] + d["box"][rows, 2]) / 2 - g["box_cx"]).argmin()]
            vote.setdefault(tid[r], []).append(g["who"])
        oracle = {t: max(set(v), key=v.count) for t, v in vote.items()}
        rep = {"raw_ids_oracle_mapping_pct": score(d, lambda k, r: oracle.get(tid[r]), gt)}
        for label, wa, wm in (("appearance_only", 1.0, 0.0), ("motion_only", 1e-3, 1.0), ("appearance_plus_motion", 1.0, 1.0)):
            path = viterbi(d, cand, mu, wa, wm)
            who_of = {r: who for h in path for who, r in h.items()}
            rep[f"{label}_pct"] = score(d, lambda k, r: who_of.get(r), gt)
            if label == "appearance_plus_motion":
                rep["frames_with_L_pct"] = round(100 * np.mean(["L" in h for h in path]), 1)
                rep["frames_with_F_pct"] = round(100 * np.mean(["F" in h for h in path]), 1)
                render(name, d, path, args.videos, f"{args.dets}/{name}_merged.mp4")
        rep["labeled_people"] = len(gt)
        report[name] = rep
        print(name, json.dumps(rep), flush=True)
    json.dump(report, open(f"{args.dets}/merge_metrics.json", "w"), indent=2)


if __name__ == "__main__":
    main()
