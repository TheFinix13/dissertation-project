#!/usr/bin/env python3
"""Build demo/index.html — the phone demo behind the QR code on the
extended deck's closing slide.

What it shows: the dissertation's transfer test as an animation. The
cumulative-wealth lines of slide 10 draw themselves month by month,
so the viewer watches buy-and-hold plunge through October 2008 while
the Deep Q agent's line stays flat (it held zero shares).

Assets: SPY first (the evaluated result). Other assets are added to
ASSETS once their numbers have been checked offline — never before
(13 Sep rule: nothing goes on the phone that could embarrass live).

Regenerate:  ./venv/bin/python scripts/make_demo_page.py
Then push demo/index.html to the public pages repo (see notes below).
"""

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "experiments" / "final_v2" / "results" / "sim_results.json"
OUT = ROOT / "demo" / "index.html"

MONTH_NAMES = ["", "Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def month_label(mid: str) -> str:
    y, m = mid.split("-")
    return f"{MONTH_NAMES[int(m)]} {y}"


def spy_series():
    res = json.loads(RES.read_text())

    def algo_cum(name):
        seeds = res["results"][name]["per_seed"]
        monthly = np.mean(
            [[m["delta_w"] for m in s["real_per_month"]] for s in seeds],
            axis=0)
        return [round(float(v), 2) for v in np.cumsum(monthly)]

    bh_months = res["baselines_real"]["B1b_true_bah"]["per_month"]
    bh_cum = [round(float(v), 2)
              for v in np.cumsum([m["delta_w"] for m in bh_months])]
    ids = [m["id"] for m in bh_months]

    return {
        "label": "SPY — S&P 500",
        "sub": "the dissertation's transfer test: 180 real months "
               "(2006\u20132017 & 2023\u20132025) the agents never saw",
        "months": [month_label(i) for i in ids],
        "series": {
            "Buy-and-hold": bh_cum,
            "Deep Q-learning": algo_cum("dqn"),
            "REINFORCE": algo_cum("reinforce"),
            "PPO": algo_cum("ppo"),
        },
        "crash": [ids.index("2008-09"), ids.index("2009-05")],
        "crashText": "Oct 2008 \u2014 the Deep Q agent holds ZERO shares",
        "split": ids.index("2023-01"),
        "splitText": "2018\u20132022 excluded (simulator calibration)",
        "evaluated": True,
    }


OTHER = ROOT / "experiments" / "final_v2" / "results" / "other_assets.json"

ALGO_NAMES = {"dqn": "Deep Q-learning", "reinforce": "REINFORCE",
              "ppo": "PPO"}


def other_series(ticker: str, label: str):
    """Series for a non-SPY asset from run_other_assets.py output.
    Only whitelisted tickers reach ASSETS — after the numbers pass the
    offline gate (13 Sep rule)."""
    def _build():
        d = json.loads(OTHER.read_text())["assets"][ticker]
        ids = d["ids"]

        def cum(monthly):
            return [round(float(v), 2) for v in np.cumsum(monthly)]

        series = {"Buy-and-hold":
                  cum([m["delta_w"] for m in d["bah_per_month"]])}
        for algo, name in ALGO_NAMES.items():
            per_seed = d["algos"][algo]
            monthly = np.mean(
                [[m["delta_w"] for m in seed] for seed in per_seed], axis=0)
            series[name] = cum(monthly)

        out = {
            "label": label,
            "sub": f"the same policies, run unchanged on {label} — an "
                   "asset the agents have NEVER seen (unevaluated "
                   "engineering demo)",
            "months": [month_label(i) for i in ids],
            "series": series,
            "crashText": "Oct 2008 \u2014 the crisis months",
            "split": ids.index("2023-01") if "2023-01" in ids else None,
            "evaluated": False,
        }
        if "2008-09" in ids and "2009-05" in ids:
            out["crash"] = [ids.index("2008-09"), ids.index("2009-05")]
        return out
    return _build


# Whitelist: SPY always; other tickers added ONLY after their numbers
# pass the offline gate in run_other_assets.py's GATE SUMMARY.
# 13 Sep gate: AAPL and QQQ PASSED — Deep Q profitable on all six seeds
# on both (AAPL +100.31, QQQ +82.03 $/month), policy methods collapsed
# to fixed habits exactly as in the dissertation.
ASSETS = {
    "SPY": spy_series,
    "AAPL": other_series("AAPL", "AAPL \u2014 Apple"),
    "QQQ": other_series("QQQ", "QQQ \u2014 Nasdaq-100 ETF"),
}

# deck colours, lightened for a dark screen
COLORS = {"Buy-and-hold": "#c9c9c9", "Deep Q-learning": "#e07a1f",
          "REINFORCE": "#35b3af", "PPO": "#a07ce8"}


def main():
    data = {k: fn() for k, fn in ASSETS.items()}
    page = TEMPLATE.replace("__DATA__", json.dumps(data)) \
                   .replace("__COLORS__", json.dumps(COLORS))
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(page)
    print(f"Wrote {OUT} ({OUT.stat().st_size / 1024:.0f} KB, "
          f"{len(data)} asset(s))")


TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1">
<title>The transfer test — live</title>
<style>
  * { box-sizing: border-box; margin: 0; }
  body {
    background: #10151f; color: #e8e6e0; min-height: 100vh;
    font-family: -apple-system, "SF Pro Text", Segoe UI, Roboto, sans-serif;
    display: flex; flex-direction: column; align-items: center;
    padding: 18px 14px 30px;
  }
  .wrap { width: 100%; max-width: 520px; }
  h1 { font-size: 22px; font-weight: 800; letter-spacing: -0.01em; }
  .sub { font-size: 13.5px; color: #93a1b8; margin-top: 6px; line-height: 1.45; }
  nav { display: flex; gap: 8px; margin: 16px 0 4px; }
  nav button {
    background: #1d2635; color: #cfd8e6; border: 1px solid #33415a;
    border-radius: 999px; padding: 7px 16px; font-size: 14px; font-weight: 600;
  }
  nav button.on { background: #2c4a7c; border-color: #4a6ea9; color: #fff; }
  .legend { display: flex; flex-wrap: wrap; gap: 6px 14px; margin: 12px 0 4px;
            font-size: 12.5px; color: #b9c4d4; }
  .legend span::before {
    content: ""; display: inline-block; width: 14px; height: 4px;
    border-radius: 2px; margin-right: 6px; vertical-align: 3px;
    background: var(--c);
  }
  .counter { display: flex; align-items: baseline; gap: 12px; margin-top: 12px;
             min-height: 30px; }
  #month { font-size: 24px; font-weight: 800; font-variant-numeric: tabular-nums; }
  #crashNote { font-size: 13px; font-weight: 700; color: #e05a4e; }
  canvas { width: 100%; height: 46vh; max-height: 430px; display: block;
           margin-top: 6px; }
  .controls { display: flex; gap: 10px; margin-top: 14px; }
  .controls button {
    background: #1d2635; color: #cfd8e6; border: 1px solid #33415a;
    border-radius: 10px; padding: 10px 22px; font-size: 15px; font-weight: 700;
  }
  .controls button#play { background: #2f5e3f; border-color: #3f8055; color: #e3f6e9; }
  .note { font-size: 11.5px; color: #64748b; line-height: 1.5; margin-top: 18px; }
</style>
</head>
<body>
<div class="wrap">
  <h1>The transfer test — live</h1>
  <p class="sub" id="assetSub"></p>
  <nav id="tabs"></nav>
  <div class="legend" id="legend"></div>
  <div class="counter"><span id="month">&nbsp;</span><span id="crashNote"></span></div>
  <canvas id="chart"></canvas>
  <div class="controls">
    <button id="play">&#9654; play</button>
    <button id="speed">1&times;</button>
  </div>
  <p class="note">Agents were trained only on simulated monthly episodes
  (calibrated on 2018&ndash;2022), then run unchanged on the months shown.
  Lines are cumulative wealth change in dollars from $10,000 starting
  capital, 0.05% fee per trade, mean of six training seeds.
  Exploratory companion to the MSc dissertation of Fiyinfoluwa Akano
  (University of Surrey, 2026) &mdash; the SPY panel is the evaluated
  result; any other assets are unevaluated engineering demos.</p>
</div>
<script>
const DATA = __DATA__;
const COLORS = __COLORS__;

let asset = Object.keys(DATA)[0];
let t = 0, playing = false, speed = 1, raf = null, last = null;
const MONTHS_PER_SEC = 6;

const cv = document.getElementById("chart");
const ctx = cv.getContext("2d");

function fmt$(v) {
  const a = Math.abs(v);
  const s = a >= 1000 ? (a/1000).toFixed(1) + "k" : a.toFixed(0);
  return (v < 0 ? "\u2212$" : "+$") + s;
}

function setupTabs() {
  const nav = document.getElementById("tabs");
  nav.innerHTML = "";
  for (const k of Object.keys(DATA)) {
    const b = document.createElement("button");
    b.textContent = k;
    b.className = k === asset ? "on" : "";
    b.onclick = () => { asset = k; reset(); setupTabs(); };
    nav.appendChild(b);
  }
  const d = DATA[asset];
  document.getElementById("assetSub").textContent = d.sub;
  const lg = document.getElementById("legend");
  lg.innerHTML = "";
  for (const name of Object.keys(d.series)) {
    const s = document.createElement("span");
    s.style.setProperty("--c", COLORS[name]);
    s.textContent = name;
    lg.appendChild(s);
  }
}

function sizeCanvas() {
  const dpr = window.devicePixelRatio || 1;
  const r = cv.getBoundingClientRect();
  cv.width = r.width * dpr; cv.height = r.height * dpr;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return r;
}

function draw() {
  const d = DATA[asset];
  const names = Object.keys(d.series);
  const N = d.months.length;
  const r = sizeCanvas();
  const P = {l: 46, r: 14, t: 12, b: 24};
  const W = r.width - P.l - P.r, H = r.height - P.t - P.b;

  let lo = 0, hi = 0;
  for (const n of names) for (const v of d.series[n]) {
    if (v < lo) lo = v; if (v > hi) hi = v;
  }
  const pad = (hi - lo) * 0.07; lo -= pad; hi += pad;
  const X = i => P.l + W * i / (N - 1);
  const Y = v => P.t + H * (1 - (v - lo) / (hi - lo));

  ctx.clearRect(0, 0, r.width, r.height);

  // horizontal gridlines
  ctx.font = "10px -apple-system, sans-serif";
  const step = (hi - lo) > 20000 ? 5000 : 2500;
  for (let g = Math.ceil(lo/step)*step; g <= hi; g += step) {
    ctx.strokeStyle = g === 0 ? "#3d4a60" : "#1c2534";
    ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(P.l, Y(g)); ctx.lineTo(P.l + W, Y(g)); ctx.stroke();
    ctx.fillStyle = "#5b6a82";
    ctx.fillText(fmt$(g), 4, Y(g) + 3);
  }

  const ti = Math.min(t, N - 1);
  const upto = Math.floor(ti);

  // crash band (appears once reached)
  if (d.crash && upto >= d.crash[0]) {
    const x0 = X(d.crash[0]), x1 = X(Math.min(ti, d.crash[1]));
    ctx.fillStyle = "rgba(192,57,43,0.16)";
    ctx.fillRect(x0, P.t, x1 - x0, H);
  }
  // calibration-gap separator
  if (d.split && upto >= d.split) {
    ctx.strokeStyle = "#4a5a75"; ctx.setLineDash([4, 4]);
    ctx.beginPath(); ctx.moveTo(X(d.split), P.t);
    ctx.lineTo(X(d.split), P.t + H); ctx.stroke();
    ctx.setLineDash([]);
  }

  // year ticks
  ctx.fillStyle = "#5b6a82";
  for (let i = 0; i < N; i += 24) {
    ctx.fillText(d.months[i].split(" ")[1], X(i) - 12, r.height - 8);
  }

  // series lines up to t (with fractional last segment)
  for (const n of names) {
    const s = d.series[n];
    ctx.strokeStyle = COLORS[n];
    ctx.lineWidth = n === "Deep Q-learning" || n === "Buy-and-hold" ? 2.6 : 1.7;
    ctx.beginPath();
    ctx.moveTo(X(0), Y(s[0]));
    for (let i = 1; i <= upto; i++) ctx.lineTo(X(i), Y(s[i]));
    if (upto < N - 1) {
      const f = ti - upto;
      ctx.lineTo(X(ti), Y(s[upto] + (s[upto+1] - s[upto]) * f));
    }
    ctx.stroke();
    // moving dot
    const vy = upto < N - 1 ? s[upto] + (s[upto+1]-s[upto])*(ti-upto) : s[N-1];
    ctx.fillStyle = COLORS[n];
    ctx.beginPath(); ctx.arc(X(ti), Y(vy), 3.4, 0, 7); ctx.fill();
    // endpoint labels once finished
    if (ti >= N - 1) {
      ctx.font = "bold 12px -apple-system, sans-serif";
      ctx.fillText(fmt$(s[N-1]), Math.min(X(N-1) - 44, r.width - 58), Y(s[N-1]) - 7);
      ctx.font = "10px -apple-system, sans-serif";
    }
  }

  // labels
  document.getElementById("month").textContent = d.months[upto];
  const inCrash = d.crash && upto >= d.crash[0] && upto <= d.crash[1] + 3;
  document.getElementById("crashNote").textContent = inCrash ? d.crashText : "";
}

function tick(ts) {
  if (!playing) return;
  if (last === null) last = ts;
  const dt = (ts - last) / 1000; last = ts;
  const N = DATA[asset].months.length;
  t = Math.min(t + dt * MONTHS_PER_SEC * speed, N - 1);
  draw();
  if (t >= N - 1) { playing = false; setPlayLabel(); return; }
  raf = requestAnimationFrame(tick);
}

function setPlayLabel() {
  const N = DATA[asset].months.length;
  document.getElementById("play").innerHTML =
    playing ? "&#10074;&#10074; pause" : (t >= N - 1 ? "&#8635; replay" : "&#9654; play");
}

document.getElementById("play").onclick = () => {
  const N = DATA[asset].months.length;
  if (!playing && t >= N - 1) t = 0;
  playing = !playing; last = null; setPlayLabel();
  if (playing) raf = requestAnimationFrame(tick);
};
document.getElementById("speed").onclick = e => {
  speed = speed === 1 ? 2 : 1;
  e.target.textContent = speed + "\u00d7";
};
function reset() { playing = false; t = 0; last = null; setPlayLabel(); draw(); }
window.addEventListener("resize", draw);

setupTabs();
reset();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
