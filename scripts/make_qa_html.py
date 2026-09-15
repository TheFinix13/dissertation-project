#!/usr/bin/env python3
"""Build notes/viva/qa.html — Nguyen Q&A board with voice tracing.

Like the transcript teleprompter, but for the Q&A half of the viva:
every question Nguyen has asked (notes/viva/nguyen_questions_compiled.md)
as a card with its fluent answer. The page listens with the Web Speech
API and scrolls to the question that best matches what it hears.

Design decisions (13 Sep):
- Matching is keyword-scored, not line-tracked: Nguyen will paraphrase,
  so each card carries a weighted bag of keywords (rare terms weigh
  more), and the best-scoring card above a confidence threshold wins.
- Auto-trace can be toggled off; a search box and arrow keys are the
  manual fallback. Wrong auto-jumps must never trap Fiyin: hitting
  Esc or typing always overrides.
- Works best when Fiyin REPEATS the question back before answering
  ("So the question is why lambda is 0.25...") — good viva technique
  anyway, and it puts the keywords on the mic even with headphones on.

Regenerate:  ./venv/bin/python scripts/make_qa_html.py
Serves at:   http://localhost:8749/qa.html
"""

import html
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "notes" / "viva" / "nguyen_questions_compiled.md"
OUT = ROOT / "notes" / "viva" / "qa.html"
SERVE_DIR = Path.home() / ".viva_prompter"

EQ_CHARS = set("𝔼𝓡∇θφγΣπτÂ≤←δσ")

# Display equations are rendered as typeset maths (matplotlib mathtext →
# PNG data-URI), exactly like the deck. Keyed by the space-stripped
# prefix of the markdown line; order matters (most specific first).
EQ_TEX = [
    ("L^{CLIP}(θ)=",
     r"$L^{CLIP}(\theta)=\mathbb{E}_t\left[\,\min\!\left(r_t(\theta)\,"
     r"\hat{A}_t,\ \mathrm{clip}(r_t(\theta),\,1-\epsilon,\,1+\epsilon)\,"
     r"\hat{A}_t\right)\right]$"),
    ("∇_θJ(θ)=",
     r"$\nabla_\theta J(\theta)=\mathbb{E}_{\tau\sim\pi_\theta}\left[\,"
     r"\sum_{t}\nabla_\theta\log\pi_\theta(a_t\mid s_t)\cdot G_t\right]$"),
    ("J(θ)=",
     r"$J(\theta)=\mathbb{E}_{\tau\sim\pi_\theta}\left[\,\sum_{t=0}^{T-1}"
     r"\gamma^{t}\,r_t\right]$"),
    ("L(θ)=",
     r"$L(\theta)=\mathbb{E}_{(s,a,r,s')\sim\mathcal{D}}\left[\left(r+"
     r"\gamma\,\max_{a'}Q_{\theta^-}(s',a')-Q_\theta(s,a)\right)^{2}"
     r"\right]$"),
    ("𝓡(f)=",
     r"$\mathcal{R}(f)=\mathbb{E}_{(x,y)\sim\mathcal{D}}\left[\,\ell(f(x),"
     r"y)\,\right]\qquad\hat{\mathcal{R}}_n(f)=\frac{1}{n}\sum_{i}"
     r"\ell(f(x_i),y_i)$"),
    # --- signal definitions (ATR / MA / volatility / nine states) ---
    ("TR_t=max",
     r"$\mathrm{TR}_t=\max\left(H_t-L_t,\ \left|H_t-C_{t-1}\right|,\ "
     r"\left|L_t-C_{t-1}\right|\right)$"),
    ("ATR_14=",
     r"$\mathrm{ATR}_{14}=\frac{1}{14}\sum_{i=0}^{13}\mathrm{TR}_{t-i}"
     r"\,,\qquad \mathrm{atr_{norm}}=\mathrm{ATR}_{14}\,/\,P_t$"),
    ("MA_20=",
     r"$\mathrm{MA}_{20}=\frac{1}{20}\sum_{i=0}^{19}P_{t-i}\,,\qquad"
     r"\ \mathrm{ma_{gap}}=\frac{P_t}{\mathrm{MA}_{20}}-1$"),
    ("σ_5=",
     r"$\sigma_5=\sqrt{\frac{1}{5}\sum_{i=0}^{4}\left(r_{t-i}-\bar{r}"
     r"\right)^{2}}\,,\qquad r_t=\frac{P_t}{P_{t-1}}-1$"),
    ("ret_1=",
     r"$\mathrm{ret}_1=\frac{P_t}{P_{t-1}}-1\,,\quad"
     r"\ \mathrm{mom}_5=\frac{P_t}{P_{t-5}}-1\,,\quad"
     r"\ \mathrm{ma_{gap}}=\frac{P_t}{\mathrm{MA}_{20}}-1$"),
    ("vol_5=",
     r"$\mathrm{vol}_5=\sqrt{\frac{1}{5}\sum_{i=0}^{4}\left(r_{t-i}"
     r"-\bar{r}\right)^{2}}\,,\qquad"
     r"\ \mathrm{atr_{norm}}=\mathrm{ATR}_{14}\,/\,P_t$"),
    ("clock=",
     r"$\mathrm{clock}=\frac{t}{T}\,,\quad"
     r"\ \mathrm{cash}=\frac{C_t}{C_0}\,,\quad"
     r"\ \mathrm{exposure}=\frac{h_t\,P_t}{C_0}\,,\quad"
     r"\ \mathrm{pnl}=\frac{P_t-\bar{P}_{\mathrm{entry}}}"
     r"{\bar{P}_{\mathrm{entry}}}$"),
    ("drawdown=",
     r"$\mathrm{drawdown}=\frac{W^{\mathrm{peak}}-W_t}{W^{\mathrm{peak}}}$"),
    ("r_t=W_{t+1}",
     r"$r_t=W_{t+1}-W_t\,,\qquad W_t=C_t+h_t P_t$"),
    ("L_{RF}(θ)=",
     r"$L_{RF}(\theta)=-\sum_{t}\log\pi_\theta(a_t\mid s_t)\cdot\hat{G}_t\,,"
     r"\qquad \hat{G}_t=\frac{G_t-\bar{G}}{\mathrm{std}(G)}$"),
    ("H(π_θ)=",
     r"$H(\pi_\theta)=-\sum_{a}\pi_\theta(a\mid s)\,\log\pi_\theta(a\mid s)$"),
    ("A(s_t)=",
     r"$\mathcal{A}(s_t)=\{\mathrm{hold}\}\ \cup\ \{\mathrm{buy}\ \mathrm{if}"
     r"\ C_t\geq\mathrm{slice\ cost}\}\ \cup\ \{\mathrm{sell}\ \mathrm{if}"
     r"\ h_t>0\}$"),
    ("π_θ^{mask}(a|s)=",
     r"$\pi_\theta^{mask}(a\mid s)=\mathrm{softmax}(z_a+m_a)\,,\quad"
     r"m_a=0\ \mathrm{if}\ a\in\mathcal{A}(s)\,,\ \ {-10^{9}}"
     r"\ \mathrm{otherwise}$"),
    ("y=r+γ",
     r"$y=r+\gamma\max_{a'\in\mathcal{A}(s')}Q_{\theta^-}(s',a')\,,\qquad"
     r"a_t=\arg\max_{a\in\mathcal{A}(s_t)}Q_\theta(s_t,a)$"),
    ("max_θ𝔼[",
     r"$\max_\theta\ \mathbb{E}\left[\frac{\pi_\theta(a\mid s)}"
     r"{\pi_{\theta_{old}}(a\mid s)}\,\hat{A}\right]\quad\mathrm{s.t.}"
     r"\quad KL(\pi_\theta\,\Vert\,\pi_{\theta_{old}})\leq\delta$"),
    ("r_t^{risk}=",
     r"$r_t^{risk}=\Delta W_t-\lambda\,C_0\,\max(0,\ dd_{t+1}-dd_t)\,,"
     r"\qquad\lambda=0.25$"),
    ("dd_t=",
     r"$dd_t=\left(W_t^{peak}-W_t\right)/\,W_t^{peak}$"),
    ("ret_1=",
     r"$ret_1=\frac{P_t}{P_{t-1}}-1\,,\quad mom_5=\frac{P_t}{P_{t-5}}-1"
     r"\,,\quad ma\_gap=\frac{P_t}{MA_{20}}-1$"),
    ("vol_5=",
     r"$vol_5=\mathrm{std}(r_{t-4},\ldots,r_t)\,,\quad "
     r"atr\_norm=\overline{TR}_{14}\,/\,P_t$"),
    ("Sharpe=",
     r"$Sharpe=\frac{\bar{R}_p-R_f}{\sigma_p}\,,\qquad here:\ "
     r"\bar{R}=\mathrm{mean}\!\left(\Delta W/C_0\right),\ R_f=0$"),
]


def _match_eq(text: str):
    stripped = text.replace(" ", "")
    for key, tex in EQ_TEX:
        if stripped.startswith(key):
            return tex
    return None


_EQ_CACHE: dict = {}


def _render_eq(tex: str):
    """Typeset with mathtext → (base64 PNG, display height in --fs units)."""
    if tex in _EQ_CACHE:
        return _EQ_CACHE[tex]
    import base64
    import io
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    fig = plt.figure(figsize=(14, 2.5))
    fig.text(0.5, 0.5, tex, fontsize=30, color="#FFD9A0",
             ha="center", va="center")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=300, transparent=True,
                bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    data = base64.b64encode(buf.getvalue()).decode()
    buf.seek(0)
    h = Image.open(buf).size[1]
    ratio = max(0.9, min(3.5, h / 120))   # height in units of --fs
    _EQ_CACHE[tex] = (data, ratio)
    return data, ratio

STOP = set("""the a an and or but of to in on at is are was were it its we i
you so that this as for with be by not now if can from one all what why how
your do does did when where which who whose whom would could should about
have has had them they he she his her my me us our their there here up out
no yes than then very much more most any some just like get got
say said asked ask question answer""".split())


def norm(w: str) -> str:
    return re.sub(r"[^a-z0-9]", "", w.lower())


def tokens(text: str) -> list[str]:
    out = []
    for w in text.split():
        t = norm(w)
        if t and t not in STOP and (len(t) >= 3
                                    or (t.isdigit() and len(t) >= 2)):
            out.append(t[:5])          # 5-char prefix stem ("64" survives)
    return out


def parse(md: str):
    """Return list of cards: {section, title, paras[(kind, text)]}."""
    cards = []
    section = ""
    cur = None

    def flush(c):
        if c and c["_buf"]:
            text = " ".join(c["_buf"])
            if c["_kind"] == "li":
                c["paras"].append(("li", text))
            else:
                c["paras"].append(_classify(text))
            c["_buf"] = []
        if c:
            c["_kind"] = "p"

    for raw in md.splitlines():
        line = raw.rstrip()
        m = re.match(r"^## (.+)$", line)
        if m:
            section = m.group(1).strip()
            cur = None
            continue
        m = re.match(r'^### (?:Q: )?"?(.+?)"?\s*$', line)
        if m and section:
            cur = {"section": section, "title": m.group(1),
                   "paras": [], "_buf": [], "_kind": "p"}
            cards.append(cur)
            continue
        if cur is None:
            continue
        if line.startswith(">"):
            body = line.lstrip("> ").rstrip()
            if not body:
                flush(cur)
            elif body.startswith("- "):
                flush(cur)
                cur["_kind"] = "li"
                cur["_buf"].append(body[2:])
            elif re.match(r"^\d+\. ", body):
                flush(cur)
                cur["_kind"] = "li"
                cur["_buf"].append(body)
            else:
                # continuation line joins whatever is open (para or li)
                cur["_buf"].append(body)
        elif line.strip() == "":
            flush(cur)
    for c in cards:
        flush(c)
        del c["_buf"], c["_kind"]
    return [c for c in cards if c["paras"]]


def _classify(text: str):
    """A paragraph is an equation only if we have proper LaTeX for it;
    prose that merely mentions symbols stays prose (readable serif)."""
    if _match_eq(text):
        return ("eq", text)
    return ("p", text)


def keyword_weights(cards):
    """Per-card weighted keyword bags; rare terms weigh more."""
    bags = []
    for c in cards:
        q_toks = tokens(c["title"])
        a_toks = tokens(" ".join(t for _, t in c["paras"]))
        bags.append((set(q_toks), set(q_toks) | set(a_toks)))
    df = {}
    for _, full in bags:
        for t in full:
            df[t] = df.get(t, 0) + 1
    out = []
    for q_set, full in bags:
        kw = {}
        for t in full:
            w = 3.0 if df[t] <= 2 else (1.5 if df[t] <= 5 else 0.6)
            if t in q_set:
                w += 3.0            # words from the question itself
            kw[t] = round(w, 1)
        out.append(kw)
    return out


SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"$\u201c(])")


def build():
    cards = parse(SRC.read_text())
    kws = keyword_weights(cards)

    sections_html = []
    last_section = None
    js_questions = []      # {q, first, kw, title}
    line_id = 0
    for qi, c in enumerate(cards):
        if c["section"] != last_section:
            sections_html.append(
                f'<p class="sec">{html.escape(c["section"])}</p>')
            last_section = c["section"]
        body = []
        first_line = line_id
        for kind, text in c["paras"]:
            if kind == "eq":
                data, ratio = _render_eq(_match_eq(text))
                body.append(
                    f'<p class="line eqline" id="L{line_id}" '
                    f'data-i="{line_id}"><img alt="{html.escape(text)}" '
                    f'src="data:image/png;base64,{data}" '
                    f'style="height:calc(var(--fs) * {ratio:.2f})"></p>')
                line_id += 1
            elif kind == "li":
                bullet = "" if text[:1].isdigit() else "• "
                body.append(
                    f'<p class="line li" id="L{line_id}" '
                    f'data-i="{line_id}">{bullet}{html.escape(text)}</p>')
                line_id += 1
            else:
                for sent in SENT_SPLIT.split(text):
                    sent = sent.strip()
                    if not sent:
                        continue
                    body.append(
                        f'<p class="line" id="L{line_id}" '
                        f'data-i="{line_id}">{html.escape(sent)}</p>')
                    line_id += 1
        sections_html.append(f"""
<section class="slide" id="Q{qi}">
  <h2><span class="chip num">Q{qi + 1}</span> {html.escape(c["title"])}</h2>
  {''.join(body)}
</section>""")
        js_questions.append({"q": qi, "first": first_line,
                             "kw": kws[qi], "title": c["title"]})

    page = (TEMPLATE
            .replace("__SECTIONS__", "\n".join(sections_html))
            .replace("__QUESTIONS__", json.dumps(js_questions))
            .replace("__NLINES__", str(line_id)))
    OUT.write_text(page)
    SERVE_DIR.mkdir(exist_ok=True)
    (SERVE_DIR / "qa.html").write_text(page)
    print(f"Wrote {OUT} + serve copy ({len(cards)} questions, "
          f"{line_id} lines)")


TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Nguyen Q&A — read-along</title>
<style>
  :root { --fs: 30px; }
  * { box-sizing: border-box; }
  html, body { margin: 0; padding: 0; }
  body {
    background: #10151f;
    color: #e8e6e0;
    font-family: Charter, Georgia, "Times New Roman", serif;
  }
  header {
    position: fixed; top: 0; left: 0; right: 0; z-index: 10;
    display: flex; align-items: center; gap: 14px;
    padding: 10px 22px;
    background: rgba(16,21,31,0.94);
    border-bottom: 1px solid #263143;
    font-family: -apple-system, "SF Pro Text", Helvetica, sans-serif;
    font-size: 14px; color: #93a1b8;
  }
  header .title { font-weight: 600; color: #cfd8e6; }
  header input {
    flex: 1; max-width: 380px; background: #1d2635; color: #e8e6e0;
    border: 1px solid #33415a; border-radius: 8px; padding: 6px 12px;
    font-size: 14px;
  }
  header button {
    background: #1d2635; color: #cfd8e6; border: 1px solid #33415a;
    border-radius: 8px; padding: 6px 14px; font-size: 14px; cursor: pointer;
  }
  header button:hover { background: #263349; }
  header button.on { background: #14532d; border-color: #1e7a42; color: #d7f5e2; }
  #heard { font-style: italic; color: #64748b; max-width: 280px;
           white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  main {
    max-width: 1180px;
    margin: 0 auto;
    padding: 42vh 60px 55vh 60px;   /* current line sits mid-screen */
  }
  .sec {
    font-family: -apple-system, "SF Pro Text", Helvetica, sans-serif;
    font-size: 15px; font-weight: 700; color: #5f7292;
    text-transform: uppercase; letter-spacing: 0.06em;
    margin: 2.8em 0 0.2em 0;
  }
  .slide h2 {
    font-family: -apple-system, "SF Pro Text", Helvetica, sans-serif;
    font-size: 21px; font-weight: 700; color: #cfd8e6;
    margin: 1.6em 0 0.9em 0; letter-spacing: 0.01em; line-height: 1.4;
  }
  .chip {
    display: inline-block; padding: 2px 10px; border-radius: 999px;
    background: #1d2635; border: 1px solid #33415a;
    font-size: 13px; font-weight: 600; color: #9fb3d1;
    vertical-align: 2px; margin-right: 6px;
  }
  .chip.num { background: #24344e; color: #cfe0f5; }
  .line {
    font-size: var(--fs);
    line-height: 1.55;
    margin: 0.55em 0;
    padding-left: 22px;
    border-left: 4px solid transparent;
    opacity: 0.42;
    cursor: pointer;
    transition: opacity .25s, border-color .25s, color .25s;
  }
  .line.past { opacity: 0.22; }
  .line.current {
    opacity: 1; color: #ffffff;
    border-left-color: #e8912d;
  }
  .line.eqline { padding-top: 4px; padding-bottom: 4px; }
  .line.eqline img { display: block; max-width: 100%; }
  .line.eqline.current { filter: brightness(1.25); }
  .slide.dim { display: none; }
  #sugg {
    position: fixed; bottom: 18px; left: 50%; transform: translateX(-50%);
    display: none; align-items: center; gap: 8px; z-index: 20;
    max-width: 92vw; flex-wrap: wrap; justify-content: center;
    font-family: -apple-system, "SF Pro Text", Helvetica, sans-serif;
  }
  #sugg .stag { font-size: 13px; color: #7f92ad; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.05em; }
  #sugg button {
    background: #24344e; color: #cfe0f5; border: 1px solid #3d5378;
    border-radius: 999px; padding: 8px 16px; font-size: 14px;
    font-weight: 600; cursor: pointer;
  }
  #sugg button:hover { background: #2f4266; }
  body.light #sugg .stag { color: #5a6b85; }
  body.light #sugg button { background: #dce6f5; color: #24344e;
    border-color: #b3c4de; }
  body.light #sugg button:hover { background: #cbd9ef; }
  footer {
    font-family: -apple-system, sans-serif; font-size: 13px;
    color: #4b5a70; text-align: center; padding-bottom: 40px;
  }

  /* ---- light mode (14 Sep — the dark page reflected in Fiyin's
     glasses on camera). The equation PNGs are warm gold, so in light
     mode they are inverted to a dark ink blue. ---- */
  body.light { background: #f7f4ee; color: #2a3040; }
  body.light header { background: rgba(247,244,238,0.95);
    border-bottom: 1px solid #d8d2c4; color: #5a6578; }
  body.light header .title { color: #33415a; }
  body.light header input { background: #eae5da; color: #2a3040;
    border-color: #c9c2b2; }
  body.light header button { background: #eae5da; color: #33415a;
    border-color: #c9c2b2; }
  body.light header button:hover { background: #e0dacc; }
  body.light header button.on { background: #d9f2e2;
    border-color: #2f9e5d; color: #14532d; }
  body.light #heard { color: #8a94a6; }
  body.light .sec { color: #7a8296; }
  body.light .slide h2 { color: #33415a; }
  body.light .chip { background: #eae5da; border-color: #c9c2b2;
    color: #4c5c74; }
  body.light .chip.num { background: #dce6f5; color: #24344e; }
  body.light .line.current { color: #000000; }
  body.light .line.eqline img { filter: invert(0.85); }
  body.light .line.eqline.current { filter: none; }
  body.light footer { color: #a49f92; }
</style>
</head>
<body>
<header>
  <span class="title">nguyen Q&amp;A — read-along</span>
  <input id="search" placeholder="filter questions · Enter jumps · Esc clears">
  <span id="heard"></span>
  <button id="voiceBtn">🎤 trace questions</button>
  <button id="themeBtn">☀️ light</button>
  <button id="smaller">A−</button>
  <button id="bigger">A+</button>
</header>
<main>
__SECTIONS__
<footer>↓ / space = next line &nbsp;·&nbsp; ↑ = back &nbsp;·&nbsp; click any line to jump &nbsp;·&nbsp; voice tracing jumps to the question it hears &nbsp;·&nbsp; / focuses search</footer>
</main>
<div id="sugg"></div>
<script>
const QUESTIONS = __QUESTIONS__;
const NLINES = __NLINES__;
let cur = 0;

function render() {
  for (let i = 0; i < NLINES; i++) {
    const el = document.getElementById("L" + i);
    el.classList.toggle("current", i === cur);
    el.classList.toggle("past", i < cur);
  }
  const el = document.getElementById("L" + cur);
  if (el) el.scrollIntoView({ behavior: "smooth", block: "center" });
}
function go(i) { cur = Math.max(0, Math.min(NLINES - 1, i)); render(); }

document.querySelectorAll(".line").forEach(el =>
  el.addEventListener("click", () => go(+el.dataset.i)));

const search = document.getElementById("search");
document.addEventListener("keydown", e => {
  if (document.activeElement === search) return;
  if (e.key === "ArrowDown" || e.key === " ") { e.preventDefault(); go(cur + 1); }
  if (e.key === "ArrowUp") { e.preventDefault(); go(cur - 1); }
  if (e.key === "Home") { e.preventDefault(); go(0); }
  if (e.key === "/") { e.preventDefault(); search.focus(); }
});

/* ---- search: hides non-matching questions ---- */
search.addEventListener("input", () => {
  const q = search.value.trim().toLowerCase();
  document.querySelectorAll(".slide").forEach(el => {
    el.classList.toggle("dim",
      q !== "" && !el.textContent.toLowerCase().includes(q));
  });
});
search.addEventListener("keydown", e => {
  if (e.key === "Enter") {
    e.preventDefault();
    const first = document.querySelector(".slide:not(.dim)");
    if (first) {
      const qi = +first.id.slice(1);
      search.value = ""; search.dispatchEvent(new Event("input"));
      search.blur();
      go(QUESTIONS[qi].first);
    }
  }
  if (e.key === "Escape") {
    search.value = ""; search.dispatchEvent(new Event("input")); search.blur();
  }
});

/* ---- font size (same as the transcript prompter) ---- */
let fs = +(localStorage.getItem("qafs") || 30);
function applyFs() {
  document.documentElement.style.setProperty("--fs", fs + "px");
  localStorage.setItem("qafs", fs);
}
document.getElementById("bigger").onclick = () => { fs += 2; applyFs(); };
document.getElementById("smaller").onclick = () => { fs = Math.max(18, fs - 2); applyFs(); };
applyFs();

/* ---- light / dark (both windows switch together via localStorage) ---- */
function applyTheme() {
  const light = localStorage.getItem("vivaTheme") === "light";
  document.body.classList.toggle("light", light);
  document.getElementById("themeBtn").textContent = light ? "🌙 dark" : "☀️ light";
}
document.getElementById("themeBtn").onclick = () => {
  localStorage.setItem("vivaTheme",
    localStorage.getItem("vivaTheme") === "light" ? "dark" : "light");
  applyTheme();
};
window.addEventListener("storage", applyTheme);  // follow the other window
applyTheme();

/* ---- voice tracing ----
   Nguyen paraphrases, so this doesn't track lines — it scores every
   question's weighted keyword bag against the last 14 heard words and
   jumps to that question's first line only when one clearly wins
   (absolute threshold + 40% margin over the runner-up). Repeating the
   question back before answering feeds it the keywords reliably. */
const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
let rec = null, voiceOn = false;
const heard = [];

/* ONE mic for both windows, but following is OPT-IN (14 Sep 3am —
   the board was jumping around DURING the presentation). The
   transcript window owns the mic; this page stays quiet until Fiyin
   clicks the button when the questions start. No second permission
   prompt: it consumes the shared words over the BroadcastChannel. */
let armed = false;          // follow the shared mic?
let lastBroadcast = 0;      // when words last arrived on the channel
const bc = window.BroadcastChannel ? new BroadcastChannel("viva-voice") : null;
if (bc) bc.onmessage = e => {
  lastBroadcast = Date.now();
  if (voiceOn || !armed) return;   // own mic running, or not armed yet
  for (const w of e.data) heard.push(w);
  document.getElementById("heard").textContent =
    "…" + heard.slice(-8).join(" ");
  scoreAll();
};

if (location.protocol === "file:") {
  const w = document.createElement("div");
  w.style.cssText = "position:fixed;top:52px;left:0;right:0;z-index:99;" +
    "background:#7a1f1f;color:#ffe;padding:10px 22px;font:15px " +
    "-apple-system,sans-serif;text-align:center";
  w.innerHTML = "You opened the FILE copy — the mic permission will not " +
    "stick and Chrome will keep asking. Use " +
    "<a href='http://localhost:8749/qa.html' " +
    "style='color:#ffd9a0;font-weight:700'>localhost:8749/qa.html</a>";
  document.body.appendChild(w);
}
const STOPJS = new Set(("the a an and or but of to in on at is are was were "
 + "it its we i you so that this as for with be by not now if can from one "
 + "all what why how your do does did when where which who would could "
 + "should about have has had them they he she his her my me us our their "
 + "there here up out no yes than then very much more most any some just "
 + "like get got say said asked ask question answer okay ok let "
 + "lets move moving next part parts please discussion discuss talk tell "
 + "mean means look looking going right well good sure maybe actually "
 + "basically thing things also see seems seem still bit little").split(" "));

function normJS(w) {
  let t = w.toLowerCase().replace(/[^a-z0-9]/g, "");
  if (t.startsWith("thous")) t = "3000";   // "three thousand" → 3,000
  t = t.replace(/^artif/, "artef");        // US spelling from the recogniser
  return t;
}

function scoreAll() {
  const window14 = heard.slice(-14);
  const toks = new Set();
  for (let i = 0; i < window14.length; i++) {
    const t = normJS(window14[i]);
    // "moving" is chatter ("moving on…") EXCEPT in "moving average"
    if (t === "moving" && i + 1 < window14.length &&
        normJS(window14[i + 1]).startsWith("aver")) {
      toks.add("movin");
      continue;
    }
    if (t && !STOPJS.has(t) &&
        (t.length >= 3 || (/^[0-9]+$/.test(t) && t.length >= 2)))
      toks.add(t.slice(0, 5));
  }
  if (toks.size < 1) return;   // a single RARE keyword may still jump
                               // (e.g. "artefacts") — the strong/margin
                               // gates below decide, not the count here
  const scored = [];
  for (const q of QUESTIONS) {
    let s = 0, n = 0;
    for (const t of toks) if (q.kw[t]) { s += q.kw[t]; n++; }
    if (s > 0) scored.push({ q: q, s: s, n: n });
  }
  if (!scored.length) return;
  scored.sort((a, b) => b.s - a.s);
  const best = scored[0];
  const sec = scored[1] || { q: null, s: 0, n: 0 };
  // jump on: two distinct keywords with decent weight, OR one keyword
  // strong enough that it must be a rare word from the question title
  // (>= 4.5). The margin over the runner-up must be clear — unless the
  // runner-up is a neighbouring question (same topic block), or the
  // winner matched clearly MORE distinct keywords (a runner-up scoring
  // high off one shared title word must not veto a 3-keyword match).
  // a RARE title word ("pyramid", "artefacts") is near-unique to its
  // question — hearing one is decisive even on a thin ratio margin
  let hasRare = false;
  for (const t of toks) if (best.q.kw[t] >= 5.5) { hasRare = true; break; }
  const strong = best.n >= 2 ? best.s >= 4 : best.s >= 3.5;
  const margin = best.s >= 1.5 * sec.s ||
                 (sec.q && Math.abs(best.q.q - sec.q.q) <= 2 &&
                  best.n >= 2 && best.s >= 3.5) ||
                 (best.n >= sec.n + 2 && best.s > sec.s) ||
                 (hasRare && best.s > sec.s);
  if (strong && margin && best.q.first !== cur) {
    go(best.q.first);
    heard.length = 0;          // consume the window after a jump
    hideSugg();
    return;
  }
  // no confident jump — offer the closest cards as one-click escapes
  // (14 Sep morning: for an UNSEEN question, the nearest prepared
  // answer is almost always one of these)
  showSugg(scored.filter(x => x.s >= 2).slice(0, 3));
}

/* ---- near-miss suggestion strip ---- */
let suggTimer = null;
function hideSugg() {
  document.getElementById("sugg").style.display = "none";
}
function showSugg(items) {
  const el = document.getElementById("sugg");
  if (!items.length) { hideSugg(); return; }
  el.innerHTML = "";
  const tag = document.createElement("span");
  tag.className = "stag";
  tag.textContent = "close:";
  el.appendChild(tag);
  for (const it of items) {
    const b = document.createElement("button");
    b.textContent = "Q" + it.q.q + " · " +
      (it.q.title.length > 44 ? it.q.title.slice(0, 44) + "…" : it.q.title);
    b.onclick = () => { go(it.q.first); heard.length = 0; hideSugg(); };
    el.appendChild(b);
  }
  el.style.display = "flex";
  clearTimeout(suggTimer);
  suggTimer = setTimeout(hideSugg, 12000);
}

document.getElementById("voiceBtn").onclick = () => {
  const btn = document.getElementById("voiceBtn");
  // already tracing? one click turns it OFF, whichever mode
  if (armed) { armed = false;
    btn.classList.remove("on");
    btn.textContent = "🎤 trace questions";
    document.getElementById("heard").textContent = ""; return; }
  if (voiceOn) { voiceOn = false; rec.stop();
    btn.classList.remove("on");
    btn.textContent = "🎤 trace questions";
    document.getElementById("heard").textContent = ""; return; }
  // if the transcript window's mic is live, FOLLOW it — never open a
  // second recogniser next to it
  if (Date.now() - lastBroadcast < 8000) {
    armed = true;
    btn.classList.add("on");
    btn.textContent = "🎤 following shared mic";
    document.getElementById("heard").textContent = "listening via the other window…";
    return;
  }
  if (!SR) { document.getElementById("heard").textContent =
    "not supported — use Chrome"; return; }
  rec = new SR();
  rec.lang = "en-GB"; rec.continuous = true; rec.interimResults = true;
  let fed = 0;
  let fatal = false;
  rec.onresult = e => {
    const r = e.results[e.results.length - 1];
    const words = r[0].transcript.trim().split(/\s+/).filter(Boolean);
    if (words.length > fed) {
      const fresh = words.slice(fed);
      for (const w of fresh) heard.push(w);
      if (bc) bc.postMessage(fresh);   // share the mic with transcript.html
      document.getElementById("heard").textContent =
        "…" + heard.slice(-8).join(" ");
      scoreAll();
    }
    fed = r.isFinal ? 0 : words.length;
  };
  rec.onerror = ev => {
    if (ev.error === "not-allowed" || ev.error === "service-not-allowed") {
      fatal = true; voiceOn = false;
      document.getElementById("heard").textContent =
        "MIC BLOCKED — use localhost:8749 and click Allow";
      document.getElementById("voiceBtn").classList.remove("on");
    } else if (ev.error === "aborted") {
      fatal = true; voiceOn = false;   // other window took the mic
      armed = true;                    // so follow it instead
      document.getElementById("heard").textContent =
        "following the other window's mic";
      document.getElementById("voiceBtn").textContent = "🎤 following shared mic";
    }
  };
  rec.onend = () => { if (voiceOn && !fatal) rec.start(); };
  rec.start(); voiceOn = true;
  document.getElementById("voiceBtn").classList.add("on");
  document.getElementById("voiceBtn").textContent = "🎤 tracing (own mic)";
};

render();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    build()
