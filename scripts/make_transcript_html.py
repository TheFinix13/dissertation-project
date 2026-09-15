#!/usr/bin/env python3
"""Build notes/viva/transcript.html — a teleprompter page — from
notes/viva/transcript.md.

Design goals (Fiyin, 12 Sep):
  1. Nice and readable FIRST: large serif lines, dark background, one
     column, current line highlighted, past lines dimmed.
  2. Voice tracking like Spotify lyrics: Web Speech API listens and
     advances the highlight as he reads. Manual fallback: arrow keys,
     space, or click any line.

The page lives on the LG UltraGear while the MacBook camera holds his face.
Regenerate after every transcript.md edit:  ./venv/bin/python scripts/make_transcript_html.py
"""

import html
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "notes" / "viva" / "transcript.md"
OUT = ROOT / "notes" / "viva" / "transcript.html"

SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"$\u201c(])")


def parse(md: str):
    """Return list of slides: {num, title, time, lines[], careful[], pending}."""
    slides = []
    cur = None
    block = []          # accumulating paragraph lines
    in_careful = False
    careful_buf = []

    def flush_paragraph():
        if not block:
            return
        text = " ".join(block).strip()
        block.clear()
        if not text or cur is None:
            return
        if "RECORDING PENDING" in text or text.startswith("Waiting on your recording"):
            cur["pending"] = True
            cur["lines"].append(text)
            return
        for sent in SENT_SPLIT.split(text):
            sent = sent.strip()
            if sent:
                cur["lines"].append(sent)

    def flush_careful():
        nonlocal careful_buf
        if careful_buf and cur is not None:
            text = " ".join(careful_buf).strip()
            text = text.replace("**", "")
            cur["careful"].append(text)
        careful_buf = []

    for raw in md.splitlines():
        line = raw.rstrip()
        m = re.match(r"^## Slide (\d+) · (.+?)(?:\s*\((~[^)]+)\))?\s*$", line)
        if m:
            flush_paragraph()
            flush_careful()
            cur = {"num": int(m.group(1)),
                   "title": m.group(2).replace("(RECORDING PENDING)", "").strip(),
                   "time": m.group(3) or "",
                   "pending": "RECORDING PENDING" in line,
                   "lines": [], "careful": []}
            slides.append(cur)
            continue
        if line.startswith(">"):
            flush_paragraph()
            in_careful = True
            careful_buf.append(line.lstrip("> ").strip())
            continue
        if in_careful and line.strip() == "":
            flush_careful()
            in_careful = False
            continue
        if line.strip() in ("---",) or line.startswith("# "):
            flush_paragraph()
            flush_careful()
            in_careful = False
            continue
        if line.strip() == "":
            flush_paragraph()
            continue
        if cur is None:
            continue  # intro prose before the first slide heading
        block.append(line.strip())

    flush_paragraph()
    flush_careful()
    return slides


def build(slides):
    sections = []
    line_id = 0
    js_lines = []  # [{id, slide, text}] for the tracker
    for s in slides:
        body = []
        if s["pending"]:
            body.append(
                f'<p class="pending">Recording pending — the deck\u2019s speaker '
                f'notes carry this slide for now.</p>')
        else:
            for sent in s["lines"]:
                body.append(f'<p class="line" id="L{line_id}" '
                            f'data-i="{line_id}">{html.escape(sent)}</p>')
                js_lines.append({"i": line_id, "slide": s["num"],
                                 "text": sent})
                line_id += 1
        for c in s["careful"]:
            body.append(f'<aside class="careful">{html.escape(c)}</aside>')
        time_chip = f'<span class="chip">{html.escape(s["time"])}</span>' if s["time"] else ""
        sections.append(f"""
<section class="slide" id="slide{s['num']}">
  <h2><span class="chip num">Slide {s['num']}</span> {html.escape(s['title'])} {time_chip}</h2>
  {''.join(body)}
</section>""")

    lines_json = json.dumps(js_lines)
    return TEMPLATE.replace("__SECTIONS__", "\n".join(sections)) \
                   .replace("__LINES__", lines_json)


TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>viva transcript — read-along</title>
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
  header .title { font-weight: 600; color: #cfd8e6; margin-right: auto; }
  header button {
    background: #1d2635; color: #cfd8e6; border: 1px solid #33415a;
    border-radius: 8px; padding: 6px 14px; font-size: 14px; cursor: pointer;
  }
  header button:hover { background: #263349; }
  header button.on { background: #14532d; border-color: #1e7a42; color: #d7f5e2; }
  #timer { font-variant-numeric: tabular-nums; min-width: 52px; }
  #status { font-style: italic; }
  #status.ok { color: #58d68d; font-style: normal; font-weight: 600; }
  #status.bad { color: #ff7b6b; font-style: normal; font-weight: 600; }
  #heardT { font-style: italic; color: #64748b; max-width: 300px;
            white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  main {
    max-width: 1180px;
    margin: 0 auto;
    padding: 42vh 60px 55vh 60px;   /* current line sits mid-screen */
  }
  .slide h2 {
    font-family: -apple-system, "SF Pro Text", Helvetica, sans-serif;
    font-size: 20px; font-weight: 700; color: #7f92ad;
    margin: 2.4em 0 0.9em 0; letter-spacing: 0.01em;
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
  .careful {
    font-family: -apple-system, "SF Pro Text", Helvetica, sans-serif;
    font-size: 15px; line-height: 1.5; color: #d8b36a;
    border-left: 3px solid #a97c2f;
    background: rgba(169,124,47,0.08);
    padding: 10px 16px; margin: 1em 0 1em 22px; border-radius: 0 8px 8px 0;
    max-width: 62ch;
  }
  .careful::before { content: "CAREFUL — "; font-weight: 700; }
  .pending {
    font-family: -apple-system, "SF Pro Text", Helvetica, sans-serif;
    font-size: 17px; font-style: italic; color: #64748b;
    padding-left: 26px;
  }
  footer {
    font-family: -apple-system, sans-serif; font-size: 13px;
    color: #4b5a70; text-align: center; padding-bottom: 40px;
  }

  /* ---- light mode (14 Sep — the dark page reflected in Fiyin's
     glasses on camera; warm paper, no glare, same layout) ---- */
  body.light { background: #f7f4ee; color: #2a3040; }
  body.light header { background: rgba(247,244,238,0.95);
    border-bottom: 1px solid #d8d2c4; color: #5a6578; }
  body.light header .title { color: #33415a; }
  body.light header button { background: #eae5da; color: #33415a;
    border-color: #c9c2b2; }
  body.light header button:hover { background: #e0dacc; }
  body.light header button.on { background: #d9f2e2;
    border-color: #2f9e5d; color: #14532d; }
  body.light #status.ok { color: #1e7a42; }
  body.light #status.bad { color: #c0392b; }
  body.light #heardT { color: #8a94a6; }
  body.light .slide h2 { color: #5a6b85; }
  body.light .chip { background: #eae5da; border-color: #c9c2b2;
    color: #4c5c74; }
  body.light .chip.num { background: #dce6f5; color: #24344e; }
  body.light .line.current { color: #000000; }
  body.light .careful { color: #7a5a14; background: rgba(169,124,47,0.10); }
  body.light .pending { color: #8a94a6; }
  body.light footer { color: #a49f92; }
</style>
</head>
<body>
<header>
  <span class="title">viva read-along</span>
  <span id="status">voice: off</span>
  <span id="heardT"></span>
  <span id="pace"></span>
  <button id="voiceBtn">🎤 start voice tracking</button>
  <button id="followBtn" style="display:none"
          title="mic stays on for the Q&A board — only the auto-scroll stops">🧭 follow: on</button>
  <button id="themeBtn">☀️ light</button>
  <button id="smaller">A−</button>
  <button id="bigger">A+</button>
  <button id="timerBtn">⏱ start</button>
  <span id="timer">00:00</span>
</header>
<main>
__SECTIONS__
<footer>↓ / space = next line &nbsp;·&nbsp; ↑ = back &nbsp;·&nbsp; click any line to jump &nbsp;·&nbsp; voice tracking advances automatically as you read</footer>
</main>
<script>
const LINES = __LINES__;
let cur = 0;
let started = null;

/* Pace budget (13 Sep, from the 22.7-min practice run): the second is
   the clock time each slide should be FINISHED by, summing to 19:30. */
const DEADLINE = {1: 30, 2: 90, 3: 180, 4: 285, 5: 390, 6: 510,
                  7: 615, 8: 735, 9: 900, 10: 1035, 11: 1170};
function fmt(sec) {
  return String(Math.floor(sec / 60)).padStart(2, "0") + ":" +
         String(sec % 60).padStart(2, "0");
}
function updatePace() {
  const slide = (LINES[cur] || {}).slide;
  if (!slide || !DEADLINE[slide]) return;
  const el = document.getElementById("pace");
  el.textContent = "finish slide " + slide + " by " + fmt(DEADLINE[slide]);
  if (started !== null) {
    const elapsed = (Date.now() - started) / 1000;
    const late = elapsed > DEADLINE[slide];
    el.style.color = late ? "#e06c5a" : "#6fbf8a";
    document.getElementById("timer").style.color =
      late ? "#e06c5a" : "#93a1b8";
  }
}

function norm(w) { return w.toLowerCase().replace(/[^a-z0-9]/g, ""); }
const tokens = LINES.map(l => l.text.split(/\s+/).map(norm).filter(Boolean));

function render() {
  LINES.forEach(l => {
    const el = document.getElementById("L" + l.i);
    el.classList.toggle("current", l.i === cur);
    el.classList.toggle("past", l.i < cur);
  });
  const el = document.getElementById("L" + cur);
  if (el) el.scrollIntoView({ behavior: "smooth", block: "center" });
  updatePace();
}
function go(i) {
  cur = Math.max(0, Math.min(LINES.length - 1, i));
  ptr = 0;
  matched = 0;
  render();
}

/* Clicking a line is a hard anchor: "I am HERE." The tracker starts
   fresh at that line, forgets recent mishears (so a stale resync
   cannot fire), and silently absorbs leftover words of the previous
   line. From an anchor it can only move ONE line forward, and only
   on voice evidence for that next line. */
document.querySelectorAll(".line").forEach(el =>
  el.addEventListener("click", () => {
    const i = +el.dataset.i;
    go(i);
    nextPtr = 0;
    missCount = 0;
    recentHeard.length = 0;
    tailLine = i - 1;
    tailPtr = 0;
  }));

document.addEventListener("keydown", e => {
  if (e.key === "ArrowDown" || e.key === " ") { e.preventDefault(); go(cur + 1); }
  if (e.key === "ArrowUp") { e.preventDefault(); go(cur - 1); }
  if (e.key === "Home") { e.preventDefault(); go(0); }
});

/* ---- timer: a plain stopwatch (14 Sep — keep the controls simple).
   One button. Click = start from 00:00. Click again = stop (the time
   stays on screen). Click again = start a fresh run from 00:00. ---- */
let timerInterval = null;
document.getElementById("timerBtn").onclick = () => {
  const b = document.getElementById("timerBtn");
  if (timerInterval) {                     // running -> stop
    clearInterval(timerInterval);
    timerInterval = null;
    started = null;
    b.textContent = "⏱ start";
    b.classList.remove("on");
    return;
  }
  started = Date.now();                    // stopped -> run from zero
  document.getElementById("timer").textContent = "00:00";
  b.textContent = "⏱ stop";
  b.classList.add("on");
  timerInterval = setInterval(() => {
    const s = Math.floor((Date.now() - started) / 1000);
    document.getElementById("timer").textContent = fmt(s);
    updatePace();
  }, 1000);
  updatePace();
};

/* ---- follow toggle: the mic keeps running (and keeps feeding the
   Q&A board over the shared channel) but the transcript stops
   auto-scrolling when following is off ---- */
let followOn = true;
document.getElementById("followBtn").onclick = () => {
  followOn = !followOn;
  const b = document.getElementById("followBtn");
  b.textContent = followOn ? "🧭 follow: on" : "🧭 follow: OFF";
  b.classList.toggle("on", followOn);
  if (!followOn) setStatus("voice: heard but NOT moving the page — use ↓/↑", "");
};

/* ---- font size ---- */
let fs = +(localStorage.getItem("fs") || 30);
function applyFs() {
  document.documentElement.style.setProperty("--fs", fs + "px");
  localStorage.setItem("fs", fs);
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

/* ---- voice tracking (Spotify-style) ----
   Stricter rules (13 Sep — the first version ran ahead of the reader):
   - numbers only match spoken numbers, never ordinary words;
   - filler words (the, and, of...) only count at the exact pointer;
   - a line completes only when ~85% of it has been heard;
   - jumping to the next line needs TWO consecutive matching words,
     the first of them a content word;
   - auto-advance is rate-limited: never twice within 1.1 seconds. */
let ptr = 0;
let rec = null, voiceOn = false;
const SR = window.SpeechRecognition || window.webkitSpeechRecognition;

const STOP = new Set(["the","a","an","and","or","but","of","to","in","on",
  "at","is","are","was","were","it","its","we","i","you","so","that","this",
  "as","for","with","be","by","not","now","if","can","from","one","all"]);

function fuzzyEq(a, b) {
  if (!a || !b) return false;
  if (/\d/.test(a)) return /\d/.test(b);   // numbers only match numbers
  if (a === b) return true;
  return a.length >= 5 && b.length >= 5 &&
         (a.startsWith(b.slice(0, 5)) || b.startsWith(a.slice(0, 5)));
}

let nextPtr = 0;         // consecutive hits into the NEXT line
let lastAdvance = 0;
let matched = 0;         // words of the CURRENT line actually heard
let tailLine = -1;       // line just completed by an early advance...
let tailPtr = 0;         // ...and how far into it we had matched

function autoAdvance(toPtr) {
  const now = Date.now();
  if (now - lastAdvance < 1100) return;    // no line takes <1.1s to read
  lastAdvance = now;
  // remember the leftover of the line we are leaving: the reader will
  // still SPEAK its last words, and they must not count as misses on
  // the new line (14 Sep 3am — that pattern made resync drag the
  // highlight BACK to the finished line)
  tailLine = cur;
  tailPtr = ptr;
  go(cur + 1);
  ptr = toPtr;
  nextPtr = 0;
}

/* Stall recovery, tightened 13 Sep late evening. The first version
   resynced on TWO consecutive words over a 9-line window and it made
   the highlight leap ahead of the reader. Now: THREE consecutive
   content words, a short window (one line back, three ahead), closest
   line wins, and at most one resync every 4 seconds. */
let missCount = 0;
let lastResync = 0;
const recentHeard = [];

function resync() {
  const now = Date.now();
  if (now - lastResync < 2500) return false;
  const tail = recentHeard.slice(-3);
  if (tail.length < 3) return false;
  // ONE-LINE RULE (Fiyin, 14 Sep): the tracker may never move more
  // than one line at a time, and never backwards on its own. Resync
  // only re-anchors within the current line or steps to the next.
  const order = [cur, cur + 1];
  for (const L of order) {
    if (L < 0 || L >= LINES.length) continue;
    const exp = tokens[L] || [];
    for (let k = 0; k + 2 < exp.length; k++) {
      if (fuzzyEq(exp[k], tail[0]) && fuzzyEq(exp[k + 1], tail[1]) &&
          fuzzyEq(exp[k + 2], tail[2])) {
        if (L !== cur) go(L);
        ptr = k + 3;
        matched = 3;
        nextPtr = 0;
        missCount = 0;
        lastResync = now;
        return true;
      }
    }
  }
  return false;
}

function feed(words) {
  for (const wRaw of words) {
    const w = norm(wRaw);
    if (!w) continue;
    const exp = tokens[cur] || [];
    // content words may look 4 ahead (survives a dropped ASR word);
    // filler only matches at the pointer
    const ahead = STOP.has(w) ? 1 : 4;
    let hit = -1;
    for (let k = ptr; k < Math.min(ptr + ahead, exp.length); k++) {
      if (fuzzyEq(exp[k], w)) { hit = k; break; }
    }
    if (hit >= 0) {
      ptr = hit + 1;
      matched++;
      nextPtr = 0;
      missCount = 0;
      // a line completes when the pointer has reached ~70% of it AND
      // close to half its words were actually HEARD. (85%/55% proved
      // too slow on 14 Sep — the recogniser lags the voice, so the
      // highlight must move a beat BEFORE the line is fully spoken.)
      const need = exp.length <= 4 ? exp.length
                                   : Math.ceil(exp.length * 0.70);
      const heardEnough = matched >= Math.max(2, Math.ceil(exp.length * 0.45));
      if (ptr >= need && heardEnough) autoAdvance(0);
      continue;
    }
    // moved on already? require consecutive words of the next line —
    // two if a fair part of the current line was heard, three if not
    const nxt = tokens[cur + 1] || [];
    const nextNeed = matched >= exp.length * 0.35 ? 2 : 3;
    if (nextPtr > 0 && nextPtr < nxt.length && fuzzyEq(nxt[nextPtr], w)) {
      nextPtr++;
      if (nextPtr >= nextNeed) autoAdvance(nextPtr);
      continue;
    } else if (nxt.length && !STOP.has(w) && fuzzyEq(nxt[0], w)) {
      nextPtr = 1;
    } else {
      nextPtr = 0;
    }
    // leftover of the line we just advanced out of? consume silently
    if (tailLine === cur - 1) {
      const tl = tokens[tailLine] || [];
      let tHit = -1;
      for (let k = tailPtr; k < Math.min(tailPtr + 4, tl.length); k++) {
        if (fuzzyEq(tl[k], w)) { tHit = k; break; }
      }
      if (tHit >= 0) { tailPtr = tHit + 1; continue; }
    }
    // a genuine miss — remember it and consider resyncing
    if (!STOP.has(w)) {
      recentHeard.push(w);
      if (recentHeard.length > 6) recentHeard.shift();
      missCount++;
      if (missCount >= 4) resync();
    }
  }
}

function setStatus(t, cls) {
  const el = document.getElementById("status");
  el.textContent = t;
  el.className = cls || "";
}

/* ---- ONE mic for both windows ----
   transcript.html and qa.html share heard words over a
   BroadcastChannel: start voice tracking in EITHER window and the
   other follows the same microphone. (13 Sep night — the two pages
   were fighting over the mic and Chrome kept re-prompting.) */
const bc = window.BroadcastChannel ? new BroadcastChannel("viva-voice") : null;
if (bc) bc.onmessage = e => {
  if (voiceOn) return;                    // this window owns the mic
  feed(e.data);
  document.getElementById("heardT").textContent =
    "…" + e.data.slice(-6).join(" ");
  setStatus("voice: shared from the other window ✓", "ok");
};

/* file:// pages cannot remember mic permission — send Fiyin to the
   served copy where Chrome asks exactly once. */
if (location.protocol === "file:") {
  const w = document.createElement("div");
  w.style.cssText = "position:fixed;top:52px;left:0;right:0;z-index:99;" +
    "background:#7a1f1f;color:#ffe;padding:10px 22px;font:15px " +
    "-apple-system,sans-serif;text-align:center";
  w.innerHTML = "You opened the FILE copy — the mic permission will not " +
    "stick and Chrome will keep asking. Use " +
    "<a href='http://localhost:8749/transcript.html' " +
    "style='color:#ffd9a0;font-weight:700'>localhost:8749/transcript.html</a>";
  document.body.appendChild(w);
}

document.getElementById("voiceBtn").onclick = () => {
  if (!SR) { setStatus("voice: not supported in this browser — use Chrome"); return; }
  if (voiceOn) { voiceOn = false; rec.stop(); setStatus("voice: off");
    document.getElementById("voiceBtn").classList.remove("on");
    document.getElementById("followBtn").style.display = "none"; return; }
  document.getElementById("followBtn").style.display = "";
  rec = new SR();
  rec.lang = "en-GB";
  rec.continuous = true;
  rec.interimResults = true;
  let fed = 0;              // words of the live utterance already consumed
  let fatal = false;
  rec.onresult = e => {
    const r = e.results[e.results.length - 1];
    const words = r[0].transcript.trim().split(/\s+/).filter(Boolean);
    // feed only the words not seen before — NEVER replay an utterance
    // (the old version re-fed every final result from word 0, which is
    // what made the highlight run ahead of the reader)
    if (words.length > fed) {
      const fresh = words.slice(fed);
      if (followOn) feed(fresh);       // follow OFF = hear, don't scroll
      if (bc) bc.postMessage(fresh);   // share the mic with qa.html
      document.getElementById("heardT").textContent =
        "…" + words.slice(-6).join(" ");
      setStatus(followOn ? "voice: hearing you ✓"
                         : "voice: hearing (follow off)", "ok");
    }
    fed = r.isFinal ? 0 : words.length;
  };
  rec.onerror = ev => {
    if (ev.error === "not-allowed" || ev.error === "service-not-allowed") {
      fatal = true; voiceOn = false;
      setStatus("MIC BLOCKED — open http://localhost:8749/transcript.html " +
                "in Chrome and click Allow", "bad");
      document.getElementById("voiceBtn").classList.remove("on");
    } else if (ev.error === "aborted") {
      // the OTHER window took the mic — follow it over the channel
      fatal = true; voiceOn = false;
      setStatus("mic moved to the other window — following it here ✓", "ok");
      document.getElementById("voiceBtn").classList.remove("on");
    } else if (ev.error !== "no-speech") {
      setStatus("voice: error (" + ev.error + ") — restarting", "bad");
    }
  };
  rec.onend = () => { if (voiceOn && !fatal) rec.start(); };  // auto-restart
  rec.start();
  voiceOn = true;
  setStatus("voice: listening (say something to confirm)", "ok");
  document.getElementById("voiceBtn").classList.add("on");
};

render();
</script>
</body>
</html>
"""


SERVE_DIR = Path.home() / ".viva_prompter"   # served by the LaunchAgent
                                             # com.fiyin.viva-transcript on
                                             # http://localhost:8749 (Documents
                                             # is TCC-blocked for launchd)


def main():
    slides = parse(SRC.read_text())
    page = build(slides)
    OUT.write_text(page)
    SERVE_DIR.mkdir(exist_ok=True)
    (SERVE_DIR / "transcript.html").write_text(page)
    n_lines = sum(len(s["lines"]) for s in slides if not s["pending"])
    print(f"Wrote {OUT} + serve copy  ({len(slides)} slides, {n_lines} readable lines)")


if __name__ == "__main__":
    main()
