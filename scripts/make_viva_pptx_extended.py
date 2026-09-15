#!/usr/bin/env python3
"""Extended viva deck — the frozen 11-slide deck plus extra slides that
Fiyin directs, inserted where the flow wants them.

Current structure (12 slides):
  1-5   identical to the main deck (title ... dataset splits)
  6     NEW — the validation experiments (trade size, fee, risk
        coefficient lambda) and what they found
  7-12  identical to the main deck's slides 6-11

Rebuilt from scratch on 13 Sep: the old divider + backup-slides version
was retired ("somewhat unreliable and unnecessary" — Fiyin). The main
deck (make_viva_pptx.py) is FROZEN; only this file grows from here.
"""

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import make_viva_pptx as base
from make_viva_pptx import (
    SLIDE_W_IN, SLIDE_H_IN, NAVY, INK, GREY, ACCENT,
    blank_slide, add_title, add_caption, add_notes, add_slide_number,
    _panel, _points_textbox,
    slide1_title, slide2_decision, slide3_rl, slide4_algorithms,
    slide5_rules, slide6_first_result, slide8_diagnosis,
    slide9_simulator, slide10_flip, slide11_stress, slide12_closing,
)

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "latex" / "viva" / "viva_deck_extended.pptx"
FIGS_EXTRA = ROOT / "latex" / "viva" / "figs_extra"


def _eq_png(name, tex, fontsize=30):
    """Render an equation to a crisp PNG with matplotlib mathtext, in
    the dissertation's notation (Fiyin: 'copy the equation from the
    actual dissertation if it's impossible to write it properly')."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FIGS_EXTRA.mkdir(exist_ok=True)
    path = FIGS_EXTRA / f"{name}.png"
    fig = plt.figure(figsize=(10, 1.2))
    fig.text(0.5, 0.5, tex, fontsize=fontsize, color="#1F3864",
             ha="center", va="center")
    fig.savefig(path, dpi=300, transparent=True,
                bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    return path

TOTAL_SLIDES = 17


def _mini_table(s, left_in, top_in, w_in, rows, highlight_row=None,
                col_widths=None, left_align_all=False, size_pt=8.5):
    """Small data table inside a panel. rows[0] is the header."""
    nrows, ncols = len(rows), len(rows[0])
    gf = s.shapes.add_table(nrows, ncols,
                            Inches(left_in), Inches(top_in),
                            Inches(w_in), Inches(0.235 * nrows))
    tbl = gf.table
    if col_widths:
        for c, cw in enumerate(col_widths):
            tbl.columns[c].width = Inches(cw)
    for r, row in enumerate(rows):
        tbl.rows[r].height = Inches(0.235)
        for c, val in enumerate(row):
            cell = tbl.cell(r, c)
            cell.margin_top = cell.margin_bottom = Emu(12700)
            cell.margin_left = cell.margin_right = Emu(38100)
            tf = cell.text_frame
            tf.word_wrap = False
            p = tf.paragraphs[0]
            if left_align_all:
                p.alignment = PP_ALIGN.LEFT
            else:
                p.alignment = PP_ALIGN.LEFT if c == 0 else PP_ALIGN.RIGHT
            run = p.add_run()
            run.text = str(val)
            f = run.font
            f.name = "Calibri"
            f.size = Pt(size_pt)
            f.bold = (r == 0) or (r == highlight_row)
    return gf


def slide_validation(prs):
    """NEW slide 6 — what was decided on validation, and what those
    experiments found. All numbers from ch5 (Tables 5.1 setup, 5.5)."""
    s = blank_slide(prs)
    add_title(s, "The validation experiments",
              "trade size · transaction fee · risk coefficient — "
              "decided once on validation, then frozen")

    ORANGE = RGBColor(0xE0, 0x7A, 0x1F)
    GREEN = RGBColor(0x2A, 0x9D, 0x3D)

    _panel(s, 0.45, 1.35, 4.10, 5.15, "Trade size", NAVY)
    _points_textbox(s, [
        ("Each Buy commits a fixed slice of the starting money.",
         " We tested fractions from 5% up to 50% of capital on the "
         "validation data, and watched how large a position each one "
         "could actually build within a month."),
        ("A quarter of the capital won.",
         " It was the smallest slice that could reach 90% of "
         "buy-and-hold's position. Smaller slices simply cannot get "
         "there in 21 days; bigger ones give up fine control."),
    ], 0.73, 1.95, 3.54, 2.5, size_pt=10.5, space_after=7)
    _mini_table(s, 0.73, 4.65, 3.54, [
        ["Slice", "Days to full", "Max exposure", "90%?"],
        ["0.05\u00b7C\u2080", "20", "0.452", "no"],
        ["0.10\u00b7C\u2080", "10", "0.691", "no"],
        ["0.25\u00b7C\u2080", "4", "0.834", "yes \u2713"],
        ["0.50\u00b7C\u2080", "2", "0.882", "yes"],
        ["Buy-&-hold", "\u2014", "0.906", "\u2014"],
    ], highlight_row=3)

    _panel(s, 4.62, 1.35, 4.10, 5.15, "Transaction fee", ORANGE)
    _points_textbox(s, [
        ("Every trade costs something.",
         " We charge 0.05% of each trade — five basis points — the "
         "same for every agent and every benchmark, so nobody trades "
         "for free."),
        ("Then we stress-tested that choice.",
         " Sweeping the fee up to ten times the assumption, Deep "
         "Q-learning kept trading and kept earning. Buy-and-hold "
         "turns negative from 10 basis points."),
    ], 4.90, 1.95, 3.54, 2.5, size_pt=10.5, space_after=7)
    _mini_table(s, 4.90, 4.65, 3.54, [
        ["Fee", "Buy-&-hold", "Deep Q", "DQ trades"],
        ["0 bps", "+$15.67", "+$172.25", "5.5"],
        ["5 bps", "+$5.66", "+$159.63", "4.9"],
        ["10 bps", "\u2212$4.34", "+$152.30", "4.8"],
        ["25 bps", "\u2212$34.29", "+$142.24", "4.1"],
        ["50 bps", "\u2212$83.99", "+$95.27", "3.7"],
    ], highlight_row=2)

    _panel(s, 8.79, 1.35, 4.10, 5.15, "Risk coefficient \u03bb", GREEN)
    _points_textbox(s, [
        ("\u03bb sets how much losing along the way hurts.",
         " The risk-aware reward takes the month's profit and "
         "subtracts \u03bb times the worst dip the portfolio suffered "
         "on the way."),
        ("We raised it until the agent stopped trading.",
         " Past \u03bb = 0.25 the penalty outweighs any possible gain "
         "and the agent barely acts. So we kept the largest value "
         "where it still trades properly: \u03bb = 0.25."),
    ], 9.07, 1.95, 3.54, 2.5, size_pt=10.5, space_after=7)
    _mini_table(s, 9.07, 4.65, 3.54, [
        ["\u03bb", "Profit", "Exposure", "Passes rule?"],
        ["0", "+$158.29", "0.598", "\u2014"],
        ["0.05", "+$166.58", "0.575", "yes"],
        ["0.25", "+$99.13", "0.338", "yes \u2713"],
        ["0.50", "+$15.65", "0.039", "no"],
        ["2.00", "\u2212$0.04", "0.000", "no"],
    ], highlight_row=3)

    add_caption(s, "The rule of the study: every knob is decided on "
                   "validation data, then frozen. The test sets are "
                   "opened once, at the end.")
    add_slide_number(s, 6, total=TOTAL_SLIDES)
    add_notes(s, """
Say (about 75 seconds):

"Before the first study's results, a word on what validation was
actually FOR. Three things were decided there, then frozen.

The trade size. Each Buy commits a fixed fraction of starting
capital. We swept that fraction on the 2023 validation months, and
a quarter of capital was the smallest size that reached at least
90% of buy-and-hold's exposure --- big enough for decisions to
count, small enough to keep cash in hand for falling months.

The transaction fee: 0.05% per trade, five basis points, the same
for every agent and every benchmark. Later, in the robustness
sweep, we pushed that to ten times the level and Deep Q-learning
stayed profitable.

And the risk coefficient, lambda. The risk-aware reward subtracts
lambda times drawdown from the wealth change. We swept lambda from
zero to two on the simulated validation episodes. Past 0.25 the
agent simply stops trading --- the penalty outweighs any gain. Our
rule kept a floor on wealth and exposure, and the largest lambda
that passed was 0.25, which cut drawdown by 46.6%.

None of these numbers ever touched the test sets. That is the rule
of the whole study: decide on validation, freeze, then open the
test data once, at the end."

ANTICIPATED QUESTION (Nguyen): "Why lambda = 0.25 and not the
best-return value?" Answer: the sweep selects by a RULE, not by
return --- the largest penalty that still keeps the agent trading
(wealth >= +$79.14, exposure >= 0.299). Picking by return would
have chosen 0.05 and barely changed behaviour.
""")


def slide_state(prs):
    """NEW slide 4 — the state: ten numbers at most. Facts: ch3 Table 3.1
    and the risk-aware state extension (s_risk in R^10)."""
    s = blank_slide(prs)
    add_title(s, "What the agent sees — the state",
              "nine numbers in the main studies; a tenth — drawdown — "
              "added for the risk experiments")

    _mini_table(s, 0.45, 1.45, 8.00, [
        ["#", "Feature", "Formula", "What it tells the agent"],
        ["1", "Daily return", "(P\u209c \u2212 P\u209c\u208b\u2081) / P\u209c\u208b\u2081",
         "the most recent price move"],
        ["2", "Momentum (5-day)", "(P\u209c \u2212 P\u209c\u208b\u2085) / P\u209c\u208b\u2085",
         "is that move part of a short trend?"],
        ["3", "Volatility (5-day)", "\u03c3(last 5 daily returns)",
         "how much prices vary lately"],
        ["4", "Gap to 20-day average", "P\u209c / MA\u2082\u2080 \u2212 1",
         "how far price sits from its average"],
        ["5", "True range", "ATR\u2081\u2084 / P\u209c",
         "the size of recent daily swings"],
        ["6", "Cash fraction", "C\u209c / C\u2080",
         "how much cash is left to buy with"],
        ["7", "Position value", "h\u209c P\u209c / C\u2080",
         "how big the open position is"],
        ["8", "Open profit/loss", "(P\u209c \u2212 entry price) / entry price",
         "is the position in profit or loss?"],
        ["9", "Time left", "t / T",
         "how far through the month we are"],
        ["10", "Drawdown  dd\u209c",
         "(W\u1d56\u1d49\u1d43\u1d4f \u2212 W\u209c) / W\u1d56\u1d49\u1d43\u1d4f",
         "risk only: the worst dip so far"],
    ], highlight_row=10, col_widths=[0.42, 1.80, 2.60, 3.18],
        left_align_all=True, size_pt=9.5)

    ORANGE = RGBColor(0xE0, 0x7A, 0x1F)
    _panel(s, 8.70, 1.45, 4.20, 5.05, "How it was chosen", ORANGE)
    _points_textbox(s, [
        ("Nothing redundant.",
         " A feature made the list only if its information cannot be "
         "recovered from the others — no pile of indicators."),
        ("Nothing from the future.",
         " Every feature is computed from information available at "
         "decision time. The agent never peeks ahead."),
        ("Drawdown kept separate on purpose.",
         " Features 1–9 are the base state. Drawdown joins as a tenth "
         "only in the risk experiments — so we can test whether "
         "seeing loss history actually changes behaviour."),
    ], 8.98, 2.05, 3.64, 4.3, size_pt=10.5, space_after=9)

    add_caption(s, "This is everything the agent knows when it "
                   "decides: five numbers about the market, four about "
                   "its own money — and nothing from the future. "
                   "(P = price, C = cash, h = shares held, W = wealth, "
                   "t = day in month.)")
    add_slide_number(s, 4, total=TOTAL_SLIDES)
    add_notes(s, """
Say (about 60 seconds):

"Before the algorithms — what does the agent actually see each day?

At most ten numbers. Five describe the market: the day's return, a
five-day momentum, a five-day volatility, the gap between the price
and its twenty-day average, and the true range --- the size of
recent daily swings. Four describe the agent's own situation: how
much cash is left, how big its position is, whether that position is
in profit, and how far through the month we are.

Two rules shaped this list. Nothing redundant --- a feature joins
only if the others cannot recover its information. And nothing from
the future --- everything is computable at decision time.

The tenth number, drawdown --- the worst dip the portfolio has
suffered this month --- is deliberately kept OUT of the base state.
It joins only in the risk experiments, so we can measure whether
seeing loss history changes behaviour at all. That experiment comes
later."

Do NOT read the formula column aloud --- it is there so the
examiners can check each feature is well-defined. If asked, the
legend is in the caption: P price, C cash, h shares held, W wealth.

ANTICIPATED QUESTION (Nguyen): "What is the formulation of
drawdown?" Answer: dd_t = (peak wealth so far minus current wealth)
divided by peak wealth so far, within the episode — a number in
[0, 1], 0 when at a new high.
""")


def slide_masking(prs):
    """NEW slide 4 — what action masking is and how each algorithm
    applies it. Facts from ch3 Sections 3.4 and the algorithm chapters."""
    s = blank_slide(prs)
    add_title(s, "Action masking — only possible actions allowed",
              "the environment tells the agent, each day, which of its "
              "three actions it can actually execute")

    ORANGE = RGBColor(0xE0, 0x7A, 0x1F)

    _panel(s, 0.45, 1.35, 6.05, 5.15, "What it is", NAVY)
    _points_textbox(s, [
        ("The problem it solves.",
         " An agent cannot buy when it has no cash left, and it "
         "cannot sell shares it does not hold. Something has to stop "
         "it from picking impossible actions."),
        ("The mask is three switches.",
         " Each day the environment hands the agent "
         "M\u209c = [hold, buy, sell], each switch 1 (available) or "
         "0 (not). Available cash sets the buy switch, current "
         "holdings set the sell switch, and hold is always on."),
        ("What it is NOT.",
         " It carries no market information — it is built from the "
         "portfolio the agent already knows about. It is a constraint "
         "on the agent, not a hint."),
    ], 0.73, 1.95, 5.49, 4.3, size_pt=11.5, space_after=10)

    _panel(s, 6.85, 1.35, 6.05, 5.15, "How each algorithm uses it",
           ORANGE)
    _points_textbox(s, [
        ("REINFORCE and PPO.",
         " Unavailable actions are removed before the softmax, so "
         "their probabilities are exactly zero — the agent cannot "
         "even sample them. PPO uses MaskablePPO from sb3-contrib."),
        ("Deep Q-learning — masked twice.",
         " When acting, it picks the best LEGAL action. And inside "
         "the learning target, the maximum runs over legal actions "
         "only, so no value is ever assigned to an impossible move."),
        ("Verified, not assumed.",
         " The 21-test suite includes mask-timing tests: if an "
         "illegal action ever slips through, the build fails."),
    ], 7.13, 1.95, 5.49, 4.3, size_pt=11.5, space_after=10)

    # caption wording set by Fiyin in PowerPoint (13 Sep)
    add_caption(s, "Why it matters: every decision in every result was "
                   "genuinely available at the moment it was made.")
    add_slide_number(s, 4, total=TOTAL_SLIDES)
    add_notes(s, """
Say (about 60 seconds):

"One piece of plumbing worth showing, because every result depends
on it: action masking.

The agent cannot buy when it has no cash left, and it cannot sell
shares it does not hold. So each day the environment hands it three
switches --- hold, buy, sell --- each one on or off. Cash decides
the buy switch, holdings decide the sell switch, and hold is always
on. The mask carries no market information; it is built from the
portfolio the agent already knows. It is a constraint, not a hint.

Each algorithm respects it in its own way. For REINFORCE and PPO,
unavailable actions are removed before the softmax, so their
probabilities are exactly zero. For Deep Q-learning it applies
twice: the agent picks the best legal action when acting, and
inside the learning target the maximum runs over legal actions
only --- no value is ever assigned to an impossible move.

And it is verified: the test suite includes mask-timing tests that
fail if an illegal action ever slips through."

ANTICIPATED QUESTION (Nguyen): "Is the mask part of the state?"
Answer: no — it is derived from the portfolio part of the state and
acts as a constraint on action selection; it adds no information
the state does not already contain.
""")


def slide_updates(prs):
    """NEW slide 6 — the update rules / loss functions, intuition first
    (Nguyen: equations must be ready; one line of plain words each)."""
    s = blank_slide(prs)
    add_title(s, "How each algorithm learns — the update rules",
              "the intuition in one line, then the objective each "
              "method optimises")

    def _algo_block(top_in, head, body, eq_name, eq_tex, eq_h=None):
        _points_textbox(s, [(head, body)], 1.0, top_in, 11.3, 0.95,
                        size_pt=13, space_after=2)
        # equation as a typeset image, exactly as in the dissertation
        # (Fiyin, 13 Sep: the unicode text versions were illegible).
        # Uniform scale so all three equations show the same font size.
        path = _eq_png(eq_name, eq_tex)
        from PIL import Image
        with Image.open(path) as im:
            px_w, px_h = im.size
        scale = 0.52                       # 300 dpi render shown at 52%
        w_in, h_in = px_w / 300 * scale, px_h / 300 * scale
        if w_in > 11.3:
            h_in *= 11.3 / w_in
            w_in = 11.3
        pic = s.shapes.add_picture(str(path), Inches(1.0),
                                   Inches(top_in + 0.86),
                                   width=Inches(w_in),
                                   height=Inches(h_in))
        pic.left = int((Inches(SLIDE_W_IN) - pic.width) / 2)

    _algo_block(
        1.50,
        "Deep Q-learning learns a score for each action, then picks "
        "the best.",
        " Q(s, a) is its estimate of how much money action a leads to "
        "from state s. Training shrinks the gap between that estimate "
        "and a better-informed one, y\u209c: the reward actually "
        "received plus the best legal score in tomorrow's state. The "
        "frozen copy Q\u207b holds the target still while it learns.",
        "eq_dqn",
        r"$L(\phi) = \mathbb{E}\left[\,(y_t - Q_\phi(s_t, a_t))^2\right]"
        r"\qquad y_t = r_t + \gamma\,(1-d_t)"
        r"\max_{a'\,\in\,\mathcal{A}_{\mathrm{legal}}}"
        r" Q_{\phi^-}(s_{t+1}, a')$")

    _algo_block(
        3.30,
        "REINFORCE repeats whatever paid off.",
        " After a month of trading it looks back at each action and "
        "asks one question: how much money followed it? That amount "
        "is G\u209c. Actions followed by more money than usual become "
        "more likely next time; actions followed by less become less "
        "likely. That is the whole rule.",
        "eq_reinforce",
        r"$\nabla_\theta J(\theta) = \mathbb{E}\left[\,"
        r"\sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t \mid s_t)"
        r"\; G_t\right]$",
        eq_h=0.62)

    _algo_block(
        5.15,
        "PPO is REINFORCE with a safety limit on each step.",
        " \u03c1\u209c measures how much more likely the new policy "
        "makes an action than the old one did. The clip caps that "
        "change at a small fraction \u03b5 — so one month of data can "
        "never drag the policy far from the one that collected it.",
        "eq_ppo",
        r"$L^{\mathrm{CLIP}}(\theta) = \mathbb{E}_t\left[\,"
        r"\min\left(\rho_t(\theta)\,A_t,\ "
        r"\mathrm{clip}(\rho_t(\theta),\,1-\varepsilon,\,1+\varepsilon)"
        r"\,A_t\right)\right]"
        r"\qquad \rho_t(\theta) = \pi_\theta(a_t \mid s_t)\,/\,"
        r"\pi_{\theta_{\mathrm{old}}}(a_t \mid s_t)$")

    add_caption(s, "How to say it: Deep Q learns how good each action "
                   "is and picks the best; REINFORCE repeats what "
                   "paid; PPO does the same with a cap on each change. "
                   "The equations are the dissertation's, for "
                   "reference.")
    add_slide_number(s, 6, total=TOTAL_SLIDES)
    add_notes(s, """
Say (about 60 seconds — read the BOLD lines, never the symbols):

"For the examiners' reference, the update rules themselves. In
plain words:

Deep Q-learning learns a score for each action --- how much money
it expects that action to lead to --- and picks the best. Training
shrinks the gap between its current score and a better-informed one
computed a day later: the reward it actually received, plus the
best legal score in the next state. The frozen copy of the network
holds that target still while it learns.

REINFORCE is simpler: after each month it looks back at every
action and asks how much money followed it. Actions followed by
more money than usual become more likely; the rest less likely.

And PPO is that same rule with a safety limit: no single update may
change how likely any action is by more than a small fraction.

That difference --- learned values with forced exploration, versus
probabilities that can quietly collapse --- is exactly why their
behaviour splits in the results."

IF ASKED about G_t: it is the reward-to-go --- everything earned
from that action to the end of the month. An action is credited
only with what came after it, not before.

IF ASKED about the baseline / variance: in practice the code
subtracts the average return from G_t before the update. That
removes noise (one lucky month cannot swing the policy) without
changing the expected direction, because E[grad log pi] = 0.

IF ASKED for a derivation, do it on the REINFORCE gradient:
log-derivative trick, then a Monte-Carlo average over sampled
episodes.
""")


def slide_risk(prs):
    """NEW slide (after the results flip) — the two risk experiments:
    drawdown state feature, then the risk-aware reward. Facts: ch5
    lines ~516-534 and ~648-668."""
    s = blank_slide(prs)
    add_title(s, "The risk experiments — a new input, then a new "
                 "objective",
              "adding drawdown to the state · switching to the "
              "risk-aware reward (\u03bb = 0.25)")

    GREEN = RGBColor(0x2A, 0x9D, 0x3D)

    # the story first: WHY these experiments exist at all
    _points_textbox(s, [
        ("Why these experiments exist.",
         " The main agent is paid for profit alone, so it has no "
         "reason to care how rough the ride is — a real investor "
         "does. The question: can the same agent learn to protect "
         "itself from losses? We tested it in two steps, changing "
         "one thing at a time."),
    ], 0.55, 1.42, 12.30, 0.85, size_pt=12, space_after=0)

    _panel(s, 0.45, 2.45, 6.05, 3.80, "Step 1 — let it SEE its losses",
           NAVY)
    _points_textbox(s, [
        ("What we changed.",
         " One new input: drawdown — how far the portfolio has "
         "fallen from its best point this month. If the agent merely "
         "lacked information, this alone should change how it "
         "trades."),
        ("What happened: almost nothing.",
         " Mean profit +$126.94 → +$131.31 per episode — a $4 move "
         "against a seed spread near $7. The agent could now see its "
         "losses, but nothing paid it to act on them."),
    ], 0.73, 3.10, 5.49, 3.0, size_pt=11, space_after=9)

    _panel(s, 6.85, 2.45, 6.05, 3.80, "Step 2 — make losses HURT",
           GREEN)
    _points_textbox(s, [
        ("What we changed.",
         " The pay rule. Reward = month's profit \u2212 \u03bb \u00d7 "
         "that same drawdown, with \u03bb = 0.25 picked on "
         "validation. Losing along the way now costs the agent "
         "directly."),
        ("What happened: caution, exactly where needed.",
         " In falling markets it loses 73% less (\u2212$87.28 → "
         "\u2212$23.26). In rising markets it keeps 87% of its gains "
         "(+$411.01 vs +$470.01)."),
        ("At almost no cost.",
         " Mean profit moves 1.5%; the Sharpe ratio — return earned "
         "per unit of risk taken — rises from 0.285 to 0.332."),
    ], 7.13, 3.10, 5.49, 3.0, size_pt=11, space_after=9)

    add_caption(s, "The lesson: information only helps when the "
                   "objective gives the agent a reason to use it.")
    add_slide_number(s, 14, total=TOTAL_SLIDES)
    add_notes(s, """
Say (about 80 seconds):

"Why do these experiments exist at all? Because the main agent is
paid for profit alone. It has no reason to care how rough the ride
is --- but a real investor does. So the question was: can the same
agent learn to protect itself from losses? We tested it in two
steps, changing one thing at a time.

Step one: let it SEE its losses. We gave it one new input ---
drawdown, how far the portfolio has fallen from its best point this
month. If the agent merely lacked information, this alone should
change how it trades. It didn't. The mean moved four dollars
against a seed spread of seven. It could see its losses; nothing
paid it to act on them.

Step two: make losses hurt. We changed the pay rule --- reward is
now the month's profit minus lambda times that same drawdown. Now
behaviour changed exactly where it should. In falling markets the
agent loses seventy-three percent less. In rising markets it keeps
87 percent of its gains. And the cost was almost nothing: mean
profit moved one and a half percent, and the Sharpe ratio ---
return per unit of risk --- rose from 0.285 to 0.332.

The lesson in one line: information only helps when the objective
gives the agent a reason to use it."

ANTICIPATED QUESTION (Nguyen asked about drawdown repeatedly):
"Why keep the feature and the reward separate?" Answer: to separate
the two effects --- seeing losses versus being paid to avoid them.
Testing them together would confound which one changed behaviour.
""")


def slide_demo(prs):
    """Slide 17 — the QR demo. Offered ONLY at the very end, after the
    examiners have finished their questions (Fiyin, 13 Sep)."""
    s = blank_slide(prs)
    add_title(s, "See it run — a live demo on your phone",
              "the transfer test as an animation: the same 180 months, "
              "drawn month by month")

    # left panel: the QR code and address
    _panel(s, 0.45, 1.45, 4.30, 5.05, "Scan to open", NAVY)
    s.shapes.add_picture(str(FIGS_EXTRA / "qr_demo.png"),
                         Inches(1.05), Inches(2.15),
                         width=Inches(3.10), height=Inches(3.10))
    tb = s.shapes.add_textbox(Inches(0.73), Inches(5.45),
                              Inches(3.74), Inches(0.8))
    p = tb.text_frame.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    r = p.add_run()
    r.text = "thefinix13.github.io/viva-agent-demo"
    r.font.name = "Calibri"
    r.font.size = Pt(12)
    r.font.bold = True
    r.font.color.rgb = NAVY

    # middle: what the phone shows
    s.shapes.add_picture(str(FIGS_EXTRA / "demo_phone.png"),
                         Inches(5.05), Inches(1.45), height=Inches(5.05))

    # right panel: what they will see
    GREEN = RGBColor(0x2A, 0x9D, 0x3D)
    _panel(s, 9.15, 1.45, 3.75, 5.05, "What it shows", GREEN)
    _points_textbox(s, [
        ("The transfer test, animated.",
         " The cumulative-wealth lines draw themselves month by "
         "month across the 180 unseen months."),
        ("October 2008, as it happens.",
         " Buy-and-hold plunges through the crash band; the Deep Q "
         "line stays flat — it holds zero shares."),
        ("The real numbers.",
         " Every line ends on its Table 5.3 value — this is the "
         "evaluated result, not a mock-up."),
    ], 9.43, 2.05, 3.19, 4.3, size_pt=10.5, space_after=9)

    add_caption(s, "A demo of the SPY evaluated results from the "
                   "dissertation")
    add_slide_number(s, 17, total=TOTAL_SLIDES)
    add_notes(s, """
OFFER THIS ONLY AT THE VERY END — after the examiners confirm they
have no more questions. Never mid-presentation.

Say:

"Before we finish — if you would like to see the transfer test run,
I built a small demo. If you scan this code, or open the link I've
just put in the chat, you can watch the same 180 months play out on
your phone: buy-and-hold falling through October 2008 while the
Deep Q agent's line stays flat."

Then PASTE THE LINK IN THE TEAMS CHAT:
https://thefinix13.github.io/viva-agent-demo/

IF ASKED whether this is part of the dissertation: "The numbers are
the dissertation's Table 5.3 transfer result exactly — the page is
just an animated view of them. Anything beyond SPY would be
post-submission engineering, not an evaluated result."
""")


def _renumber(prs):
    """Rewrite every 'n / total' stamp to n / TOTAL_SLIDES."""
    target_left = Inches(12.5)
    target_top = Inches(7.05)
    for idx, slide in enumerate(prs.slides, start=1):
        for shape in slide.shapes:
            if not shape.has_text_frame:
                continue
            if shape.left == target_left and shape.top == target_top:
                for para in shape.text_frame.paragraphs:
                    for run in para.runs:
                        if "/" in run.text:
                            run.text = f"{idx} / {TOTAL_SLIDES}"
                break


def main():
    prs = Presentation()
    prs.slide_width = Inches(SLIDE_W_IN)
    prs.slide_height = Inches(SLIDE_H_IN)

    slide1_title(prs)
    slide2_decision(prs)
    slide3_rl(prs)
    slide_state(prs)           # NEW 4 — the ten-number state
    slide_masking(prs)         # NEW 5 — action masking
    slide4_algorithms(prs)     # -> 6
    slide_updates(prs)         # NEW 7 — update rules / losses
    slide5_rules(prs)          # -> 8
    slide_validation(prs)      # NEW 9 — validation experiments
    slide6_first_result(prs)   # -> 10
    slide8_diagnosis(prs)      # -> 11
    slide9_simulator(prs)      # -> 12
    slide10_flip(prs)          # -> 13
    slide_risk(prs)            # NEW 14 — drawdown state + risk reward
    slide11_stress(prs)        # -> 15
    slide12_closing(prs)       # -> 16
    slide_demo(prs)            # NEW 17 — QR phone demo (end of Q&A only)

    _renumber(prs)

    prs.save(OUT)
    print(f"Wrote {OUT}  ({OUT.stat().st_size / 1024:.0f} KB, "
          f"{len(prs.slides)} slides)")


if __name__ == "__main__":
    main()
