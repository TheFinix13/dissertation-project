"""Build the viva PowerPoint deck (editable .pptx).

Design brief: examiners are ML researchers who do not know reinforcement
learning or finance. Every slide has ONE picture that explains itself.
The visible text is a single caption line. All talking points sit in the
speaker notes.

Output: latex/viva/viva_deck.pptx  (16:9, 12 slides)
"""
from __future__ import annotations

from pathlib import Path

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR

ROOT = Path(__file__).resolve().parents[1]
FIGS = ROOT / "latex" / "viva" / "figs"
LOGO = ROOT / "latex" / "dissertation" / "figs" / "surrey_logo.png"
AGENT_LOOP = ROOT / "latex" / "dissertation" / "media" / "media" / "image1_ch3.png"
PER_SEED_SCATTER = ROOT / "latex" / "dissertation" / "media4" / "media" / "image4.png"
OUT = ROOT / "latex" / "viva" / "viva_deck.pptx"

# 16:9 dimensions
SLIDE_W_IN = 13.333
SLIDE_H_IN = 7.5

NAVY = RGBColor(0x1F, 0x33, 0x5A)
INK = RGBColor(0x1E, 0x1E, 0x1E)
GREY = RGBColor(0x66, 0x66, 0x66)
LIGHTGREY = RGBColor(0xE8, 0xE4, 0xDA)
ACCENT = RGBColor(0xE0, 0x7A, 0x1F)  # DQN orange, our thread colour


def add_title(slide, text: str, subtitle: str | None = None):
    """Consistent title header at the top of every slide."""
    box = slide.shapes.add_textbox(Inches(0.55), Inches(0.28),
                                   Inches(12.2), Inches(0.9))
    tf = box.text_frame
    tf.margin_top = Emu(0)
    tf.margin_bottom = Emu(0)
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.LEFT
    r = p.add_run()
    r.text = text
    r.font.name = "Calibri"
    r.font.size = Pt(30)
    r.font.bold = True
    r.font.color.rgb = NAVY

    if subtitle:
        p2 = tf.add_paragraph()
        p2.alignment = PP_ALIGN.LEFT
        r2 = p2.add_run()
        r2.text = subtitle
        r2.font.name = "Calibri"
        r2.font.size = Pt(15)
        r2.font.color.rgb = GREY
        r2.font.italic = True

    # accent bar
    bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE,
                                 Inches(0.55), Inches(1.30),
                                 Inches(0.5), Inches(0.06))
    bar.fill.solid()
    bar.fill.fore_color.rgb = ACCENT
    bar.line.fill.background()


def add_caption(slide, text: str):
    """Single caption line at the bottom of a slide."""
    box = slide.shapes.add_textbox(Inches(0.55), Inches(6.85),
                                   Inches(12.2), Inches(0.45))
    tf = box.text_frame
    tf.margin_top = Emu(0)
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.LEFT
    r = p.add_run()
    r.text = text
    r.font.name = "Calibri"
    r.font.size = Pt(14)
    r.font.italic = True
    r.font.color.rgb = GREY


def add_slide_number(slide, n: int, total: int = 11):
    box = slide.shapes.add_textbox(Inches(12.5), Inches(7.05),
                                   Inches(0.75), Inches(0.35))
    tf = box.text_frame
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.RIGHT
    r = p.add_run()
    r.text = f"{n} / {total}"
    r.font.name = "Calibri"
    r.font.size = Pt(10)
    r.font.color.rgb = GREY


def add_notes(slide, text: str):
    slide.notes_slide.notes_text_frame.text = text.strip()


def centred_image(slide, path: Path, max_w_in: float, max_h_in: float,
                  top_in: float, centre_x_in: float = SLIDE_W_IN / 2):
    """Add an image, capped by max_w and max_h, horizontally centred."""
    from PIL import Image
    with Image.open(path) as im:
        w, h = im.size
    aspect = w / h
    if max_w_in / aspect <= max_h_in:
        w_in = max_w_in
        h_in = w_in / aspect
    else:
        h_in = max_h_in
        w_in = h_in * aspect
    left = centre_x_in - w_in / 2
    slide.shapes.add_picture(str(path), Inches(left), Inches(top_in),
                             width=Inches(w_in), height=Inches(h_in))


def blank_slide(prs) -> "Slide":
    return prs.slides.add_slide(prs.slide_layouts[6])  # Blank


# ---- individual slides --------------------------------------------------

def slide1_title(prs):
    """Classic centred academic title page (matches the beamer layout)."""
    s = blank_slide(prs)

    # title banner — light grey band behind title + subtitle, centred
    band = s.shapes.add_shape(MSO_SHAPE.RECTANGLE,
                              Inches(1.5), Inches(0.85),
                              Inches(SLIDE_W_IN - 3.0), Inches(1.75))
    band.fill.solid()
    band.fill.fore_color.rgb = RGBColor(0xEC, 0xEC, 0xEC)
    band.line.fill.background()
    tf = band.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    r = p.add_run()
    r.text = "Evaluating Reinforcement Learning Algorithms\nfor Portfolio Trading"
    r.font.name = "Calibri"
    r.font.size = Pt(30)
    r.font.bold = True
    r.font.color.rgb = INK
    p2 = tf.add_paragraph()
    p2.alignment = PP_ALIGN.CENTER
    p2.space_before = Pt(8)
    r2 = p2.add_run()
    r2.text = "A Reinforcement Learning Study on the S&P 500 ETF (SPY)"
    r2.font.name = "Calibri"
    r2.font.size = Pt(18)
    r2.font.color.rgb = INK

    # centred author block
    abox = s.shapes.add_textbox(Inches(1.5), Inches(3.15),
                                Inches(SLIDE_W_IN - 3.0), Inches(2.4))
    tf = abox.text_frame
    tf.word_wrap = True
    rows = [
        ("Fiyinfoluwa Akano", 22, True, INK, 0),
        ("URN 6962514  ·  MSc Artificial Intelligence", 14, False, GREY, 2),
        ("Supervisor: Dr Cuong Nguyen", 15, False, INK, 14),
        ("University of Surrey", 15, False, INK, 2),
        ("September 2026", 14, False, GREY, 14),
    ]
    for i, (text, size, bold, colour, space_before) in enumerate(rows):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = PP_ALIGN.CENTER
        p.space_before = Pt(space_before)
        r = p.add_run()
        r.text = text
        r.font.name = "Calibri"
        r.font.size = Pt(size)
        r.font.bold = bold
        r.font.color.rgb = colour

    # Surrey logo — centred at the bottom, like the beamer title page
    if LOGO.exists():
        from PIL import Image
        with Image.open(LOGO) as im:
            lw, lh = im.size
        aspect = lw / lh
        target_h = 0.9
        target_w = target_h * aspect
        max_w = 2.6
        if target_w > max_w:
            target_w = max_w
            target_h = target_w / aspect
        s.shapes.add_picture(
            str(LOGO),
            Inches((SLIDE_W_IN - target_w) / 2),
            Inches(6.15),
            width=Inches(target_w), height=Inches(target_h),
        )

    add_notes(s, """
Opening — your own words from rehearsal, spoken slowly.

"Good morning, professors. Today I'll be presenting my
dissertation, which is an evaluation of reinforcement learning
algorithms for portfolio trading --- the applicable field here
being finance: trading a stock-market fund.

I'll take you through it in eleven slides: first the problem
and how reinforcement learning fits it, then the two studies
and what they showed."

Do NOT introduce yourself as an RL expert. The second examiner
is from the Institute for Communication Systems and may know
neither RL nor finance. Keep every explanation plain.
""")


def slide2_decision(prs):
    s = blank_slide(prs)
    # Title and subtitle wording set by Fiyin in PowerPoint (11 Sep)
    add_title(s, "The Problem (Main Research Question)",
              "An investor holding a stock can decide each day whether "
              "to: buy, do nothing, or sell.")
    centred_image(s, FIGS / "slide2_decision.png",
                  max_w_in=10.8, max_h_in=4.1, top_in=1.55)

    # Main research question, in a highlighted box (Fiyin's wording)
    qbox = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,
                              Inches(0.9), Inches(5.95),
                              Inches(11.5), Inches(0.95))
    qbox.fill.solid()
    qbox.fill.fore_color.rgb = RGBColor(0xFB, 0xF0, 0xE1)
    qbox.line.color.rgb = ACCENT
    qbox.line.width = Pt(1.5)
    tf = qbox.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf.margin_left = Inches(0.25)
    tf.margin_right = Inches(0.25)
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.LEFT
    rh = p.add_run()
    rh.text = "Main Research Question — "
    rh.font.name = "Calibri"
    rh.font.size = Pt(15)
    rh.font.bold = True
    rh.font.color.rgb = NAVY
    rb = p.add_run()
    rb.text = ("Can reinforcement learning make these decisions better "
               "than simply buying and holding an asset long term?")
    rb.font.name = "Calibri"
    rb.font.size = Pt(15)
    rb.font.color.rgb = INK

    add_slide_number(s, 2)
    add_notes(s, """
Your own words from rehearsal — keep this flow:

"So what exactly is the problem, or the research question?

A regular person who invests in stocks decides on a day-to-day
basis whether to buy more shares, leave the position and do
nothing, or sell --- depending on a number of reasons. Maybe
the market is declining and they want to protect their profit,
or they simply need cash at hand for an emergency.

This leads to the main research question: can a reinforcement
learning algorithm learn a strategy that is overall better than
simply buying and holding the asset for the long term?

Performance here can be measured in two ways. First, how much
more profitable the learned strategy is compared to just buying
and holding. Second, whether the learned policy reduces its
exposure --- holds fewer shares --- during a market decline,
protecting wealth in exactly the periods that matter."

Timing: 60 seconds. Questions 2 (behaviour) and 3 (data) are
NOT on this slide — they arrive naturally on slides 7 and 8.
""")


def slide3_rl(prs):
    s = blank_slide(prs)
    add_title(s, "The Trading Environment (MDP Formulation)",
              "how reinforcement learning works — state, action, reward")
    # The dissertation's loop layout, redrawn with plain-language text
    centred_image(s, FIGS / "slide3_env_loop.png",
                  max_w_in=11.6, max_h_in=3.85, top_in=1.55)

    # Definition strip — the four terms, one line each
    tb = s.shapes.add_textbox(Inches(0.9), Inches(5.55),
                              Inches(11.5), Inches(1.35))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_top = Emu(0)
    tf.margin_bottom = Emu(0)
    defs = [
        ("State.",
         " What the agent sees right now — market information, "
         "portfolio details, and time within the episode."),
        ("Action.",
         " What the agent does with it — Buy, Hold, or Sell."),
        ("Reward.",
         " The consequence — the change in portfolio wealth after fees. "
         "The agent is never told the 'correct' action, only this outcome."),
        ("Episode.",
         " One month of trading (18–23 days). The loop above runs once "
         "per trading day."),
    ]
    for i, (head, body) in enumerate(defs):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = PP_ALIGN.LEFT
        p.space_after = Pt(2)
        rh = p.add_run()
        rh.text = head
        rh.font.name = "Calibri"
        rh.font.size = Pt(12)
        rh.font.bold = True
        rh.font.color.rgb = NAVY
        rb = p.add_run()
        rb.text = body
        rb.font.name = "Calibri"
        rb.font.size = Pt(12)
        rb.font.color.rgb = INK

    add_slide_number(s, 3)
    add_notes(s, """
Your own words from rehearsal — keep this flow:

"To understand the trading environment, we first need to
understand how reinforcement learning works.

Unlike supervised learning, where a model is given true labels
to predict from a data set, reinforcement learning learns by
trial and error --- from the rewards obtained by taking an
action in the current state.

Let me break that down with the diagram. We have an agent and
an environment. The agent is a small neural network --- in our
case a policy network pi, or a Q-network Q --- which uses its
current state to choose an action: buy, hold, or sell.

The environment is the world the agent acts in. An episode is
one month of trading --- typically 18 to 23 trading days,
depending on the month. So the environment is the portfolio
over that month: it executes the action, applies the trading
fee, the price moves to the next day, and it hands back the
new state and the reward --- the change in portfolio wealth."

Do not mention MDPs, Bellman equations, or policy vs value here.
That comes on the next slide (the three algorithms).
""")


def slide4_algorithms(prs):
    s = blank_slide(prs)
    # Title wording set by Fiyin in PowerPoint (11 Sep)
    add_title(s, "RL Algorithms (Value-Based Methods vs Policy-Based Methods)",
              "the three algorithms compared in the study")
    centred_image(s, FIGS / "slide4_algorithms.png",
                  max_w_in=12.0, max_h_in=4.9, top_in=1.55)
    add_caption(s, "REINFORCE and DQN are implemented from scratch. PPO uses the standard library — a modern reference point.")
    add_slide_number(s, 4)
    add_notes(s, """
Your own words from rehearsal, tightened:

"Moving on to the reinforcement learning algorithms. There are two
families here: policy-based methods, where REINFORCE and PPO fall,
and value-based methods, like Deep Q-learning.

Deep Q-learning learns the VALUE of an action. It estimates how
good each action is --- how good is buying here, how good is
selling here --- and picks the action with the highest estimate.

REINFORCE learns the POLICY directly. The policy is a probability
distribution over actions --- say buy at 0.6, sell at 0.3, hold at
0.1. It adjusts those probabilities based on the outcome that
followed: actions that led to profit become more likely, actions
that led to loss become less likely.

PPO is exactly like REINFORCE, but it learns the policy in a safer
manner. It has a clipping parameter that caps how large any single
update to the policy can be, so the new policy never strays too
far from the old one.

REINFORCE and Deep Q-learning I wrote from scratch. For PPO I used
the standard library implementation deliberately --- it is the
modern reference point."

Timing: 90 seconds. Don't try to derive anything here — the full
objectives are on backup slides 14-16 if they ask.
""")


def slide5_rules(prs):
    s = blank_slide(prs)
    # Title and subtitle wording set by Fiyin in PowerPoint (11 Sep)
    add_title(s, "Dataset Splits",
              "To avoid look-ahead leakage — data was not shuffled and "
              "episodes were made independent of one another")
    centred_image(s, FIGS / "slide5_rules.png",
                  max_w_in=12.2, max_h_in=4.9, top_in=1.55)
    # Caption wording set by Fiyin in PowerPoint (11 Sep, second round)
    add_caption(s, "Two studies, two splits — the first is chronological; "
                   "the second uses a simulator trained with the same "
                   "training months and holds out every other real month "
                   "for testing.")
    add_slide_number(s, 5)
    add_notes(s, """
Your own words from rehearsal, tightened (three slips fixed —
see the CAREFUL notes at the bottom):

"Now for the dataset splits. The first study used real data from
the SPY index, to be as realistic as possible --- but this did not
turn out great, which led to the main study we will come back to.

Top bar: we used 2018 to 2022 as the training set --- 60 monthly
episodes the agent was TRAINED on. For validation we used the
whole of 2023: that is where the hyper-parameters were tuned, to
fix one setting for every test so the comparisons stay equal.
Testing was 2024 to 2025, touched exactly once at the end.

Because of the dataset constraints, the main study moved to a
simulator plus transfer tests, where we generated far more
episodes than the 60 real months could give. We kept the same
2018-to-2022 window, but now it CALIBRATES the simulator rather
than training the agent. From it we generated 3,000 episodes:
1,000 for upward markets, 1,000 for downward markets, and 1,000
for consolidating markets, where the price stays in the same
range. Testing was then done on real months from 2006 to 2017 ---
covering the 2008 market crash --- and from 2023 to 2025: 180
months in total that were never shown to the agents whatsoever.

All of this is to prevent look-ahead leakage, and to stop the
agent from simply memorising patterns."

CAREFUL — three slips from rehearsal to avoid on the day:
1. Say the agent was TRAINED on 2018-22 in study one; reserve the
   word "calibrated" for the simulator in the main study.
2. The simulator window is 2018 to 2022 (you said "2020 to 2022").
3. The transfer test starts at 2006, not 2008 (2008 is merely the
   crash year inside it). And it is "look-AHEAD leakage".

If asked what the 21 tests check: environment accounting, feature
timing, action masking, reward-state consistency.
""")


def slide6_first_result(prs):
    s = blank_slide(prs)
    # Title wording set by Fiyin in PowerPoint (12 Sep)
    add_title(s, "First Study - No agent beat the buy-and-hold benchmark",
              "trained on 60 real months, tested on 2024–2025")
    # two figures side by side: monthly earnings (left), per-seed detail (right)
    centred_image(s, FIGS / "slide6_first_result.png",
                  max_w_in=6.6, max_h_in=4.55, top_in=1.55,
                  centre_x_in=3.65)
    centred_image(s, PER_SEED_SCATTER,
                  max_w_in=5.9, max_h_in=4.55, top_in=1.75,
                  centre_x_in=10.05)

    # takeaway paragraph (Nguyen-corrected phrasing: every algorithm
    # RECEIVES its state; the failures did not act on the information in it)
    tb = s.shapes.add_textbox(Inches(0.55), Inches(6.30),
                              Inches(12.2), Inches(0.95))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_top = Emu(0)
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.LEFT
    r = p.add_run()
    r.text = ("No agent beat the buy-and-hold benchmark. The methods that "
              "earned the most did not act on the information provided by "
              "their state on any seed; the only method whose decisions "
              "changed with its state earned the least. The contradiction "
              "needed an explanation.")
    r.font.name = "Calibri"
    r.font.size = Pt(14)
    r.font.color.rgb = INK
    add_slide_number(s, 6)
    add_notes(s, """
Your own words from the latest rehearsal (Slide 6 v2), tightened —
one number slip fixed, see CAREFUL:

"For the first study we can see that no agent actually beat the
buy-and-hold benchmark. This was trained on 60 real months and
tested on the 2024-to-2025 period. So technically, if you bought
the asset at the beginning of January 2024 and held it to the end
of 2025, you would have earned an average of $180.25 per month.

The closest learner to this was REINFORCE at $171.51, then PPO at
$127.35, and Deep Q-learning at the bottom with $74.68.

At first glance you might say: okay, that looks efficient. But
looking closer at the seed distribution --- the right-hand chart
--- we see that REINFORCE and PPO, the policy methods, actually
stuck to fixed patterns: always buying, or never trading at all.

Part of what's to blame is the testing period itself: 2024 to
2025 was a strong upward market, so the most favourable strategy
was simply to always buy. But there was also a deeper underlying
cause --- the models overfitting --- which we will get to on the
next slide.

And here is the contradiction: the only method whose decisions
DID change with its state, Deep Q-learning, earned the least.
That contradiction needed an explanation."

CAREFUL — slip from rehearsal: the closest learner was
REINFORCE at $171.51 (you said PPO). PPO was $127.35.

The right-hand chart shows the per-seed detail: filled dots are
state-dependent seeds, hollow are mask-only --- point at it when
you say "looking closer at the seed distribution".

If asked HOW the state-dependence test works, answer here:
"I compared moments where the agent had the SAME set of legal
actions and asked whether it picked DIFFERENT actions. If it did,
the difference can only have come from the state it saw. If it
always picks the same action, it is a fixed rule, not a policy."

Next slide answers WHY this happened — the diagnosis.
""")


def slide7_state_test(prs):
    s = blank_slide(prs)
    add_title(s, "The behavioural test that changed the interpretation",
              "Question 2 — behaviour: does the learned agent actually use its market information?")
    centred_image(s, FIGS / "slide7_state_test.png",
                  max_w_in=11.2, max_h_in=4.9, top_in=1.55)
    add_caption(s, "If two moments have the SAME legal actions but the agent picks DIFFERENT actions, the difference must come from the state.")
    add_slide_number(s, 7)
    add_notes(s, """
Say:

"This is where the second question of the dissertation comes in:
does the learned agent actually USE its market information?

This is one of the contributions I want to explain properly.

To test whether an agent is really using its state information, I
compared moments where the AVAILABLE actions were identical --- so
the agent could buy or hold or sell in both --- and asked: did the
agent pick DIFFERENT actions?

If yes, the difference cannot be explained by what was legal. It
must be explained by what the agent SAW. That's genuine state
dependence.

If the agent picks the same action every time regardless of the
state, it's not really an agent --- it's a fixed rule wearing an
agent's clothes.

That test is what showed REINFORCE and PPO were converging on
'always buy' patterns despite looking profitable, and it's what
showed Deep Q-learning WAS state-dependent even though it earned
the least. Financial returns alone would have hidden this
completely."

This slide is worth two minutes. Examiners will ask about it.
""")


def _points_textbox(s, points, left_in, top_in, w_in, h_in,
                    size_pt=12, space_after=4):
    """Bold-led explanation lines, one idea per paragraph."""
    tb = s.shapes.add_textbox(Inches(left_in), Inches(top_in),
                              Inches(w_in), Inches(h_in))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_top = Emu(0)
    tf.margin_bottom = Emu(0)
    for i, (head, body) in enumerate(points):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = PP_ALIGN.LEFT
        p.space_after = Pt(space_after)
        rh = p.add_run()
        rh.text = head
        rh.font.name = "Calibri"
        rh.font.size = Pt(size_pt)
        rh.font.bold = True
        rh.font.color.rgb = NAVY
        rb = p.add_run()
        rb.text = body
        rb.font.name = "Calibri"
        rb.font.size = Pt(size_pt)
        rb.font.color.rgb = INK
    return tb


def slide8_diagnosis(prs, number: int = 7):
    """All about the FIRST study: why it failed, where 18,000 comes from."""
    s = blank_slide(prs)
    # Title/subtitle wording set by Fiyin in PowerPoint (12 Sep)
    add_title(s, "The diagnosis",
              "Data: is the historical record of one asset large enough to train these networks at all?")
    # first-study panel only, on the left
    centred_image(s, FIGS / "slide7_diagnosis_first.png",
                  max_w_in=5.7, max_h_in=4.9, top_in=1.60,
                  centre_x_in=3.35)

    # explanation on the right (one idea per line)
    points = [
        ("The network is a multi-layer perceptron (MLP).",
         " Two hidden layers of 128 units — 9 inputs \u2192 128 \u2192 128 "
         "\u2192 3 actions — is roughly 18,000 trainable weights."),
        # body wording edited by Fiyin in PowerPoint (12 Sep)
        ("60 examples are far too few for 18,000 weights.",
         " The network can memorise all 60 training months perfectly, "
         "preventing it from learning anything new — like a student who "
         "memorises the demo questions and then meets a completely "
         "different exam."),
        ("The training months lean upward.",
         " 29 of the 60 rose by more than 2%; only 16 fell (the COVID "
         "crash among them). In that market, 'always buy' is almost the "
         "best possible answer — and it needs no market information."),
        ("Why Deep Q-learning earned least.",
         " Estimating what an action is worth needs many visits to similar "
         "situations; sixty ever-shifting months never repeat, so its value "
         "estimates stayed noisy."),
        ("None of this can be tested on the real data — the real data is "
         "the problem.",
         " Testing it needs control over volume, balance and market "
         "conditions. That is what the main study builds \u2192 next slide."),
    ]
    _points_textbox(s, points, 6.55, 1.75, 6.2, 4.9, size_pt=12.5,
                    space_after=8)

    add_slide_number(s, number)
    add_notes(s, """
Your own words from rehearsal (Slide 7 p1+p2), tightened —
three slips fixed, see CAREFUL at the bottom:

"For the first study we used a multi-layer perceptron with two
hidden layers of 128 units, creating around 18,000 trainable
parameters --- with only 60 data points, keep in mind. With far
more trainable weights than training months, the network can
memorise all 60 months perfectly. Nothing forces it to learn a
rule for a different market condition. It's like a student
preparing for an exam with the demo questions, then getting to
the exam and seeing completely different questions.

What added to this is that most of the training months --- 29 of
the 60 --- were rising markets, and only 16 were falling markets,
like the COVID crash in 2020. This uneven batch of market
conditions teaches the agent to always buy, because that returns
the highest reward for the policy methods. Coupled with
memorisation from insufficient data, the policies did not need to
rely on the information provided by their state --- even though
the state contains multiple features that describe the market and
portfolio conditions. That is why REINFORCE and PPO essentially
won by chance during the testing period, which was also a
high-rising market.

Deep Q-learning, on the other hand, was lowest because estimating
the worth of an action needs many similar situations to learn
from. Sixty months that never repeat, with no balanced market
conditions, make its value estimates very noisy."

CAREFUL — slips from rehearsal:
1. 128 units per hidden layer, not 120.
2. 60 training months (you said "16 data points" once).
3. 16 of 60 fell, not 19.

If asked "why not just make the model smaller?" — I did also
shrink it (2,800 parameters, next slide). Both moves were needed.
""")


def slide9_simulator(prs):
    """All about the MAIN study: the simulator and the right-sized model."""
    s = blank_slide(prs)
    # Title wording set by Fiyin in PowerPoint (12 Sep evening)
    add_title(s, "The Main Study",
              "a simulator calibrated on the training months; and a "
              "smaller MLP network")
    # three-condition sample paths (left) + episodes-vs-parameters (right)
    centred_image(s, FIGS / "slide9_simulator.png",
                  max_w_in=7.6, max_h_in=3.6, top_in=1.70,
                  centre_x_in=4.30)
    centred_image(s, FIGS / "slide8_diagnosis_main.png",
                  max_w_in=4.3, max_h_in=3.6, top_in=1.65,
                  centre_x_in=10.55)

    # the beamer PDF's main-study bullets, in plain language
    # heads/wording edited by Fiyin in PowerPoint (12 Sep); the 20:1
    # arithmetic is spelled out because "20 to 1" alone was unexplained
    points = [
        ("A Market generator",
         " - with three known conditions: Rising, Falling, and "
         "Consolidating months, calibrated on the 60 real training "
         "months only — every other real month stays untouched for "
         "testing."),
        ("More Volume",
         " - 3,000 balanced training episodes (1,000 per condition), "
         "300 validation, 600 test — about 60,000 transitions per pass."),
        ("MLP Network",
         " - Cut to 64 and 32 hidden units \u2248 2,800 parameters, so "
         "the training data now outnumbers the parameters: 60,000 "
         "transitions \u00f7 2,800 weights \u2248 20 examples per weight."),
        ("On the balanced test, no fixed pattern is profitable.",
         " Buy-and-hold loses $60.29 per episode. Any profit must come "
         "from reading the state."),
    ]
    _points_textbox(s, points, 0.9, 5.35, 11.8, 1.75, size_pt=12,
                    space_after=3)

    add_slide_number(s, 8)
    add_notes(s, """
Your own words from rehearsal (Slide 8), tightened — one recurring
slip fixed, see CAREFUL:

"After understanding the limitations of the first study, I built a
small market simulator that generates monthly prices for the main
study. It is calibrated on the same training period, 2018 to 2022.

The good thing about a simulator is that we can recreate different
market conditions on demand. So we created 3,000 balanced training
episodes --- 1,000 per condition: an upward market, a downward
market, and a consolidation market where prices move side to side.

Now that we had substantial volume in the data, the other fix was
the network itself. We reduced the MLP's hidden layers to 64 and
32 units --- about 2,800 parameters in total --- which sits
comfortably against 3,000 training episodes, roughly twenty
examples per... [pause] --- the data now outnumbers the parameters
about 20 to 1. The exact opposite of the first study's
60-versus-18,000.

And the simulator gives us not just volume but LABELLED market
conditions: every episode is tagged rising, falling or flat, so we
can evaluate how the agent behaves in each condition separately.
On real data those conditions come mixed together.

One property by construction: on the balanced test set, no fixed
pattern profits. Buy-and-hold loses $60.29 per episode; never
trading earns $0. Any positive number must come from reading the
state.

The critical rule: the simulator only reads 2018 to 2022. Every
other real month --- 2006 to 2017, and 2023 to 2025 --- stays
completely untouched. Those months become the transfer test on
slide 10. With these changes came massive changes in the results
and in the behaviours the agents established."

CAREFUL — recurring slip: the calibration window is 2018 to 2022
(you said "2018 to 2012" this time, "2020 to 2022" before).
Also say "LABELLED market conditions", not "labour".

Anticipated question: "Isn't this just fitting a simple GBM?" —
Yes, deliberately. The whole point is a controlled experiment.
More realistic generators are future work; the transfer test
measures how much the simplification costs.
""")


def slide10_flip(prs):
    """Main-study result — drawn as a twin of the first-study chart."""
    s = blank_slide(prs)
    # Title wording set by Fiyin in PowerPoint (12 Sep evening)
    add_title(s, "The Analysis — controlled data, properly sized model",
              "Deep Q-learning improves overall")
    # two separate images: conditions LEFT, overall verdict RIGHT
    centred_image(s, FIGS / "slide10_conditions.png",
                  max_w_in=6.1, max_h_in=4.45, top_in=1.50,
                  centre_x_in=3.55)
    centred_image(s, FIGS / "slide10_overall.png",
                  max_w_in=6.1, max_h_in=4.45, top_in=1.50,
                  centre_x_in=9.85)

    # two text blocks, one under each image — wording set by Fiyin
    # in PowerPoint (12 Sep evening); his right block ended mid-
    # sentence ("but .") and is completed with the ch5 conclusion
    # layout (indented dash lines) set by Fiyin in PowerPoint (13 Sep)
    left_points = [
        ("In the simulated held-out test set", ""),
        ("Left — DQN performed well in all market conditions", " "),
        ("", "\t— keeping $473.72 of the $477.22 always-buy earns in "
         "rising markets (99%), "),
        ("", "\t— and losing $103.35 where always-buy loses $663.91 in "
         "falling ones (84% less)."),
    ]
    right_points = [
        ("Right — only Deep Q-learning earns.",
         " +$126.94 per episode on all six seeds. This is simulated "
         "data, not real-market prediction: with no constant rising "
         "markets, fixed patterns lose to transaction fees and falling "
         "markets."),
        ("REINFORCE and PPO collapse to fixed patterns again",
         " — never trade, or always buy. One REINFORCE seed in six "
         "established a policy that relies on information from the "
         "state (+$83.13), but the method does not find it reliably."),
    ]
    _points_textbox(s, left_points, 0.55, 6.05, 6.1, 1.30, size_pt=11.5,
                    space_after=2)
    _points_textbox(s, right_points, 6.75, 6.05, 6.3, 1.30, size_pt=11.5,
                    space_after=2)
    add_slide_number(s, 9)
    add_notes(s, """
Say (this is the main result --- slow down):

"This is the main study's exam: 600 SIMULATED test episodes the
agents never saw in training --- the simulator made 3,000
training episodes, 300 validation, and these 600 for testing,
all balanced across rising, falling and flat markets.

The left chart picks up the three conditions from the last
slide and shows what each method earned in each one. In rising
markets, Deep Q earns $473.72 of the $477.22 that always-buying
earns --- it keeps 99% of the upside. In falling markets, it
loses $103.35 where always-buying loses $663.91 --- an 84%
smaller loss, because it steps aside instead of holding on. In
flat markets it roughly matches the market. Stay in while it
rises, get out while it falls: that is the state-dependent
behaviour the first study looked for and could not find.

The right chart puts all 600 episodes together. Balance removes
the market's free ride --- think of testing a gambler with a
fair coin instead of a coin weighted towards heads: 'always bet
heads' stops looking clever. So blind holding loses about $60
per episode, because it pays fees and takes every falling market
in full. Doing nothing earns exactly $0 --- the zero line. A
positive number is only possible if the agent's decisions change
with the market. And the ranking completely inverts: Deep
Q-learning --- the worst method on real data --- is the only one
above the line. 127 dollars per episode, profitable on all six
seeds, decisions changing with the state on all six.

And why do REINFORCE and PPO end up in these fixed corners?
On balanced data, always-buying loses money, so never trading
becomes the locally safe answer: it earns nothing, but it cannot
lose. Once a policy-gradient method drifts close to a
deterministic answer, its gradient carries almost no signal, so
it stops exploring and gets stuck there. Deep Q-learning does
not have that failure mode --- its epsilon-greedy exploration
keeps trying every action a fraction of the time, no matter what
the current policy prefers.

The story becomes: the data set the limit, not the algorithm."

Seed detail if asked: REINFORCE --- 3 seeds never traded ($0.00
exactly), 2 always bought (-$58.70, matching the always-buy row
to the cent), 1 learned the conditioned policy (+$83.13, keeping
$417.02 of the up-market profit, down-loss $162.11). PPO --- 5
seeds never traded, 1 fixed buying pattern, 0 state-dependent.

ANTICIPATED QUESTION (asked by Fiyin himself, so examiners may
ask it too): "Are you saying buy-and-hold would lose money by the
end of 2025?" --- NO. Answer: "This is a controlled simulated
exam, not a market forecast. On the real 2024-25 data,
buy-and-hold earned $180.25 a month --- slide 6. The balanced
test deliberately removes the upward drift to check whether a
strategy has any skill beyond riding a rising market.
Buy-and-hold has none --- that is not a flaw of the market, it is
the definition of buy-and-hold."

ANTICIPATED QUESTION: "Is this the transfer test?" --- NO.
Answer: "This is the simulated test set --- 600 held-out episodes
from the simulator, unseen in training. The transfer test is the
next slide: the same policies run unchanged on 180 REAL months
the simulator never touched." Order of exams: training (3,000
sim) -> validation (300 sim) -> simulated test (600, THIS slide)
-> real transfer test (180 real months, slide 10).

Exact per-condition numbers (Table 5.3, $ per episode) if asked:
Deep Q  +473.72 up / -103.35 down / +10.45 flat;
REINFORCE  +228.58 / -248.32 / +2.60;
PPO  +67.69 / -99.04 / -2.14;
Always-buy  +477.22 / -663.91 / +10.58;
Buy-and-hold  +510.17 / -710.06 / +19.00.

Timing: 3 minutes. This is the slide the examiners will remember.
""")


def slide11_stress(prs):
    s = blank_slide(prs)
    # subtitle wording set by Fiyin ("running unchanged")
    add_title(s, "The transfer test — 180 real months",
              "policies trained on the simulator, running unchanged on "
              "real data — including the 2008 crash")
    # two separate images: lines LEFT, transfer means RIGHT
    centred_image(s, FIGS / "slide11_lines.png",
                  max_w_in=7.9, max_h_in=4.9, top_in=1.55,
                  centre_x_in=4.35)
    centred_image(s, FIGS / "slide11_bars.png",
                  max_w_in=4.7, max_h_in=4.9, top_in=1.55,
                  centre_x_in=10.55)
    add_caption(s, "Sharpe ratio (risk-adjusted return): 0.224 for our agent vs 0.198 for buy-and-hold, over 180 real months.")
    add_slide_number(s, 10)
    add_notes(s, """
Say:

"The final test is the TRANSFER test: I took the policies trained
on the simulator and ran them, unchanged, on 180 real months the
training never saw. That's 2006 to 2017 plus 2023 to 2025 ---
everything OUTSIDE the 2018-2022 window that calibrated the
simulator.

The left chart is drawn from the actual experiment data --- the
same numbers behind the dissertation figure --- and now shows
ALL the methods month by month. Notice buy-and-hold finishes
HIGHEST. I want to be honest about that: on raw return we do not
beat the market's average direction, because the market mostly
rose over these two decades. The right panel gives the means:
buy-and-hold $80.24 a month, always-buy $76.48, our agent
$60.28, REINFORCE $31.66, PPO $8.32 --- the policy-gradient
methods are weaker because most of their seeds collapsed to
fixed patterns.

So why does the agent still win? Look at the shaded band: late
2008. Buy-and-hold lost $1,665 in October 2008 alone --- you can
see its line plunge. All six of our agents held ZERO shares for
that entire month, so their line barely dips. Not because they
were told about 2008 --- they had never seen a real market in
training --- but because the volatility features they observed
matched their learned 'falling condition', so they stepped aside.

On risk-adjusted return, the Sharpe ratio, our agent finishes
ahead of buy-and-hold: 0.224 against 0.198. It gives up some
upside and avoids most of the crashes."

If asked why the deck's earlier draft chart differed from the
dissertation figure: it was a stylised sketch; this one is
plotted directly from sim_results.json (real_per_month), so it
now matches the thesis exactly.

Anticipated question: "Isn't the agent just classifying your
generator's regimes?" — Partly, yes. But those regularities
carried to 2008, which is not something the generator ever
produced. See viva Q&A prep for the full answer.
""")


def _panel(s, x_in, top_in, w_in, h_in, head, colour):
    """Rounded panel with a coloured border and bold header."""
    box_shape = s.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE,
        Inches(x_in), Inches(top_in), Inches(w_in), Inches(h_in),
    )
    box_shape.fill.solid()
    box_shape.fill.fore_color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
    box_shape.line.color.rgb = colour
    box_shape.line.width = Pt(2.0)
    box_shape.shadow.inherit = False

    tb = s.shapes.add_textbox(Inches(x_in + 0.25), Inches(top_in + 0.14),
                              Inches(w_in - 0.5), Inches(0.42))
    tf = tb.text_frame
    tf.margin_top = Emu(0); tf.margin_bottom = Emu(0)
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.LEFT
    r = p.add_run()
    r.text = head
    r.font.name = "Calibri"
    r.font.size = Pt(17)
    r.font.bold = True
    r.font.color.rgb = colour


def slide12_closing(prs):
    """Slide 11 — single closing slide, four panels: Findings and
    Contributions on the left, Limitations and Future work on the
    right (layout requested by Fiyin, 12 Sep evening)."""
    s = blank_slide(prs)
    add_title(s, "Conclusions",
              "findings · contributions · limitations · future work")

    RED = RGBColor(0xC0, 0x39, 0x2B)
    GREEN = RGBColor(0x2A, 0x9D, 0x3D)
    ORANGE = RGBColor(0xE0, 0x7A, 0x1F)

    col_l, col_r = 0.45, 6.80
    w = 6.10
    top1, top2 = 1.28, 4.06
    h = 2.66

    # LEFT TOP — findings
    _panel(s, col_l, top1, w, h, "Findings", NAVY)
    _points_textbox(s, [
        ("The data caused the limitation, not the algorithm.",
         " 60 real months could never teach an 18,000-weighted "
         "network; but 3,000 balanced simulated episodes could."),
        ("Balanced data separated the algorithms.",
         " Deep Q-learning learned a state-dependent policy on all six "
         "seeds; REINFORCE and PPO mostly collapsed to fixed habits — "
         "always buying, or never trading."),
        ("The learned behaviour carried into real markets.",
         " Run on 180 unseen months, every method stayed "
         "profitable — and Deep Q moved to zero shares through the "
         "October 2008 crash, a period no agent had ever seen."),
    ], col_l + 0.28, top1 + 0.58, w - 0.56, h - 0.75, size_pt=10.5,
        space_after=5)

    # LEFT BOTTOM — contributions
    _panel(s, col_l, top2, w, h, "Contributions", ORANGE)
    _points_textbox(s, [
        ("A careful way to evaluate.",
         " Every decision was made on validation data, with six "
         "seeds, and automated tests protecting the accounting."),
        ("A behavioural test.",
         " It tells whether a policy genuinely reads the market apart "
         "from a fixed habit — and changed how every result was read."),
        ("A calibrated simulator.",
         " It turns market condition into something we can control "
         "and test, grown from 60 real months."),
    ], col_l + 0.28, top2 + 0.58, w - 0.56, h - 0.75, size_pt=10.5,
        space_after=5)

    # RIGHT TOP — limitations
    _panel(s, col_r, top1, w, h, "Limitations", RED)
    _points_textbox(s, [
        ("One asset.",
         " We studied SPY only, so the findings do not automatically "
         "carry to other markets."),
        ("A deliberately simple simulator.",
         " Months are categorized by their rising / falling / "
         "consolidating conditions with no market changes inside "
         "a month."),
        ("Policy-gradient methods were not separately tuned.", ""),
        ("Modest evidence.",
         " Six seeds with no formal significance tests."),
    ], col_r + 0.28, top1 + 0.58, w - 0.56, h - 0.75, size_pt=10.5,
        space_after=5)

    # RIGHT BOTTOM — future work
    _panel(s, col_r, top2, w, h, "Future work", GREEN)
    _points_textbox(s, [
        ("Better simulators.",
         " Markets should realistically include changes inside an "
         "episode rather than fixed categories."),
        ("Continuous episodes",
         " to capture a realistic market where months carry over into "
         "each other with volatility, and walk-forward recalibration."),
        ("Better reward function.",
         " This prevents an agent from refusing to trade."),
        ("Stronger evidence.",
         " More seeds with confidence intervals, and use of other "
         "assets."),
    ], col_r + 0.28, top2 + 0.58, w - 0.56, h - 0.75, size_pt=10.5,
        space_after=5)

    add_caption(s, "What these methods learn is decided by the data "
                   "they are given.  —  Thank you.")
    add_slide_number(s, 11)
    add_notes(s, """
Closing statement (about 90 seconds — walk the four boxes):

"To conclude. Three findings.

The data set the limit, not the algorithm: 60 real months could
never teach an 18,000-weight network, and 3,000 balanced
simulated episodes could. Fixing the data flipped the winner ---
Deep Q-learning went from last on real data to the only method
that earns on the balanced test. And the caution it learned
survived reality: zero shares through October 2008, a crisis it
had never seen.

What the work contributes: a careful way to evaluate --- every
decision on validation data, six matched seeds, 21 automated
tests; a behavioural test that tells a real policy apart from a
fixed habit; and a calibrated simulator that turns market
condition into something we can control.

What it cannot claim: it is one asset; the simulator is
deliberately simple --- the transfer test measures exactly what
that simplicity costs; and it is six seeds without formal
significance testing.

Future work follows directly: richer simulators, rewards that
cannot be gamed by refusing to trade, and stronger evidence.

If you take away one line: what these methods learn is decided
by the data they are given.

Thank you. I'm happy to take questions."

Timing: 90 seconds max. Then stop talking.
""")


def slide13_demo(prs):
    """Slide 12 — the QR demo (Fiyin pasted it into the main deck on
    13 Sep; ported here so rebuilds keep it). Offered ONLY at the very
    end, after the examiners have finished their questions. No slide
    number badge: it sits outside the numbered 11-slide talk."""
    FIGS_EXTRA = ROOT / "latex" / "viva" / "figs_extra"
    GREEN = RGBColor(0x2A, 0x9D, 0x3D)
    s = blank_slide(prs)
    add_title(s, "See it run — a live demo on your phone",
              "the transfer test as an animation: the same 180 months, "
              "drawn month by month")

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

    s.shapes.add_picture(str(FIGS_EXTRA / "demo_phone.png"),
                         Inches(5.05), Inches(1.45), height=Inches(5.05))

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


def main():
    prs = Presentation()
    prs.slide_width = Inches(SLIDE_W_IN)
    prs.slide_height = Inches(SLIDE_H_IN)

    slide1_title(prs)
    slide2_decision(prs)
    slide3_rl(prs)
    slide4_algorithms(prs)
    slide5_rules(prs)
    slide6_first_result(prs)
    # slide7_state_test removed from the flow (11 Sep) — its explanation
    # now lives in slide 6's notes; the builder is kept for the extended
    # deck's import and possible backup use.
    slide8_diagnosis(prs)   # now position 7
    slide9_simulator(prs)
    slide10_flip(prs)
    slide11_stress(prs)
    slide12_closing(prs)   # position 11 — single closing slide
    slide13_demo(prs)      # position 12 — QR demo, offered after Q&A only

    prs.save(OUT)
    print(f"Wrote {OUT}  ({OUT.stat().st_size/1024:.0f} KB, {len(prs.slides)} slides)")


if __name__ == "__main__":
    main()
