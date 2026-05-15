"""Build the EEEM004 Interim Review Word document.

Outputs:
    reports/generated/exports/InterimReview.docx

The document follows the supervisor-facing Interim Review form structure,
with all student-owned sections populated from the rebuilt dissertation
framing. The blue supervisor boxes are intentionally left empty.

Run:
    venv/bin/python reports/builders/build_interim_review_docx.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
from docx import Document
from docx.enum.table import WD_ALIGN_VERTICAL
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS = ROOT / "experiments" / "results"
EXPORTS = ROOT / "reports" / "generated" / "exports"
EXPORTS.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Helpers (kept parallel to build_main_dissertation_docx.py)
# --------------------------------------------------------------------------- #
def latest_json(prefix: str) -> dict | list:
    files = sorted(p for p in RESULTS.glob(f"{prefix}_*.json"))
    if not files:
        return []
    return json.loads(files[-1].read_text(encoding="utf-8"))


def avg(rows: Iterable[dict], key: str) -> float:
    rows = list(rows)
    if not rows:
        return float("nan")
    return float(np.mean([float(r[key]) for r in rows]))


def add_heading(doc: Document, text: str, level: int) -> None:
    h = doc.add_heading(text, level=level)
    for run in h.runs:
        run.font.color.rgb = RGBColor(0, 0, 0)


def add_para(doc: Document, text: str, *, italic: bool = False, bold: bool = False, align=None) -> None:
    p = doc.add_paragraph()
    if align is not None:
        p.alignment = align
    run = p.add_run(text)
    run.italic = italic
    run.bold = bold


def add_bullets(doc: Document, items: list[str]) -> None:
    for item in items:
        doc.add_paragraph(item, style="List Bullet")


def add_supervisor_box(doc: Document, prompt: str) -> None:
    """Render a labelled empty box where the supervisor will write."""
    p = doc.add_paragraph()
    run = p.add_run(prompt)
    run.italic = True
    run.font.color.rgb = RGBColor(0x1F, 0x49, 0x7D)
    run.font.size = Pt(10)
    table = doc.add_table(rows=1, cols=1)
    table.style = "Table Grid"
    cell = table.rows[0].cells[0]
    cell.text = " "
    cell.paragraphs[0].runs[0].font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
    cell.vertical_alignment = WD_ALIGN_VERTICAL.TOP
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:fill"), "EAF1F8")
    tc_pr.append(shd)
    set_cell_height(cell, Cm(2.5))


def set_cell_height(cell, height) -> None:
    tr = cell._tc.getparent()
    trPr = tr.get_or_add_trPr()
    trHeight = OxmlElement("w:trHeight")
    trHeight.set(qn("w:val"), str(int(height.emu / 635)))
    trHeight.set(qn("w:hRule"), "atLeast")
    trPr.append(trHeight)


def page_break(doc: Document) -> None:
    doc.add_page_break()


def set_default_font(doc: Document, family: str = "Calibri", size: int = 11) -> None:
    style = doc.styles["Normal"]
    style.font.name = family
    style.font.size = Pt(size)
    rpr = style.element.rPr
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.append(rfonts)
    for attr in ("w:ascii", "w:hAnsi", "w:cs"):
        rfonts.set(qn(attr), family)


def add_cover_table(doc: Document, fields: list[tuple[str, str]]) -> None:
    table = doc.add_table(rows=len(fields), cols=2)
    table.style = "Light Grid Accent 1"
    for i, (label, value) in enumerate(fields):
        lc, vc = table.rows[i].cells
        lc.text = label
        vc.text = value
        for run in lc.paragraphs[0].runs:
            run.bold = True
        lc.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
        vc.vertical_alignment = WD_ALIGN_VERTICAL.CENTER


def add_results_table(doc: Document, rows: list[dict]) -> None:
    headers = [
        "Agent",
        "Final value (USD)",
        "Sharpe",
        "Max DD",
        "VaR-95 viol.",
        "Terminal preservation",
        "Path preservation (1−MDD)",
    ]
    table = doc.add_table(rows=1, cols=len(headers))
    table.style = "Light Grid Accent 1"
    hdr_cells = table.rows[0].cells
    for i, h in enumerate(headers):
        hdr_cells[i].text = h
        for r in hdr_cells[i].paragraphs[0].runs:
            r.bold = True
    for row in rows:
        cells = table.add_row().cells
        cells[0].text = row["agent"]
        cells[1].text = f"${row['final']:,.0f}"
        cells[2].text = f"{row['sharpe']:+.4f}"
        cells[3].text = f"{row['mdd']:.4f}"
        cells[4].text = f"{row['var']:.4f}"
        cells[5].text = f"{row['pres']:.4f}"
        cells[6].text = f"{1.0 - float(row['mdd']):.4f}"
    for r in table.rows:
        for c in r.cells:
            c.vertical_alignment = WD_ALIGN_VERTICAL.CENTER


def add_plan_table(doc: Document, rows: list[tuple[str, str, str]]) -> None:
    headers = ["Working period", "Tasks to undertake", "Milestones (with target dates)"]
    table = doc.add_table(rows=1, cols=len(headers))
    table.style = "Light Grid Accent 1"
    hdr_cells = table.rows[0].cells
    for i, h in enumerate(headers):
        hdr_cells[i].text = h
        for r in hdr_cells[i].paragraphs[0].runs:
            r.bold = True
    for period, tasks, milestones in rows:
        cells = table.add_row().cells
        cells[0].text = period
        cells[1].text = tasks
        cells[2].text = milestones
        for run in cells[0].paragraphs[0].runs:
            run.bold = True


def add_status_table(doc: Document, rows: list[tuple[str, str, str]]) -> None:
    headers = ["Step", "Status", "Notes"]
    table = doc.add_table(rows=1, cols=len(headers))
    table.style = "Light Grid Accent 1"
    hdr_cells = table.rows[0].cells
    for i, h in enumerate(headers):
        hdr_cells[i].text = h
        for r in hdr_cells[i].paragraphs[0].runs:
            r.bold = True
    for step, status, notes in rows:
        cells = table.add_row().cells
        cells[0].text = step
        cells[1].text = status
        cells[2].text = notes


# --------------------------------------------------------------------------- #
# Content assembly
# --------------------------------------------------------------------------- #
def build_results_rows() -> list[dict]:
    baseline = latest_json("baseline")
    prob = latest_json("probabilistic")
    bench = latest_json("benchmarks")
    rules = latest_json("rule_baseline")

    def _spy_only(rows: list) -> list:
        if not rows:
            return []
        return [r for r in rows if r.get("ticker", "SPY") == "SPY"]

    baseline = _spy_only(baseline)
    prob = _spy_only(prob)
    bench_spy = _spy_only(bench)
    rules_spy = _spy_only(rules)
    bench_lookup = {r["agent"]: r for r in bench_spy}
    rule_lookup = {r["agent"]: r for r in rules_spy}

    rows: list[dict] = []
    if baseline and prob:
        rows.extend([
            {
                "agent": "Baseline PPO",
                "final": avg(baseline, "final_portfolio_value"),
                "sharpe": avg(baseline, "sharpe_ratio"),
                "mdd": avg(baseline, "max_drawdown"),
                "var": avg(baseline, "var_95_violation_rate"),
                "pres": avg(baseline, "capital_preservation_rate_95pct_hwm"),
            },
            {
                "agent": "Probabilistic PPO",
                "final": avg(prob, "final_portfolio_value"),
                "sharpe": avg(prob, "sharpe_ratio"),
                "mdd": avg(prob, "max_drawdown"),
                "var": avg(prob, "var_95_violation_rate"),
                "pres": avg(prob, "capital_preservation_rate_95pct_hwm"),
            },
        ])
    for label, display in (
        ("stop_loss_5pct", "Rule-based stop-loss (5%)"),
        ("stop_loss_10pct", "Rule-based stop-loss (10%)"),
    ):
        r = rule_lookup.get(label)
        if r:
            rows.append({
                "agent": display,
                "final": float(r["final_portfolio_value"]),
                "sharpe": float(r["sharpe_ratio"]),
                "mdd": float(r["max_drawdown"]),
                "var": float(r["var_95_violation_rate"]),
                "pres": float(r["capital_preservation_rate_95pct_hwm"]),
            })
    for label in ("buy_and_hold", "all_cash"):
        b = bench_lookup.get(label)
        if b:
            rows.append({
                "agent": "Buy-and-hold (SPY)" if label == "buy_and_hold" else "All-cash",
                "final": float(b["final_portfolio_value"]),
                "sharpe": float(b["sharpe_ratio"]),
                "mdd": float(b["max_drawdown"]),
                "var": float(b["var_95_violation_rate"]),
                "pres": float(b["capital_preservation_rate_95pct_hwm"]),
            })
    return rows


# --------------------------------------------------------------------------- #
# Build
# --------------------------------------------------------------------------- #
def build() -> Path:
    doc = Document()
    set_default_font(doc)
    for section in doc.sections:
        section.top_margin = Cm(2.0)
        section.bottom_margin = Cm(2.0)
        section.left_margin = Cm(2.2)
        section.right_margin = Cm(2.2)

    # ----- Cover banner -----
    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = title.add_run("EEEM004 — MSc Project")
    r.bold = True
    r.font.size = Pt(20)

    sub = doc.add_paragraph()
    sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = sub.add_run("Interim Review")
    r.italic = True
    r.font.size = Pt(14)

    doc.add_paragraph()
    add_para(
        doc,
        "This document is the student's submission for the EEEM004 interim review. "
        "It mirrors the structure of the official Interim Review form. The blue boxes "
        "are reserved for the supervisor's written assessment and are intentionally "
        "left empty.",
        italic=True,
    )

    # ----- Cover information -----
    add_heading(doc, "Cover information", 1)
    add_cover_table(doc, [
        ("Name", "Fiyin Akano"),
        ("URN", "6962514"),
        ("Supervisor", "Dr Cuong Nguyen"),
        ("Second supervisor (if applicable)", "[INSERT IF APPLICABLE]"),
        ("Date of meeting", "[INSERT MEETING DATE]"),
        ("Module", "EEEM004 — MSc Dissertation (cross-year)"),
        ("Department", "Electrical and Electronic Engineering, University of Surrey"),
        ("Target submission date", "1 September 2026"),
    ])

    # ----- Project title -----
    add_heading(doc, "Project title", 1)
    add_para(
        doc,
        "Probabilistic Deep Reinforcement Learning for Portfolio Risk Analysis: "
        "drawdown-constrained portfolio control with an uncertainty-aware "
        "reinforcement-learning policy.",
        bold=True,
    )

    # ----- Problem statement -----
    add_heading(doc, "Problem statement", 1)
    add_para(
        doc,
        "Start with a concrete picture. Imagine putting one million US dollars into "
        "the US stock market in January 2022 and holding on. By January 2025 the "
        "account is worth about $1.52 million — a clean win. But on the way there, "
        "in October 2022, the same account briefly read $750,000 — a 25 % drop from "
        "the peak. For an individual that drop is scary; for a pension fund or an "
        "endowment it is something different — it is a breach of contract. Many "
        "institutional mandates carry an explicit drawdown limit (a maximum permitted "
        "loss measured from the peak rather than from the starting balance), and "
        "when that limit is breached, redemption rights kick in, trustees can be "
        "removed, and the fund can be forcibly liquidated. Buy-and-hold violates "
        "these limits routinely; manually setting a stop-loss (sell when the price "
        "has fallen 5 % below its peak) violates them too — it sells late and buys "
        "back later. The dissertation asks whether a small AI agent can do better.",
    )
    add_para(
        doc,
        "Reframed problem statement. Many investors and institutional mandates are "
        "required to keep portfolio drawdown from peak below a stated limit (commonly "
        "between 5 % and 20 %) while still beating cash and ideally beating passive "
        "index exposure. The standard quantitative answers — Markowitz mean-variance, "
        "risk parity, fixed-rule stop-losses — either assume the joint distribution "
        "of returns is stationary or react too slowly when the market regime changes. "
        "This dissertation studies whether a deep-reinforcement-learning agent that "
        "conditions on its own forecaster's predictive uncertainty (how confident the "
        "forecaster is, not just what it predicts) can sit on a more attractive point "
        "of the return-versus-drawdown trade-off than (a) passive buy-and-hold, (b) "
        "a rule-based stop-loss policy, and (c) a baseline PPO with no uncertainty "
        "signal — measured on a held-out test window that contains real macro shocks "
        "(2022–2025), with reproducible random seeds and an out-of-time generalisation "
        "check.",
    )
    add_para(
        doc,
        "Where the standard answers fail. Mean-variance (Markowitz, 1952) is single-"
        "period and treats upside and downside wiggles symmetrically — it has no "
        "memory of the running peak. Value-at-Risk and expected shortfall "
        "(Rockafellar and Uryasev, 2000) measure how bad a single bad day could be, "
        "but they cannot tell you how many bad days in a row you can endure before "
        "the loss-from-peak crosses the limit. Conditional drawdown-at-risk "
        "(Chekhlov, Uryasev and Zabarankin, 2005) is path-dependent and matches the "
        "institutional constraint shape, but it is a one-shot static optimisation: "
        "it picks one weight vector and does not adapt mid-window. Reactive trailing "
        "stop-losses adapt sequentially but fire after the drawdown has already "
        "begun and typically forfeit the recovery on the way back up.",
    )
    add_para(
        doc,
        "Why this matters in practice. Drawdown control is a real, billion-dollar "
        "institutional problem. CalPERS and other major pension funds, sovereign-"
        "wealth funds, university endowments and CTA funds all run explicit drawdown "
        "limits in their governance documents. Bridgewater Associates' All Weather "
        "fund, which has managed over 150 billion US dollars at peak, is publicly "
        "described by its founder as designed to lose less in any environment — an "
        "explicit drawdown-control objective. The drawdown literature itself "
        "(Magdon-Ismail and Atiya, 2004; Young, 1991; Sortino and Price, 1994; "
        "Chekhlov, Uryasev and Zabarankin, 2005) is the formal home for these "
        "constraints; this dissertation does not invent them, it picks up the "
        "tradition and extends it from one-shot optimisation to a sequential, "
        "uncertainty-aware decision policy.",
    )

    # ----- Objectives -----
    add_heading(doc, "Objectives", 1)
    add_para(
        doc,
        "The objectives have been refined during Phase 0 and Phase 1 in light of "
        "supervisor feedback. They are stated in the order in which the dissertation "
        "answers them: the core scientific question first, the empirical evidence "
        "(both single-asset headline and multi-asset / out-of-time generalisation) "
        "second, the reproducibility apparatus third, and the honest position on "
        "where the method works and where it does not fourth.",
    )
    add_bullets(doc, [
        "O1 — the core scientific question. Can a deep reinforcement-learning agent that conditions on its own forecaster's predictive uncertainty (how confident the forecaster is, not just what it predicts) sit closer to the drawdown-constrained risk-adjusted return frontier than uncertainty-blind alternatives? Operationally: a DeepAR-style probabilistic LSTM emits predictive mean and variance; a Proximal Policy Optimization (PPO) policy reads the variance as a state feature and as a hard guard that blocks new long-side trades when the uncertainty score exceeds a quantile threshold.",
        "O2 — the empirical evidence. Evaluate the resulting policy on a fixed held-out window containing real macro shocks (2022 to 2025) against three named comparators — passive buy-and-hold, a rule-based trailing stop-loss policy, and a baseline PPO with no uncertainty signal — and check that the conclusions survive contact with (a) a 70-ticker diversified-equity test universe (41 single-name US large-cap equities + 29 ETFs spanning broad-market, sector, dividend, thematic and commodity exposure) and (b) a four-fold walk-forward grid in which the train, validation and test windows roll forward across 2018–2025. Headline metrics: Sharpe ratio, terminal value relative to buy-and-hold, and the capital-preservation ratio against the running high-watermark.",
        "O3 — reproducibility. Pin down a fully reproducible evaluation protocol of fixed splits, fixed random seeds, scripted experiment runners, scripted reporting and a shared metric set, so that any comparison made in this dissertation is genuinely like-for-like and can be reproduced from the public repository in a single command sequence.",
        "O4 — honest position on where it works and where it does not. Diagnose the regimes in which the uncertainty-aware policy beats the alternatives and the regimes in which it does not, and take a defensible position — on the strength of O1 to O3 — on when an explicit uncertainty signal earns a place in a portfolio control loop and, just as important, on when it does not.",
    ])

    page_break(doc)

    # ----- Literature -----
    add_heading(doc, "Literature review (key references)", 1)
    add_para(
        doc,
        "Below are ten items I rely on in this project. For each one there is a short "
        "paragraph in plain English: what it is, why people use it, and how it connects "
        "to my work. Each paragraph ends by tagging one of three roles:",
    )
    add_bullets(doc, [
        "Development — the reference directly shaped what I built in code: the "
        "learning algorithm, the forecasting style, or the software library I call "
        "from the runners.",
        "Evaluation — the reference shaped what I measure and how I explain the "
        "results: risk metrics, baselines, or the language for “large losses” and tail "
        "risk. I may cite the paper even when I do not run their optimisation method.",
        "Positioning (related work) — the reference situates this project next to "
        "other people's ideas or tools so a reader can see what is new and what is not. "
        "I may cite work I do not import or re-implement, for example a public library "
        "that other finance-RL papers use.",
    ])
    add_para(
        doc,
        "The full dissertation cites more papers in Chapter 2; this list is the compact "
        "set for the interim review.",
    )

    INTERIM_LITERATURE = [
        (
            "Markowitz, H. (1952). Portfolio Selection. Journal of Finance, 7(1), 77–91.",
            "This paper started modern portfolio theory: choose weights to balance "
            "expected return against variance in one period. People still use it as the "
            "baseline language for diversification and efficient portfolios. In my "
            "project it is evaluation background only: it explains why a single-period "
            "mean–variance picture does not capture path risk like losing money from a "
            "peak. I do not solve a Markowitz optimisation problem; I compare learning "
            "agents with simple baselines under fixed rules.",
        ),
        (
            "Sortino, F. A., & Price, L. N. (1994). Performance Measurement in a Downside "
            "Risk Framework. Journal of Investing, 3(3), 59–64.",
            "Sortino and Price argue that upside and downside should not be punished the "
            "same way when we score performance. They replace the Sharpe denominator with "
            "downside deviation below a target return. In my project this supports the "
            "plain-English goal of caring about large losses, not just volatility. I "
            "still report Sharpe for comparability with other work, but Sortino motivates "
            "why drawdown-style measures sit beside it in the metric table.",
        ),
        (
            "Rockafellar, R. T., & Uryasev, S. (2000). Optimization of Conditional "
            "Value-at-Risk. Journal of Risk, 2(3), 21–42.",
            "This paper sets up convex optimisation for tail risk using Conditional "
            "Value-at-Risk (expected loss in the worst tail of outcomes). Practitioners "
            "use it when they care about bad days, not just average variance. In my "
            "project it is evaluation background for the VaR-style violation metric I "
            "report: it explains what “tail behaviour” means in the tables. I do not "
            "solve their optimisation programme; I train RL agents and read tail metrics "
            "off simulated paths.",
        ),
        (
            "Magdon-Ismail, M., & Atiya, A. F. (2004). Maximum Drawdown. Risk, 17(10), "
            "99–102.",
            "This paper studies maximum drawdown — how far wealth can fall from a "
            "running peak — as a path risk measure. That matches how investors actually "
            "feel pain during crashes. In my project it justifies reporting maximum "
            "drawdown and capital preservation against a high watermark: those numbers "
            "answer “how bad did it get along the way,” not only “how did we finish.”",
        ),
        (
            "Chekhlov, A., Uryasev, S., & Zabarankin, M. (2005). Drawdown Measure in "
            "Portfolio Optimization. International Journal of Theoretical and Applied "
            "Finance, 8(1), 13–58.",
            "These authors build portfolio optimisation so that drawdown risk is "
            "controlled inside the optimisation problem itself (conditional "
            "drawdown-at-risk). Pension-style mandates often think in drawdown limits. "
            "In my project this is related work for positioning: classical methods pick "
            "weights once from historical scenarios; I study a daily rule that can react "
            "when the forecaster looks unsure. I do not implement their linear programme.",
        ),
        (
            "Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). "
            "Proximal Policy Optimization Algorithms. arXiv:1707.06347.",
            "Proximal Policy Optimization (PPO) is a stable policy-gradient algorithm "
            "for reinforcement learning in continuous or discrete action spaces. It is "
            "widely used because it trains reliably with reasonable defaults. In my "
            "project PPO is the optimiser for both the baseline agent and the "
            "uncertainty-aware agent (via Stable-Baselines3). All comparisons keep the "
            "same algorithm class so differences come from the uncertainty controls, "
            "not from swapping RL methods.",
        ),
        (
            "Jiang, Z., Xu, D., & Liang, J. (2017). A Deep Reinforcement Learning "
            "Framework for the Financial Portfolio Management Problem. arXiv:1706.10059.",
            "This paper trains a deep network end-to-end from prices to portfolio "
            "weights using reinforcement learning. It showed that RL can learn portfolio "
            "rules without hand-writing a trading strategy. In my project it is related "
            "work that motivates using RL on markets at all. My setup differs because I "
            "add a daily unsureness score from a probabilistic forecaster and I compare "
            "against rule-based stops and buy-and-hold on the same protocol.",
        ),
        (
            "Liu, X.-Y., Yang, H., Gao, J., & Wang, C. D. (2021). FinRL: A Deep "
            "Reinforcement Learning Library for Automated Stock Trading in Quantitative "
            "Finance. arXiv:2011.09607.",
            "FinRL is an open-source library that packages trading environments and RL "
            "training workflows so finance experiments can be repeated on a shared Gym-"
            "style API. In my project FinRL is cited as related infrastructure only. The "
            "Phase 1 experiments that produce my dissertation numbers do not import "
            "FinRL: they use Stable-Baselines3 PPO with a custom Gymnasium environment "
            "in experiments/common.py so every rule about trade scaling and guards is "
            "visible in one file. An optional Phase 0 demo script "
            "(phase0_examples/finrl_ppo_example.py) can load FinRL for illustration; it "
            "is not part of the reported evaluation pipeline.",
        ),
        (
            "Salinas, D., Flunkert, V., Gasthaus, J., & Januschowski, T. (2020). DeepAR: "
            "Probabilistic Forecasting with Autoregressive Recurrent Networks. "
            "International Journal of Forecasting, 36(3), 1181–1191.",
            "DeepAR trains a recurrent network to output a full predictive distribution "
            "for the next step, not just a point forecast. That gives a natural handle "
            "on “how spread out” tomorrow’s outcome might be. In my project I use a "
            "stripped-down DeepAR-style forecaster with a Gaussian output head: it "
            "produces a mean and a spread for the next log return, and I map the spread "
            "to a single daily unsureness score that the trading rule uses to shrink "
            "trades or block new long trades.",
        ),
        (
            "Raffin, A., Hill, A., Gleave, A., Kanervisto, A., Ernestus, M., & Dormann, "
            "N. (2021). Stable-Baselines3: Reliable Reinforcement Learning "
            "Implementations. Journal of Machine Learning Research, 22(268), 1–8.",
            "Stable-Baselines3 is a maintained implementation of standard RL algorithms "
            "(including PPO) that follows the Gymnasium API and is widely used for "
            "reproducibility. In my project it is the training engine behind both "
            "runners. Using a standard library keeps the algorithm choice boring on "
            "purpose so the dissertation can focus on the uncertainty controls and the "
            "fair baseline comparisons.",
        ),
    ]
    for citation, para in INTERIM_LITERATURE:
        add_para(doc, citation, bold=True)
        add_para(doc, para)

    page_break(doc)

    # ----- Technical progress -----
    add_heading(doc, "Technical progress", 1)
    add_heading(doc, "Summary", 2)
    add_para(
        doc,
        "Phase 0 and Phase 1 are in place and reproducible end-to-end. The "
        "dissertation compares a baseline PPO agent against a probabilistic-PPO "
        "variant that consumes a DeepAR-style uncertainty signal, with passive "
        "buy-and-hold, all-cash, and a rule-based trailing stop-loss policy as the "
        "three named comparators. Everything is driven from a single configuration "
        "file and a small set of scripts, so any reported number can be regenerated "
        "from the public repository in a single command sequence.",
    )

    add_heading(doc, "What has been built", 2)
    add_bullets(doc, [
        "A probabilistic forecaster (experiments/runners/run_probabilistic_agent.py): an LSTM trained with Gaussian negative log-likelihood that emits the mean and log variance of the next-step log return. The predictive standard deviation is min-max normalised across the test window into a unit-interval uncertainty score.",
        "An uncertainty-aware trading environment (experiments/common.py:StockEnv): action space [-1, 1] over a configurable max_trade_fraction of cash, with the trade size shrunk by (1 - uncertainty_level) and floored at min_trade_scale. When the uncertainty score exceeds the protocol quantile (default 0.80) the environment blocks new long-side trades but still allows exits. Reward is the per-step log of the portfolio-value ratio multiplied by 100 for numerical scale.",
        "A baseline PPO runner (experiments/runners/run_baseline.py) that uses the same environment without the uncertainty coordinate or the trade-size shrinkage, so the comparison against the probabilistic variant is genuinely controlled.",
        "A rule-based stop-loss runner (experiments/runners/run_rule_baselines.py) implementing 5 % and 10 % trailing-stop policies with a 20/50-day moving-average crossover for re-entry. This is the directly-measured non-AI comparator that the supervisor's previous feedback asked for.",
        "A benchmarks runner (experiments/runners/run_benchmarks.py) that evaluates passive buy-and-hold and all-cash on the same test window. These act as sanity checks on the metric definitions as much as competitors to beat.",
        "A single evaluation protocol (experiments/configs/dissertation_protocol.json) that fixes the splits (2009-2018 train / 2019-2021 validation / 2022-2025 test), the seeds [7, 19, 42] in the headline study, and the metric set, and is read by every script. This is the bit that actually makes the comparisons fair.",
        "A reporting layer: reports/builders/generate_dissertation_report.py for the markdown summary, reports/builders/build_supervisor_pack.py for the one-page chart, reports/builders/plot_dissertation_visuals.py for the detailed figures, and Dissertation_Walkthrough.ipynb for the embedded-output review notebook.",
    ])

    add_heading(doc, "Phase-0 to Phase-1 status table", 2)
    add_status_table(doc, [
        ("0.1 Environment + dependencies", "Done", "requirements.txt, SB3, PyTorch, gymnasium, yfinance"),
        ("0.2 PPO baseline on sample data", "Done", "phase0_examples/ppo_stock_trading_standalone.py"),
        ("0.3 DeepAR-style probabilistic example", "Done", "phase0_examples/deepar_style_example.py"),
        ("1.1 Shared protocol + metrics", "Done", "experiments/configs/dissertation_protocol.json, experiments/common.py"),
        ("1.2 Reproducible baseline / probabilistic / benchmark runners", "Done", "Three runners, seeded"),
        ("1.3 Dissertation report + supervisor pack", "Done", "reports/generated/"),
        ("1.4 Rule-based stop-loss comparator (5 % and 10 % variants)", "Done", "experiments/runners/run_rule_baselines.py"),
        ("1.5 Robustness on 70-ticker test universe (Phase-1 budget)", "Done", "Four-agent comparison on 70 tickers × 3 seeds × 10k steps; aggregate stats + per-ticker table in Section 5.5 of dissertation"),
        ("1.6 Walk-forward (out-of-time) on CPU-feasible subset", "Done", "4 tickers × 4 folds × 3 seeds × 10k steps = 96 trainings; in Section 6.4 of dissertation"),
        ("1.7 Extended seed-stability check on representative sub-universe", "Done", "8 tickers × 10 seeds × 50k steps = 80 trainings; in Section 5.5.1 of dissertation"),
        ("1.8 Phase-2 extended grid on full 70-ticker universe", "Scheduled", "GPU-only; orchestrator experiments/runners/run_extended_grid.py + notebook notebooks/extended_grid_colab.ipynb"),
    ])

    add_heading(doc, "Current results (mean across 3 seeds, test window 2022-2025)", 2)
    rows = build_results_rows()
    if rows:
        add_results_table(doc, rows)
    add_para(
        doc,
        "Reference figures: reports/generated/charts/final_value_comparison.png, "
        "equity_curve_comparison.png and uncertainty_signal.png.",
        italic=True,
    )

    add_heading(doc, "70-ticker test universe — Phase-1 robustness", 2)
    add_para(
        doc,
        "The Phase-1 robustness study runs the same four-agent comparison on a "
        "70-ticker diversified-equity test universe (41 single-name US large-cap "
        "equities spanning technology, payments and financial services, "
        "healthcare, consumer and industrials, plus 29 exchange-traded funds "
        "covering broad-market indices, sector SPDRs, dividend ETFs, thematic "
        "exposures and commodity funds) on the same 2022–2025 test window with "
        "the same metric definitions. The aggregate result is summarised below; "
        "the full per-ticker table is in Appendix B of the main dissertation.",
    )
    cs_table = doc.add_table(rows=1, cols=4)
    cs_table.style = "Light Grid Accent 1"
    cs_hdr = cs_table.rows[0].cells
    for i, h in enumerate(["Strategy", "Mean terminal value", "Mean Sharpe", "Mean Max-DD"]):
        cs_hdr[i].text = h
        for r in cs_hdr[i].paragraphs[0].runs:
            r.bold = True
    for label, final, sharpe, mdd in [
        ("Baseline PPO (no uncertainty)", "$989,430", "−0.23", "0.033"),
        ("Probabilistic PPO (this work)", "$1,998,817", "+0.60", "0.225"),
        ("Manual 5 % trailing stop", "$1,531,163", "+0.36", "0.305"),
        ("Passive buy-and-hold", "$2,099,838", "+0.54", "0.370"),
    ]:
        row = cs_table.add_row().cells
        row[0].text = label
        row[1].text = final
        row[2].text = sharpe
        row[3].text = mdd
        if "Probabilistic" in label:
            for c in row:
                for p in c.paragraphs:
                    for r in p.runs:
                        r.bold = True
    add_para(doc, "Headline findings on the 70-ticker test universe:", bold=True)
    add_bullets(doc, [
        "Drawdown reduced versus passive buy-and-hold on 70 of 70 tickers (100 % of the universe), with an average reduction of 14.5 percentage points (mean drawdown cut by 39 % in relative terms — from 37.0 % to 22.5 %). This is the strongest single number in the dissertation.",
        "Probabilistic agent beat the manually-tuned 5 % trailing stop on 61 of 70 tickers (87 %) in terminal value, and on essentially every ticker in Sharpe ratio — the empirical answer to the previous-meeting question on whether the AI agent beats a manually-tuned stop-loss alternative.",
        "Cost in mean terminal value versus passive buy-and-hold: roughly 5 % give-up in mean upside in exchange for the 39 % reduction in mean drawdown above. This is exactly the trade an institutional drawdown-mandated investor runs every quarter.",
        "Where the agent loses (45 of 70 tickers in terminal value, all winning on drawdown), the losses cluster in two diagnosable regimes: persistent, low-uncertainty bull-market trends in single names (NVDA, AVGO, LLY) where the uncertainty-guard's caution costs the right tail, and very-low-drawdown defensives (JNJ, MCD, SCHD, GLD) where there is essentially nothing for a drawdown overlay to add. Sector-aware uncertainty-quantile calibration is the targeted Phase-2 fix.",
    ])

    add_heading(doc, "How to read these numbers", 2)
    add_bullets(doc, [
        "The headline criterion is the joint of Sharpe ratio and drawdown control, not either half alone. Meeting either half on its own is trivial: an all-cash policy achieves perfect preservation with zero return, and a return-only policy ignores the constraint entirely. On the 70-ticker test universe the probabilistic agent meets the joint constraint: it controls drawdown on 100 % of the universe (mean drawdown 22.5 % vs buy-and-hold's 37.0 %) and earns a higher mean Sharpe than passive buy-and-hold (+0.60 vs +0.54). The baseline PPO meets neither half: it ends roughly where it started with a slightly negative Sharpe.",
        "The manually-tuned trailing-stop comparator (5 % stop with 20/50-day moving-average re-entry) is the directly measured manual alternative. The probabilistic agent beats it on 87 % of the 70-ticker universe in terminal value and on essentially every ticker in Sharpe. This is a directly-measured rather than asserted answer to the previous-meeting question on whether the AI agent beats a manually-tuned stop-loss alternative.",
        "Max drawdown on the baseline looks small only because the baseline barely compounds in the first place. Both terminal preservation and path preservation (1 − MDD) are reported in the dissertation so the reader can apply whichever definition matches their mandate.",
        "These numbers are provisional Phase-1 evidence. The Phase-2 extended grid (10 seeds × 50 000 timesteps × 4 walk-forward folds × 16 bootstrap paths × 70 tickers) on the Colab GPU runtime will tighten the seed-variability bands and provide out-of-time confirmation across the full universe.",
    ])

    add_heading(doc, "Reproducibility", 2)
    p = doc.add_paragraph()
    run = p.add_run(
        "python experiments/runners/run_baseline.py\n"
        "python experiments/runners/run_probabilistic_agent.py\n"
        "python experiments/runners/run_benchmarks.py\n"
        "python experiments/runners/run_rule_baselines.py\n"
        "python reports/builders/generate_dissertation_report.py\n"
        "python reports/builders/build_supervisor_pack.py\n"
        "python reports/builders/plot_dissertation_visuals.py"
    )
    run.font.name = "Consolas"
    run.font.size = Pt(10)
    add_para(
        doc,
        "Artifacts land in experiments/results/ and reports/generated/. The full "
        "source is on GitHub at TheFinix13/Dissertation_Sample_Project, with the "
        "walkthrough notebook (Dissertation_Walkthrough.ipynb) as the single entry "
        "point for someone reading the project for the first time.",
    )

    page_break(doc)

    # ----- Future plan -----
    add_heading(doc, "Future plan", 1)
    add_para(
        doc,
        "Progress against the previous plan. The May 2026 milestones in the original "
        "plan have been completed: the dissertation has been reframed around drawdown-"
        "constrained risk-adjusted return; a finance and risk-management background "
        "section has been added; the rule-based stop-loss comparator is checked in and "
        "reported alongside the AI agents; the test universe has been expanded from "
        "single-index SPY to a 70-ticker diversified-equity universe; and the extended "
        "seed-stability check has been run on a representative sub-universe. The plan "
        "below covers what remains.",
    )
    add_para(
        doc,
        "Each scheduled task is tied to a milestone with a target date. Milestones are "
        "tied to objectives O1 to O4. The submission-critical path is the backtest and "
        "walk-forward evidence; live execution sits outside that path and is treated as "
        "a stretch goal (see the August 2026 row).",
    )
    add_plan_table(doc, [
        (
            "June 2026 (4 weeks)",
            "Phase-2 extended grid on the full 70-ticker universe at extended "
            "budget (10 seeds × 50 000 timesteps × 4 walk-forward folds × 16 "
            "bootstrap paths) on Colab GPU runtime. Sector-aware uncertainty-"
            "quantile calibration (replace the single global threshold with a "
            "per-sector or per-regime threshold). Begin Chapter 2 (Background) "
            "and Chapter 3 (Methodology) full drafts.",
            "M1: full 70-ticker × 4-fold × 10-seed × 50k-step extended grid "
            "(mid-June). M2: sector-aware calibration ablation + Chapter 2 and "
            "Chapter 3 first drafts (end of June).",
        ),
        (
            "July 2026 (4-6 weeks)",
            "Sensitivity sweep on the uncertainty quantile threshold, the minimum "
            "trade-size scale, and the maximum trade fraction. Block-bootstrap "
            "data augmentation (Politis and Romano, 1994) to expand the effective "
            "training set. Lock the final headline results table. Draft Chapter 5 "
            "(Results) and Chapter 1 (Introduction).",
            "M3: sensitivity and bootstrap results locked (mid-July). M4: Chapters "
            "1, 2, 3 and 5 first drafts (end of July).",
        ),
        (
            "August 2026 (4 weeks)",
            "Polish phase. Write Chapter 6 (Discussion) and Chapter 7 (Conclusion). "
            "Polish figures and tables, integrate supervisor feedback on the full "
            "draft, and finalise the dissertation. Code changes from this point are "
            "bug-fix only. Stretch goal: if time and a working brokerage account "
            "permit, run a two-week paper-trading shadow run via the Alpaca API and "
            "report the live profit-and-loss as an out-of-sample case study; if it "
            "does not happen the dissertation rests on the backtest and walk-forward "
            "evidence and the live run is recorded as post-submission work in the "
            "real-world deployment roadmap (see the live/ directory in the "
            "repository).",
            "M5: full draft to supervisor (mid-August). M6: submission-ready version "
            "(end of August). M7 (stretch): paper-trading PnL added to results "
            "chapter (third week of August), only if the shadow run is in scope.",
        ),
        (
            "September 2026",
            "Submit by 1 September 2026. Viva preparation: slide deck (no more than "
            "twelve slides, no more than twenty minutes per the project handbook), "
            "demo of the reproducible pipeline, pre-emptive question and answer "
            "rehearsal using reports/templates/viva_qa_notes.md.",
            "M8: viva-ready presentation and demo by viva date.",
        ),
    ])

    add_heading(doc, "Risks and mitigations", 2)
    add_bullets(doc, [
        "Compute time. Phase-1 runs are CPU-friendly (10 000 PPO timesteps, three seeds). The full Phase-2 grid is larger but still tractable on a Google Colab T4 GPU runtime, and the runners are designed to lift onto Colab without code changes. Partial-grid results will be accepted for any interim deliverable.",
        "Data-API drift. yfinance occasionally changes its column shape. The _close_1d helper used by every runner already normalises this, and the protocol pins explicit dates so a re-pull stays comparable.",
        "Result fragility. The Phase-1 numbers may move under the full 70-ticker, walk-forward and ablation work. To guard against over-claiming, results will be reported as median and inter-quartile range across ten seeds and across tickers, evaluated on multiple sliding test windows (walk-forward) rather than a single window, and any case where the probabilistic variant fails to beat the rule-based stop-loss comparator or buy-and-hold will be called out explicitly.",
        "Paper-trading dependency (stretch goal only). The Alpaca shadow run is a stretch goal that does not gate the dissertation. If the brokerage account, the API or the time available does not support a clean two-week run during August, the dissertation rests on the backtest and walk-forward evidence and the shadow run is moved into the real-world deployment roadmap as post-submission work.",
    ])

    page_break(doc)

    # ----- Extenuating circumstances -----
    add_heading(doc, "Extenuating circumstances", 1)
    add_para(
        doc,
        "[Student to fill in. Either record \"None to declare\" or describe and "
        "indicate that the personal tutor and student-support services have been "
        "informed. Do not include medical detail in this document.]",
        italic=True,
    )

    # ----- Self-tick -----
    add_heading(doc, "Indicative project hours and progress", 1)
    add_para(
        doc,
        "Self-assessment of the first one hundred hours allocated to the project. "
        "The student should tick exactly one of the following:",
    )
    add_bullets(doc, [
        "[ ]  The work has exceeded the first 100 hours of time allocated.",
        "[X] The work has sufficiently met the first 100 hours.",
        "[ ]  The majority of the first 100 hours have been completed but some time has been lost and will be made up.",
        "[ ]  Engagement in the project has been insufficient and progress is of concern.",
    ])
    add_para(
        doc,
        "Justification: the reproducible Phase-0 and Phase-1 pipeline, the protocol "
        "document, the baseline and probabilistic agents, the benchmarks, the "
        "rule-based stop-loss comparator and the generated supervisor pack together "
        "support the second tick above.",
        italic=True,
    )

    page_break(doc)

    # ----- Supervisor section -----
    add_heading(doc, "Supervisor's assessment", 1)
    add_para(
        doc,
        "The boxes below are reserved for the supervisor's written feedback. The "
        "student-completed sections of this document are intended to give the "
        "supervisor enough material to assess each item.",
        italic=True,
    )
    add_supervisor_box(doc, "Comments on the project plan and on the student's progress to date:")
    add_supervisor_box(doc, "Comments on the literature review and the framing of the problem:")
    add_supervisor_box(doc, "Comments on the technical progress and the experimental protocol:")
    add_supervisor_box(doc, "Recommendations for the remainder of the project:")
    add_supervisor_box(doc, "Other comments (optional):")

    # ----- Save -----
    out = EXPORTS / "InterimReview.docx"
    doc.save(out)
    print(f"Wrote: {out}")
    return out


if __name__ == "__main__":
    build()
