"""Dựng infographic 1 trang (1920×1080, PDF) cho bài nộp vòng 2.

Mọi con số đọc từ reports/*.json và outputs/isb_limits.csv (sinh bởi các lệnh trong README),
nên chạy lại sau khi đổi dataset là số liệu tự cập nhật.

    python -m creditforyou limits --all --with-truth
    python -m creditforyou economics
    python scripts/build_infographic.py            # -> docs/CreditForYou_infographic.pdf
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
REPORTS = ROOT / "reports"
DOCS = ROOT / "docs"
BROWSERS = [r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
            r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
            r"C:\Program Files\Google\Chrome\Application\chrome.exe"]

NAVY, ACCENT, GREY, LIGHT = "#12305c", "#e07a1f", "#9aa3ae", "#eef2f7"
GOOD, BAD, INK, MUTED = "#1f8a5b", "#c0392b", "#16202c", "#5b6573"


def load():
    lim = pd.read_csv(ROOT / "outputs" / "isb_limits.csv")
    summary = json.loads((REPORTS / "pos_limit_summary.json").read_text(encoding="utf-8"))
    eco = json.loads((REPORTS / "pos_economics.json").read_text(encoding="utf-8"))
    inc = json.loads((REPORTS / "income_metrics.json").read_text(encoding="utf-8"))
    model = json.loads((REPORTS / "credit_model_metrics.json").read_text(encoding="utf-8"))
    quality = json.loads((REPORTS / "data_quality.json").read_text(encoding="utf-8"))

    has = lim["pos_limit"] > 0
    target = lim["profile"].isin(["freelancer", "entrepreneur"]) | lim["main_income_type"].eq("self_employed")
    thin = ~lim["has_credit_history"].astype(bool)
    trad = lim["main_income_type"].isin(["salary", "pension"]) & lim["has_credit_history"].astype(bool)
    stab = lim.groupby("profile")["income_stability"].median()
    d = {
        "n": len(lim),
        "before_after": [
            ("Freelancers & business owners", int(target.sum()), trad[target].mean(), has[target].mean()),
            ("First-time borrowers", int(thin.sum()), trad[thin].mean(), has[thin].mean()),
            ("All customers", len(lim), trad.mean(), has.mean()),
        ],
        "status": summary["trang_thai_han_muc"],
        "median_limit": summary["han_muc_trung_vi_(khach_co_han_muc)"],
        "offline_share": summary["ho_so_xem_xet_ngoai_gio"] / len(lim),
        "stability": {"Freelancer": stab.get("freelancer"), "Business owner": stab.get("entrepreneur"),
                      "Salaried manager": stab.get("manager")},
        "eco": eco,
        "income_mdape": inc["median_absolute_percentage_error"],
        "income_mdape_old": 0.088,
        "auc": model["metrics_test"][model["model_name"]]["roc_auc"],
        "lc_tier_pd": eco["kich_ban"].get("theo_PD_Lending_Club", {}).get("pd_theo_nhom", {}),
        "rows": quality["rows"],
        "q": {f["issue"].split(" {")[0]: f for f in quality["findings"]},
    }
    d["mult"] = d["before_after"][0][3] / d["before_after"][0][2]
    # Dòng chảy khách: mới được phục vụ / bị loại so với quy tắc truyền thống
    d["flows"] = {"new": int((has & ~trad).sum()), "lost": int((trad & ~has).sum()),
                  "net": int(has.sum() - trad.sum()), "trad": int(trad.sum()), "cfy": int(has.sum())}

    def wilson(k, n, z=1.96):
        p = k / n
        den = 1 + z * z / n
        c = (p + z * z / (2 * n)) / den
        h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / den
        return c - h, c + h
    d["target_ci"] = wilson(int(has[target].sum()), int(target.sum()))
    d["target_n"] = int(target.sum())
    cfg = json.loads((ROOT / "config" / "scoring.json").read_text(encoding="utf-8"))
    d["bench"] = cfg["economics"]["benchmarks"]
    d["cfg"] = cfg
    from creditforyou.report import case_mix_view
    from creditforyou.economics import average_balance_factor
    d["mix"] = case_mix_view(lim)
    # Bài toán khoản vay nhỏ: doanh thu của 1 khoản 5 triệu VND, 6 tháng, lãi nhóm B (12%) + phí đối tác
    rate, mdr, ticket = 0.12, cfg["economics"]["merchant_discount_rate"], 5_000_000
    per_vnd = average_balance_factor(6, rate) * 0.5 * rate + mdr
    d["small"] = {"ticket": ticket, "revenue": ticket * per_vnd, "breakeven_ticket": cfg["economics"]["cost_per_manual_review"] / per_vnd,
                  "review_cost": cfg["economics"]["cost_per_manual_review"], "auto_cost": cfg["economics"]["cost_per_loan_automated"]}
    return d


# --------------------------------------------------------------------------- SVG charts

def svg_segments(mix, w=570, h=160):
    keys = ["gig_platform", "online_merchant", "first_time", "salaried_no_credit"]
    left, bar_h, gap_in, gap = 222, 12, 2, 8
    scale = (w - left - 50)
    out = [f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg">']
    y = 4
    for k in keys:
        r = mix["segments"][k]
        out.append(f'<text x="0" y="{y + bar_h + 1}" class="ax">{r["label"]}</text>')
        out.append(f'<text x="0" y="{y + 2 * bar_h + 4}" class="axs">{r["case_weight"]:.1%} of applications · n = {r["n_sample"]}</text>')
        if r["traditional"] and r["traditional"] > 0:
            out.append(f'<rect x="{left}" y="{y}" width="{max(r["traditional"] * scale, 2):.1f}" height="{bar_h}" rx="3" fill="{GREY}"/>')
            out.append(f'<text x="{left + r["traditional"] * scale + 6:.1f}" y="{y + bar_h - 2}" class="val" fill="{MUTED}">{r["traditional"]:.0%}</text>')
        else:
            out.append(f'<text x="{left}" y="{y + bar_h - 2}" class="axs">no proxy baseline (rule needs credit history)</text>')
        yy = y + bar_h + gap_in
        out.append(f'<rect x="{left}" y="{yy}" width="{r["creditforyou"] * scale:.1f}" height="{bar_h}" rx="3" fill="{ACCENT}"/>')
        out.append(f'<text x="{left + r["creditforyou"] * scale + 6:.1f}" y="{yy + bar_h - 2}" class="val">{r["creditforyou"]:.0%}</text>')
        y += 2 * bar_h + gap_in + gap
    out.append(f'<rect x="{left}" y="{h - 13}" width="11" height="11" fill="{GREY}"/>'
               f'<text x="{left + 16}" y="{h - 3}" class="axs">Proxy of today&#39;s rule</text>'
               f'<rect x="{left + 150}" y="{h - 13}" width="11" height="11" fill="{ACCENT}"/>'
               f'<text x="{left + 166}" y="{h - 3}" class="axs">CreditForYou limit</text>')
    out.append("</svg>")
    return "".join(out)


def svg_profit(scen, w=570, h=160):
    names = [("co_so", "Base case"), ("cang_thang_x2", "Default ×2"), ("nghiem_trong_x3", "Default ×3"),
             ("theo_PD_Lending_Club", "Lending Club PD")]
    vals = [(lbl, scen[k]["loi_nhuan"], scen[k]["RoC"], scen[k]["pd_binh_quan_theo_du_no"]) for k, lbl in names if k in scen]
    top, label_zone = 34, 44
    vmax = max(max(v for _, v, _, _ in vals), 0)
    vmin = min(min(v for _, v, _, _ in vals), 0)
    plot_h = h - top - label_zone
    span = plot_h / (vmax - vmin)
    zero = top + vmax * span
    bw, step, x0 = 84, (w - 20) / len(vals), 20
    out = [f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg">']
    out.append(f'<text x="0" y="14" class="axs">Annual profit, million VND, by default scenario (291 simulated customers with a limit)</text>')
    out.append(f'<line x1="10" x2="{w}" y1="{zero:.1f}" y2="{zero:.1f}" stroke="{MUTED}" stroke-width="1.5"/>')
    for i, (lbl, v, roc, pdv) in enumerate(vals):
        cx = x0 + i * step + step / 2
        hgt = max(abs(v) * span, 2)
        y = zero - hgt if v >= 0 else zero
        color = GOOD if v >= 0 else BAD
        out.append(f'<rect x="{cx - bw / 2:.1f}" y="{y:.1f}" width="{bw}" height="{hgt:.1f}" rx="3" fill="{color}"/>')
        if v >= 0:
            out.append(f'<text x="{cx:.1f}" y="{y - 6:.1f}" class="val" text-anchor="middle" fill="{color}">{v / 1e6:+,.0f}M</text>')
        elif hgt > 26:
            out.append(f'<text x="{cx:.1f}" y="{y + hgt - 8:.1f}" class="val" text-anchor="middle" fill="#fff">{v / 1e6:+,.0f}M</text>')
        else:
            out.append(f'<text x="{cx + bw / 2 + 4:.1f}" y="{zero + 16:.1f}" class="val" fill="{color}">{v / 1e6:+,.0f}M</text>')
        out.append(f'<text x="{cx:.1f}" y="{h - 24}" class="ax" text-anchor="middle">{lbl}</text>')
        out.append(f'<text x="{cx:.1f}" y="{h - 6}" class="axs" text-anchor="middle">PD {pdv:.1%} · RoC {roc:.0%}</text>')
    out.append("</svg>")
    return "".join(out)


def svg_status(status, n, w=780, h=62):
    order = [("PREAPPROVED", "Full limit", NAVY), ("PREAPPROVED_REDUCED", "Reduced limit", "#4f7cba"),
             ("OFFLINE_REVIEW", "Offline review", ACCENT), ("NOT_ELIGIBLE", "Not eligible", GREY)]
    out = [f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg">']
    x = 0
    for key, label, color in order:
        c = status.get(key, 0)
        bw = w * c / n
        out.append(f'<rect x="{x:.1f}" y="0" width="{bw:.1f}" height="30" fill="{color}"/>')
        if bw > 44:
            out.append(f'<text x="{x + bw / 2:.1f}" y="21" class="val" fill="#fff" text-anchor="middle">{c}</text>')
        x += bw
    lx = 0
    for key, label, color in order:
        c = status.get(key, 0)
        text = f"{label} {c} ({c / n:.0%})"
        out.append(f'<rect x="{lx}" y="44" width="12" height="12" fill="{color}"/>'
                   f'<text x="{lx + 17}" y="55" class="axs">{text}</text>')
        lx += 18 + len(text) * 7.4 + 22
    out.append("</svg>")
    return "".join(out)


def svg_stability(stab, w=400, h=96):
    out = [f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg">']
    left, scale = 150, (w - 150 - 50)
    for i, (label, v) in enumerate(stab.items()):
        y = 4 + i * 30
        color = ACCENT if label != "Salaried manager" else GREY
        out.append(f'<text x="0" y="{y + 16}" class="ax">{label}</text>')
        out.append(f'<rect x="{left}" y="{y}" width="{v * scale:.1f}" height="20" rx="3" fill="{color}"/>')
        out.append(f'<text x="{left + v * scale + 6:.1f}" y="{y + 16}" class="val">{v:.2f}</text>')
    out.append("</svg>")
    return "".join(out)


# --------------------------------------------------------------------------- page

def build_html(d):
    e = d["eco"]
    base = e["kich_ban"]["co_so"]
    stress = e["kich_ban"]["cang_thang_x2"]
    severe = e["kich_ban"]["nghiem_trong_x3"]
    be = e["pd_hoa_von_theo_nhom"]
    sens = e["do_nhay"]
    ops = e["van_hanh"]
    fl = d["flows"]
    bm = d["bench"]
    mix = d["mix"]
    sm = d["small"]
    pl = d["cfg"]["pos_limit"]
    caps = pl["tier_caps"]
    seg = mix["segments"]
    tgt_before, tgt_after = d["before_after"][0][2], d["before_after"][0][3]
    ci_lo, ci_hi = d["target_ci"]
    q = d["q"]
    sal = q["is_salary=True nhưng không phải lương"]
    tr = q["chuyển khoản giữa các tài khoản của chính khách"]
    mort = q["khoản vay mua nhà trong nợ đang có"]
    bal = q["số dư sau giao dịch không liên tục (số dư trước + số tiền ≠ số dư sau)"]
    lc = d["lc_tier_pd"]
    lcpd = bm["lending_club_pd6m_by_grade"]
    sims = json.loads((REPORTS / "pos_limit_summary.json").read_text(encoding="utf-8"))["gia_lap_tai_quay"]
    sim_key = min(sims, key=lambda k: abs(float(k) - 5_000_000))
    conv = sims[sim_key]["tat_ca"].get("APPROVE", 0) + sims[sim_key]["tat_ca"].get("COUNTER_OFFER", 0)
    be_pd = be["toan_danh_muc"]
    M = lambda v: f"{v / 1e6:,.0f}M"  # noqa: E731
    k_ = lambda v: f"{v / 1e3:,.0f}k"  # noqa: E731

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>CreditForYou</title>
<style>
@page {{ size: 1920px 1080px; margin: 0; }}
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
html, body {{ width: 1920px; height: 1080px; overflow: hidden; }}
body {{ font-family: "Noto Sans", "Segoe UI", Arial, sans-serif; color: {INK}; background: #f4f7fb;
       -webkit-print-color-adjust: exact; print-color-adjust: exact; }}
.page {{ width: 1920px; height: 1080px; padding: 20px 30px 16px; display: grid;
        grid-template-rows: 156px 612px 1fr; gap: 12px; }}
header {{ display: grid; grid-template-columns: 1fr 940px; gap: 20px; align-items: center;
         background: {NAVY}; color: #fff; border-radius: 14px; padding: 12px 20px 12px 26px; }}
.kicker {{ font-size: 13px; letter-spacing: .07em; text-transform: uppercase; opacity: .78; }}
h1 {{ font-size: 30px; font-weight: 700; line-height: 1.12; margin: 3px 0 4px; }}
.sub {{ font-size: 19px; font-weight: 600; color: #ffc27d; line-height: 1.25; }}
.kpis {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 9px; }}
.kpi {{ background: rgba(255,255,255,.11); border-radius: 10px; padding: 9px 12px; height: 128px; position: relative; }}
.kpi b {{ display: block; font-size: 27px; line-height: 1.1; color: #ffc27d; font-weight: 700; white-space: nowrap; }}
.kpi span {{ font-size: 14px; line-height: 1.25; display: block; margin-top: 4px; }}
.kpi i {{ position: absolute; right: 10px; bottom: 7px; font-size: 11.5px; font-style: normal; opacity: .7; }}
.main {{ display: grid; grid-template-columns: 430px 1fr 600px; gap: 12px; min-height: 0; }}
.bottom {{ display: grid; grid-template-columns: 650px 1fr 640px; gap: 12px; min-height: 0; }}
.card {{ background: #fff; border-radius: 12px; padding: 11px 16px; box-shadow: 0 1px 3px rgba(18,48,92,.10);
        border-top: 5px solid {NAVY}; min-height: 0; overflow: hidden; }}
.card.accent {{ border-top-color: {ACCENT}; }}
.step {{ font-size: 13.5px; font-weight: 700; color: {ACCENT}; letter-spacing: .06em; text-transform: uppercase; }}
h2 {{ font-size: 22px; line-height: 1.15; margin: 2px 0 7px; color: {NAVY}; }}
h3 {{ font-size: 15.5px; margin: 6px 0 4px; color: {NAVY}; }}
p, li {{ font-size: 15px; line-height: 1.3; }}
ul {{ padding-left: 18px; }}
li {{ margin-bottom: 2px; }}
.small, .small li {{ font-size: 14px; line-height: 1.3; }}
.muted {{ color: {MUTED}; }}
.chain {{ display: grid; gap: 3px; margin: 3px 0 6px; }}
.box {{ border-radius: 8px; padding: 6px 11px; font-size: 14.5px; line-height: 1.25; }}
.box.grey {{ background: {LIGHT}; }}
.box.red {{ background: #fbe9e7; color: #7d241b; }}
.box.navy {{ background: {NAVY}; color: #fff; font-weight: 600; }}
.arrow {{ text-align: center; color: {MUTED}; font-size: 11px; line-height: 1; }}
.math {{ background: #fbe9e7; border-left: 4px solid {BAD}; border-radius: 0 8px 8px 0; padding: 6px 10px;
        font-size: 14px; line-height: 1.3; margin-top: 6px; }}
.math b.big {{ font-size: 16px; color: {BAD}; }}
.lever {{ background: #fff3e6; border-left: 4px solid {ACCENT}; border-radius: 0 8px 8px 0; padding: 6px 10px;
         font-size: 14px; line-height: 1.3; margin-top: 6px; }}
.fa {{ display: flex; align-items: center; justify-content: center; color: {ACCENT}; font-size: 18px; font-weight: 700; }}
.arch {{ display: grid; grid-template-columns: 1fr 14px 1.25fr 14px 0.82fr 14px 1.2fr; margin: 2px 0 6px; }}
.lane {{ background: {LIGHT}; border-radius: 10px; padding: 7px 9px; }}
.lane.act {{ background: #fff3e6; border: 2px solid {ACCENT}; }}
.lane .when {{ font-size: 11.5px; color: {ACCENT}; font-weight: 700; text-transform: uppercase; letter-spacing: .05em; }}
.lane h4 {{ font-size: 15px; color: {NAVY}; margin: 1px 0 3px; }}
.lane li, .lane p {{ font-size: 13.4px; line-height: 1.25; }}
.lane ul {{ padding-left: 14px; }}
.lane .muted {{ font-size: 12.5px; margin-top: 3px; }}
.rule {{ font-size: 13.4px; line-height: 1.27; background: {LIGHT}; border-radius: 8px; padding: 5px 9px; margin-top: 5px; }}
.impact {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; margin-top: 5px; }}
.impact div {{ border-radius: 8px; padding: 6px 9px; font-size: 13.4px; line-height: 1.26; }}
.impact b.t {{ display: block; font-size: 14px; margin-bottom: 1px; }}
.i1 {{ background: #e8f1fb; }} .i1 b.t {{ color: {NAVY}; }}
.i2 {{ background: #e3f4ec; }} .i2 b.t {{ color: {GOOD}; }}
.i3 {{ background: #eef6e6; }} .i3 b.t {{ color: #3d7a1c; }}
.mlnote {{ font-size: 13.4px; line-height: 1.27; margin-top: 5px; }}
.risk {{ background: #fbe9e7; border-left: 4px solid {BAD}; border-radius: 0 8px 8px 0; padding: 6px 10px; margin-top: 5px;
        font-size: 13.6px; line-height: 1.28; }}
.ci {{ font-size: 12.8px; color: {MUTED}; margin: 1px 0 2px; line-height: 1.25; }}
svg text.ax {{ font: 600 13.5px "Noto Sans", "Segoe UI", Arial; fill: {INK}; }}
svg text.axs {{ font: 12px "Noto Sans", "Segoe UI", Arial; fill: {MUTED}; }}
svg text.val {{ font: 700 14px "Noto Sans", "Segoe UI", Arial; }}
.road {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 9px; margin-top: 6px; }}
.phase {{ border-left: 5px solid {NAVY}; background: {LIGHT}; border-radius: 0 8px 8px 0; padding: 8px 10px; }}
.phase b {{ font-size: 15.5px; color: {NAVY}; }}
.phase p {{ font-size: 14.5px; line-height: 1.3; margin-top: 3px; }}
.gate {{ font-size: 14px; color: #a45a12; font-weight: 700; margin-top: 6px; line-height: 1.25; }}
table {{ width: 100%; border-collapse: collapse; margin-top: 3px; }}
td, th {{ font-size: 14px; padding: 3px 5px; border-bottom: 1px solid #e3e8ef; text-align: left; line-height: 1.2; }}
th {{ color: {MUTED}; font-weight: 600; font-size: 12.5px; }}
td.n {{ text-align: right; font-weight: 700; white-space: nowrap; color: {NAVY}; }}
.data p {{ font-size: 14px; line-height: 1.28; }}
.src {{ color: {MUTED}; font-size: 12.8px; }}
.sim {{ display: inline-block; background: #ffe9d2; color: #a45a12; font-weight: 700; font-size: 11.5px; border-radius: 99px; padding: 0 7px; margin-left: 6px; letter-spacing: .03em; }}
</style></head>
<body><div class="page">

<header>
  <div>
    <div class="kicker">HLBVN × UEH.ISB Business Challenge 2026 · Round 2 · Point-of-purchase lending</div>
    <h1>CreditForYou: instant checkout credit for freelancers, gig workers &amp; online sellers</h1>
    <div class="sub">Cash-flow rules decide today · ML learns from real repayments to sharpen limits tomorrow</div>
  </div>
  <div class="kpis">
    <div class="kpi"><b>1.8–4.6 days → instant</b><span>decision turnaround (case baseline, Exhibit 2) → pre-computed limit lookup, target &lt; 1 s</span></div>
    <div class="kpi"><b>{k_(sm['review_cost'])} → {k_(sm['auto_cost'])} VND</b><span>cost per decision: case benchmark → automated (team assumption)</span></div>
    <div class="kpi"><b>{tgt_before:.0%} → {tgt_after:.0%}</b><span>gig workers &amp; online merchants with a limit vs proxy rule</span><i>simulated · n = {d['target_n']}</i></div>
    <div class="kpi"><b>{be_pd:.1%}</b><span>break-even 6-month default = stop-rule; RoC {base['RoC']:.0%} in base case</span><i>simulated</i></div>
  </div>
</header>

<section class="main">
  <div class="card">
    <div class="step">① Problem diagnosis</div>
    <h2>The checkout decision is welded to manual underwriting</h2>
    <div class="chain">
      <div class="box grey">No payslip · short or no bureau file · income arrives via platforms, e-wallets, online sales</div>
      <div class="arrow">▼</div>
      <div class="box navy">Centralised manual review: documents, CIC check, credit officer</div>
      <div class="arrow">▼</div>
      <div class="box red"><b>1.8–4.6 business days</b> → the purchase moment is gone<br><b>Decline</b> → lost sale for bank &amp; merchant</div>
    </div>
    <div class="math"><b>The review costs more than a small loan earns:</b> a {M(sm['ticket'])} VND, 6-month loan at 12% earns ≈ <b class="big">{k_(sm['revenue'])} VND</b>
    (interest + 2% merchant fee) vs a <b>{k_(sm['review_cost'])} VND</b> review (Exhibit 2). Review only pays above ≈ <b>{sm['breakeven_ticket'] / 1e6:.1f}M VND</b>, yet loans start at 1.6M.</div>
    <div class="lever"><b>Highest leverage:</b> a faster review still costs 380k and keeps customers waiting; collections act only after a "yes".
    Taking the review out of checkout fixes speed <i>and</i> unit cost at once.</div>
    <h3>Root cause: income is real but invisible to a document review (stability, 1 − CV)</h3>
    {svg_stability(d['stability'])}
  </div>

  <div class="card accent">
    <div class="step">② Solution concept · data-to-decision architecture</div>
    <h2>Score cash-flow in advance; the till only checks the limit</h2>
    <div class="arch">
      <div class="lane"><div class="when">1 · Data inputs</div><h4>What we read</h4>
        <ul>
          <li>12 months of CASA transactions</li>
          <li>Active loan payments</li>
          <li>Age, tenure, segment</li>
          <li>Basket value at checkout</li>
        </ul>
        <p class="muted">Phase 2: platform / e-wallet data with consent. No payslip needed.</p></div>
      <div class="fa">▶</div>
      <div class="lane"><div class="when">2 · Logic · nightly batch</div><h4>How we decide</h4>
        <ul>
          <li>Clean: drop own transfers, refunds, disbursements</li>
          <li>Income by source, gross→net (0.89 / 0.53)</li>
          <li>Capacity = min(50% disposable, 20% income, 50% income − debt)</li>
          <li>Tier A–D + guardrails</li>
          <li>Limit = min(capacity × annuity, caps)</li>
        </ul></div>
      <div class="fa">▶</div>
      <div class="lane"><div class="when">3 · Outputs</div><h4>What we produce</h4>
        <ul>
          <li>Limit + status</li>
          <li>Reason codes</li>
          <li>Tier &amp; APR band</li>
          <li>Audit trail</li>
        </ul></div>
      <div class="fa">▶</div>
      <div class="lane act"><div class="when">4 · Business actions</div><h4>What the bank does</h4>
        <ul>
          <li><b>Till:</b> real-time eKYC &amp; fraud check, then approve / counter-offer / decline</li>
          <li><b>Targeting:</b> offers shown only to pre-approved customers</li>
          <li>3 on-time → +25%; 30+ DPD → freeze</li>
          <li><b>ML:</b> repayment labels train a default model</li>
        </ul></div>
    </div>
    <h3>Output on {d['n']} simulated customers <span class="sim">SIMULATED</span> · median limit {M(d['median_limit'])} VND · caps A {M(caps['A'])} / B {M(caps['B'])} / C {M(caps['C'])} / new {M(pl['thin_file_cap'])} · tenor 3–12 months</h3>
    {svg_status(d['status'], d['n'])}
    <div class="rule"><b>Guardrails only tighten:</b> reject if debt service &gt; 70%, spending &gt; 120% or cash-flow deficit; limit ×0.5 if tier C, debt &gt; 45% or spending &gt; 90%.
    <b>Salaried with credit</b> ({seg['salaried_with_credit']['case_weight']:.1%} of mix) keep today's bureau terms via STP; cash-flow adds an affordability flag.</div>
    <div class="impact">
      <div class="i1"><b class="t">Community impact</b>Thin-file workers get a first formal credit line instead of informal credit; repaying grows the limit and a credit history.</div>
      <div class="i2"><b class="t">Financially sustainable</b>Profitable while default &lt; {be_pd:.1%} (RoC {base['RoC']:.0%}); stop-rule; rules in a config the credit committee owns.</div>
      <div class="i3"><b class="t">Environmentally light</b>Paperless, no documents to scan or courier; one nightly CPU batch on data the bank already holds.</div>
    </div>
    <p class="mlnote"><b>Why ML is not the decision-maker yet:</b> no POP default labels exist. A Lending Club PD model (300k loans, AUC {d['auc']:.2f})
    mis-ranks tiers (B {lc.get('B', 0):.1%} &gt; C {lc.get('C', 0):.1%}), so it runs as a challenger until real labels beat the rules.</p>
  </div>

  <div class="card">
    <div class="step">③ Intervention justification · why it works and pays</div>
    <h2>3× more gig workers &amp; online merchants served, profitable while default &lt; {be_pd:.1%}</h2>
    {svg_segments(mix)}
    <div class="ci">Reweighted to the Exhibit 2 mix: instant limits for <b>{mix['case_weighted_all']['creditforyou']:.0%}</b> of inbound applications.
    Gig + MSME n = {d['target_n']} (95% CI {ci_lo:.0%}–{ci_hi:.0%}): re-validate in the backtest.</div>
    {svg_profit(e['kich_ban'])}
    <ul class="small" style="margin-top:2px">
      <li><b>Speed &amp; cost:</b> 1.8–4.6 days, 380k/file → instant, ≈ 50k; {ops['ho_so_khong_con_phai_nguoi_duyet']} files skip review.</li>
      <li><b>Reach:</b> +{fl['new']} newly served, −{fl['lost']} over-indebted (mortgage-heavy) declined → net +{fl['net']}.</li>
      <li><b>Value:</b> profit {M(base['loi_nhuan'])} VND/yr, RoC {base['RoC']:.0%}; +1 pp merchant fee = +{M(sens['them_1pp_phi_doi_tac'])} VND.</li>
    </ul>
    <div class="risk"><b>Downside is real, and capped:</b> at ×2 / ×3 default, losses reach {stress['ty_le_lo_tren_giai_ngan']:.1%} / {severe['ty_le_lo_tren_giai_ngan']:.1%} of volume vs break-even
    {be['ty_le_lo_hoa_von_tren_giai_ngan']:.1%} (US BNPL charge-off {bm['cfpb_bnpl_chargeoff_rate_2021']:.1%}). Controls: stop-rule at {be_pd:.1%} PD,
    {M(pl['thin_file_cap'])} VND starter cap, ×0.5 reduced limits, short tenor, fixed pilot loss budget, pricing lever.</div>
  </div>
</section>

<section class="bottom">
  <div class="card">
    <div class="step">④ Feasibility · 6–12 months, existing customers first</div>
    <div class="road">
      <div class="phase"><b>0–3 mo · Backtest</b>
        <p>Run read-only on real HLBVN accounts; re-learn income factors; committee signs off the config.</p>
        <div class="gate">Gate: income error ≤ 10%; segment results hold at full sample</div></div>
      <div class="phase"><b>3–6 mo · Pilot</b>
        <p>1–2 merchants, low caps, fixed loss budget, weekly arrears tracking.</p>
        <div class="gate">Gate: 6-mo default ≤ 3%; 30+ DPD A &lt; 2%, B &lt; 5%</div></div>
      <div class="phase"><b>6–12 mo · Scale + ML</b>
        <p>More merchants, limit growth on, partner data, ML challenger on repayment labels.</p>
        <div class="gate">Stop: default &gt; {be_pd:.1%} (break-even)</div></div>
    </div>
  </div>

  <div class="card">
    <div class="step">⑤ Solution concept · KPIs measuring bottleneck relief</div>
    <table>
      <tr><th>KPI</th><th>Baseline (case)</th><th>Target</th><th style="text-align:right">Sim.</th></tr>
      <tr><td>Decision turnaround</td><td>1.8–4.6 days</td><td>&lt; 1 s</td><td class="n">lookup</td></tr>
      <tr><td>Cost per decision</td><td>380k VND</td><td>≤ 50k VND</td><td class="n">assumed</td></tr>
      <tr><td>Checkout conversion*</td><td>n/a</td><td>≥ 60%</td><td class="n">{conv:.0%}</td></tr>
      <tr><td>Gig &amp; MSME with a limit</td><td>{tgt_before:.0%} proxy</td><td>≥ 50%</td><td class="n">{tgt_after:.0%}</td></tr>
      <tr><td>Files sent to offline review</td><td>100% manual</td><td>≤ 10%</td><td class="n">{d['offline_share']:.1%}</td></tr>
      <tr><td>6-month default (stop &gt; {be_pd:.1%})</td><td>n/a</td><td>≤ 3%</td><td class="n">pilot</td></tr>
    </table>
    <p class="src" style="margin-top:2px">* approve + counter-offer on a {M(float(sim_key))} VND basket</p>
  </div>

  <div class="card data">
    <div class="step">⑥ Data &amp; assumptions</div>
    <p><b>Illustrative simulation:</b> stand-in synthetic data ({d['rows']['customers']} customers, {d['rows']['transactions'] / 1e6:.2f}M transactions, USD → VND at 25,000), reweighted to the Exhibit 2 mix;
    official case data arrives in the Strategy round. Cleaned in code: {sal['count']:,} mislabelled salary credits, {tr['count']:,} own transfers, {mort['count']} mortgages out of model DTI.</p>
    <table style="margin-top:3px">
      <tr><td>Manual review 380k VND · 1.8–4.6 days · 1.6–90.1M VND · 3–24 mo</td><td class="src">Case, Exhibit 2</td></tr>
      <tr><td>LGD 91% · EAD 72%</td><td class="src">Lending Club, 90k defaulted loans</td></tr>
      <tr><td>Base PD A 2% / B 5% / C 10%</td><td class="src">≈ 2× Lending Club grades A–C (6-mo)</td></tr>
      <tr><td>Merchant fee 2% · funding 6%</td><td class="src">below CFPB BNPL 2.49% · above VN deposits 4.6–5.8%</td></tr>
      <tr><td>Automated decision 50k VND</td><td class="src">team assumption, to confirm with HLBVN</td></tr>
    </table>
  </div>
</section>

</div></body></html>"""


def render_pdf(html_path: Path, pdf_path: Path):
    browser = next((b for b in BROWSERS if Path(b).exists()), None)
    if browser is None:
        print("Không tìm thấy Edge/Chrome; mở file HTML rồi in ra PDF (khổ 1920×1080, không lề).")
        return False
    cmd = [browser, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", "--run-all-compositor-stages-before-draw",
           f"--print-to-pdf={pdf_path}", html_path.resolve().as_uri()]
    subprocess.run(cmd, check=True, capture_output=True, timeout=120)
    _normalise_page(pdf_path)
    return pdf_path.exists()


def _normalise_page(pdf_path: Path):
    """Trình duyệt in 1920×1080 px CSS thành trang 1440×810 pt. Quy về đúng 1920×1080 pt (vector, không mất nét)
    và xuất kèm PNG 1920×1080 px. Cần pymupdf; không có thì giữ nguyên file."""
    try:
        import pymupdf
    except ImportError:
        return
    src = pymupdf.open(pdf_path)
    out = pymupdf.open()
    page = out.new_page(width=1920, height=1080)
    page.show_pdf_page(page.rect, src, 0)
    tmp = pdf_path.with_suffix(".tmp.pdf")
    out.save(tmp, garbage=3, deflate=True)
    src.close()
    out.close()
    tmp.replace(pdf_path)
    doc = pymupdf.open(pdf_path)
    doc[0].get_pixmap(dpi=72).save(pdf_path.with_suffix(".png"))   # 1920×1080 pt @72 dpi = 1920×1080 px
    doc.close()


def main():
    DOCS.mkdir(exist_ok=True)
    html_path = DOCS / "CreditForYou_infographic.html"
    pdf_path = DOCS / "CreditForYou_infographic.pdf"
    html_path.write_text(build_html(load()), encoding="utf-8")
    print(f"HTML: {html_path}")
    if render_pdf(html_path, pdf_path):
        print(f"PDF : {pdf_path}")


if __name__ == "__main__":
    sys.exit(main())
