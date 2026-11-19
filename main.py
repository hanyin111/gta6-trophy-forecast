#!/usr/bin/env python3
"""Forecast GTA VI trophy earn rates (Exophase, PS5) at a future target.

Reads one case file per run from $MAKEFUN_INPUT_DIR (else /input): the
trophy list and every read of the game's Exophase page before the case's
checkpoint. Writes one forecast file per case to $MAKEFUN_OUTPUT_DIR
(else /output): five quantiles (0.05, 0.25, 0.5, 0.75, 0.95) of each
trophy's earn rate at the target, scored by pinball loss on the logit
scale.

Model, per trophy: a weighted least-squares line of logit(rate) against
ln(1 + hours since release), fitted on the distinct page snapshots (the
page only changes at Exophase refreshes, so repeated reads of one frozen
page are one observation). Slopes are shrunk toward their tier's mean
slope, the fitted median is blended with the last-rate anchor, and an
internal backtest over the case's own history picks the blend and
calibrates the quantile spread for the real horizon. Deterministic: the
output is a pure function of the case file.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone

Z = [-1.6448536269514722, -0.6744897501960817, 0.0,
     0.6744897501960817, 1.6448536269514722]
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]


def parse_time(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def clamp_rate(rate):
    return min(max(rate, 0.01), 99.99)


def logit(rate):
    p = clamp_rate(rate) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(y):
    if y >= 0:
        return 100.0 / (1.0 + math.exp(-y))
    e = math.exp(y)
    return 100.0 * e / (1.0 + e)


def round4(x):
    return round(x, 4)


def weighted_line(xs, ys, ws):
    """Weighted least squares y ~ a + b*x; returns (a, b, x_var)."""
    sw = sum(ws)
    if sw <= 0:
        return (ys[-1] if ys else 0.0), 0.0, 0.0
    mx = sum(w * x for x, w in zip(xs, ws)) / sw
    my = sum(w * y for y, w in zip(ys, ws)) / sw
    sxx = sum(w * (x - mx) ** 2 for x, w in zip(xs, ws))
    if sxx <= 1e-12:
        return my, 0.0, 0.0
    sxy = sum(w * (x - mx) * (y - my) for x, y, w in zip(xs, ys, ws))
    b = sxy / sxx
    return my - b * mx, b, sxx / sw


def snapshots(case, anchor):
    """Distinct page states, oldest first: (x, players, [rates]).

    The page is frozen between Exophase refreshes, so consecutive reads
    with identical players and rates collapse into the first sighting.
    """
    out = []
    for read in case["history"]:
        rates = read["rates"]
        players = read["players_tracked"]
        if out and out[-1][1] == players and out[-1][2] == rates:
            continue
        t = parse_time(read["at"])
        hours = (t - anchor).total_seconds() / 3600.0
        out.append((math.log1p(max(hours, -0.99)), players, rates))
    return out


def fit_predict(snaps, tiers, x_target, upto=None):
    """Fit every trophy on snapshots[:upto] and predict at x_target.

    Returns (anchor_mu, fit_mu) per trophy, in logit space, or None for
    a trophy with no usable rate.
    """
    use = snaps if upto is None else snaps[:upto]
    n = len(use)
    anchor_mu = []
    raw = []
    for ti in range(len(tiers)):
        xs, ys, ws = [], [], []
        last = None
        for k, (x, _players, rates) in enumerate(use):
            r = rates[ti] if ti < len(rates) else None
            if r is None:
                continue
            last = logit(r)
            xs.append(x)
            ys.append(last)
            # Recency weight: a snapshot's say doubles every 30 hours.
            ws.append(2.0 ** (k / 12.0))
        anchor_mu.append(last)
        if len(xs) >= 2:
            a, b, xvar = weighted_line(xs, ys, ws)
            raw.append((a, b, xvar, len(xs)))
        else:
            raw.append(None)
    # Tier mean slopes for shrinkage (precision-flavoured average).
    tier_slopes = {}
    for ti, entry in enumerate(raw):
        if entry is not None:
            tier_slopes.setdefault(tiers[ti], []).append(entry[1])
    tier_mean = {t: sum(v) / len(v) for t, v in tier_slopes.items()}
    all_slopes = [e[1] for e in raw if e is not None]
    global_mean = sum(all_slopes) / len(all_slopes) if all_slopes else 0.0
    # Predict with the shrunk slope, re-anchored on the last observed
    # point so a shrunken slope cannot drag the level away from the data.
    fit_mu = []
    for ti, entry in enumerate(raw):
        if entry is None or anchor_mu[ti] is None:
            fit_mu.append(anchor_mu[ti])
            continue
        a, b, xvar, cnt = entry
        if tiers[ti] == "platinum":
            target_slope = 0.3 * global_mean
        else:
            target_slope = tier_mean.get(tiers[ti], global_mean)
        lam = cnt / (cnt + 6.0)
        b_shrunk = lam * b + (1.0 - lam) * target_slope
        # Re-anchor the shrunk line on the last observed point so a
        # shrunken slope cannot drag the level away from the data.
        xs_last = None
        for x, _p, rates in reversed(use):
            if ti < len(rates) and rates[ti] is not None:
                xs_last = x
                break
        y_last = anchor_mu[ti]
        fit_mu.append(y_last + b_shrunk * (x_target - xs_last))
    return anchor_mu, fit_mu


def pinball(errors, spread):
    """Mean pinball loss on the logit scale for a normal spread."""
    total = 0.0
    count = 0
    for e in errors:
        for tau, z in zip(LEVELS, Z):
            d = e - z * spread
            total += 2.0 * max(tau * d, (tau - 1.0) * d)
            count += 1
    return total / count if count else 0.0


def forecast_case(case):
    anchor_time = parse_time(case["game"]["release_date"] + "T00:00:00Z")
    tiers = [t["tier"] for t in case["trophies"]]
    snaps = snapshots(case, anchor_time)
    target_t = parse_time(case["target"])
    hours = (target_t - anchor_time).total_seconds() / 3600.0
    x_target = math.log1p(max(hours, -0.99))
    horizon_days = case["horizon_minutes"] / 1440.0

    anchor_mu, fit_mu = fit_predict(snaps, tiers, x_target)
    last_x = snaps[-1][0] if snaps else x_target

    # Backtest: refit at earlier cutoffs, predict the final snapshot,
    # and score three candidate medians (anchor, fit, 80/20 blend).
    cand_errors = {0: [], 1: [], 2: []}
    dx_list = []
    if len(snaps) >= 5:
        final_rates = snaps[-1][2]
        for frac in (0.4, 0.55, 0.7, 0.85):
            upto = max(3, int(len(snaps) * frac))
            if upto >= len(snaps):
                continue
            a_mu, f_mu = fit_predict(snaps, tiers, last_x, upto=upto)
            dx = last_x - snaps[upto - 1][0]
            if dx <= 0.02:
                continue
            dx_list.append(dx)
            for ti in range(len(tiers)):
                r = final_rates[ti] if ti < len(final_rates) else None
                if r is None or a_mu[ti] is None or f_mu[ti] is None:
                    continue
                y = logit(r)
                cand_errors[0].append(y - a_mu[ti])
                cand_errors[1].append(y - f_mu[ti])
                cand_errors[2].append(y - (0.8 * f_mu[ti] + 0.2 * a_mu[ti]))
    blend = 2  # default: 80% fit, 20% anchor
    spread = None
    if dx_list and cand_errors[2]:
        dx_bt = sum(dx_list) / len(dx_list)
        dx_real = max(x_target - last_x, 0.05)
        scale = math.sqrt(dx_real / dx_bt)
        best_key, best_loss = None, None
        for key in (0, 1, 2):
            errs = cand_errors[key]
            if not errs:
                continue
            sd = math.sqrt(sum(e * e for e in errs) / len(errs))
            loss = pinball(errs, max(sd * scale, 0.05))
            if best_loss is None or loss < best_loss:
                best_key, best_loss = key, loss
        if best_key is not None:
            blend = best_key
            errs = cand_errors[blend]
            sd = math.sqrt(sum(e * e for e in errs) / len(errs))
            spread = sd * scale
    if spread is None:
        spread = 0.5 * math.sqrt(max(horizon_days, 0.01))
    spread = min(max(spread, 0.35), 2.5)

    rows = []
    for ti, trophy in enumerate(case["trophies"]):
        a_mu = anchor_mu[ti]
        f_mu = fit_mu[ti]
        if a_mu is None:
            mu = 0.0
        elif blend == 0 or f_mu is None:
            mu = a_mu
        elif blend == 1:
            mu = f_mu
        else:
            mu = 0.8 * f_mu + 0.2 * a_mu
        s_i = spread
        qs = [round4(inv_logit(mu + z * s_i)) for z in Z]
        for k in range(1, len(qs)):
            if qs[k] < qs[k - 1]:
                qs[k] = qs[k - 1]
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def main():
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    os.makedirs(out_dir, exist_ok=True)
    names = sorted(n for n in os.listdir(in_dir) if n.endswith(".in"))
    if not names:
        print("no case files in", in_dir, file=sys.stderr)
        return 1
    for name in names:
        with open(os.path.join(in_dir, name), encoding="utf-8") as fh:
            case = json.load(fh)
        result = forecast_case(case)
        out_name = name[:-3] + ".out"
        with open(os.path.join(out_dir, out_name), "w",
                  encoding="utf-8") as fh:
            json.dump(result, fh, separators=(",", ":"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
