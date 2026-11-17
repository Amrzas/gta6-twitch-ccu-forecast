"""Forecast model for the make.fun GTA VI Twitch CCU bounty.

For each hour h = 2..24 of the UTC release day it writes five quantiles of the
"Grand Theft Auto VI" category's mean concurrent Twitch viewers, inferred from
the hour-1 print and the past launches the case carries as `comparables`.

Method.  Everything is done in log space on ratios to the hour-1 print r_h =
ln(v_h / v_1).  Every comparable contributes one r_h curve, one size (ln of its
own hour-1 print), one unlock pattern (rolling midnight local / single global
unlock / PC) and a quality weight (early-access-contaminated and free-to-play
launches count less).  For each hour the model fits a weighted local-linear
surface of r_h over (log size, unlock-pattern mismatch), reads it at the
target's own size, and takes the weighted residual variance of the same fit as
the hour's log spread, with a floor of `sig_floor`; the spread is widened where
the target is bigger than every comparable, since the fit is then extrapolating.

A forecast is written as p1 * exp(mu_h + z_tau * sd_h) for the five levels
tau, rounded to two decimals and made non-decreasing.
"""

import math

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.6448536269514722, -0.6744897501960817, 0.0, 0.6744897501960817, 1.6448536269514722]
HOURS = list(range(2, 25))

PARAMS = {
    'tau_size': 0.9,       # log-size distance that divides a comparable's weight by e
    'tau_pat': 1.0,        # weight dividend for each unlock-pattern mismatch
    'size_clamp': 2.0,     # the size slope is read no further than this from the target
    'sig_floor': 0.06,     # smallest log spread of an hour
    'k': 0.70,             # spread multiplier (calibration on the comparables)
    'ridge': 1.0,          # ridge on the size slope
    'extra_spread': 0.25,  # extra log spread per unit of log size beyond the biggest comparable
    'min_neighbours': 5,
    'smooth': True,
    'bias': -0.08,         # log shift of the whole curve: the fit's mean ratio sits above the median        # 3-point smoothing of the hourly centre and spread
}


# ---------------------------------------------------------------- time helpers

def _days_from_civil(y, m, d):
    y -= m <= 2
    era = (y if y >= 0 else y - 399) // 400
    yoe = y - era * 400
    doy = (153 * (m + (-3 if m > 2 else 9)) + 2) // 5 + d - 1
    doe = yoe * 365 + yoe // 4 - yoe // 100 + doy
    return era * 146097 + doe - 719468


def parse_utc(iso):
    """ISO 8601 UTC timestamp -> seconds since the epoch (stdlib only)."""
    s = (iso or '').strip().replace('Z', '+00:00')
    date, _, rest = s.partition('T')
    y, m, d = (int(x) for x in date.split('-'))
    offset = 0
    if rest.endswith('+00:00'):
        rest = rest[:-6]
    elif '+' in rest[8:]:
        rest, _, tzs = rest.partition('+')
        hh, mm = (tzs.split(':') + ['0'])[:2]
        offset = int(hh) * 3600 + int(mm) * 60
    parts = (rest.split(':') + ['0', '0'])[:3]
    hh, mm, ss = (int(float(x)) for x in parts)
    return _days_from_civil(y, m, d) * 86400 + hh * 3600 + mm * 60 + ss - offset


def hourly_series(points, day_start):
    """{hour offset from day_start: viewers} for whole-hour points."""
    out = {}
    d0 = parse_utc(day_start)
    for p in points or ():
        t = p.get('from') or p.get('at')
        if not t:
            continue
        off = (parse_utc(t) - d0) / 3600.0
        r = int(round(off))
        if abs(off - r) < 0.01:
            out[r] = float(p.get('viewers') or 0)
    return out


# ------------------------------------------------------------------ utilities

def median(xs):
    xs = sorted(xs)
    n = len(xs)
    if not n:
        return 1.0
    return xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])


def solve(a, b):
    """Gaussian elimination with partial pivoting on a small dense system."""
    n = len(b)
    m = [row[:] + [b[i]] for i, row in enumerate(a)]
    for i in range(n):
        p = max(range(i, n), key=lambda r: abs(m[r][i]))
        if abs(m[p][i]) < 1e-12:
            return None
        m[i], m[p] = m[p], m[i]
        piv = m[i][i]
        for r in range(n):
            if r != i and m[r][i]:
                f = m[r][i] / piv
                for c in range(i, n + 1):
                    m[r][c] -= f * m[i][c]
    return [m[i][n] / m[i][i] for i in range(n)]


def patterns(text):
    """(rolling midnight local, single global unlock) read off an unlock note.

    The case's notes say when the game really unlocked, in words: "a rolling
    midnight local release", "the unlock rolls at local midnight ... US Eastern
    05:00Z", or "a single simultaneous global unlock".  Both are read with a
    little slack, since the wording differs from launch to launch; a note that
    says neither (unverified) is treated as neither.
    """
    t = (text or '').lower()
    glob = 0.0
    if 'no simultaneous' not in t:
        if 'simultaneous' in t or 'global unlock' in t or 'single global' in t:
            glob = 1.0
    midnight = 'midnight' in t
    rolls = 'roll' in t or 'midnight local' in t or 'local midnight' in t
    rolling = 1.0 if (midnight and rolls) else 0.0
    return rolling, glob


def tier_weight(text):
    t = (text or '').lower()
    if 'early-access' in t or 'early access' in t:
        return 0.35
    if 'weak' in t:
        return 0.5
    return 1.0


def prelaunch_level(series):
    """The category's ordinary level before launch: median of the last two weeks."""
    pts = [v for h, v in series.items() if -336 <= h <= -48 and v > 0]
    if len(pts) < 5:
        pts = [v for h, v in series.items() if h < 0 and v > 0]
    return median(pts) if pts else 1.0


# ---------------------------------------------------------------------- model

def build_pool(comparables):
    pool = []
    for c in comparables or ():
        series = hourly_series(c.get('hourly'), c.get('day_start'))
        v0 = series.get(0)
        if not v0 or v0 <= 0:
            continue
        curve = {}
        for h in HOURS:
            v = series.get(h - 1)
            if v is not None and v > 0:
                curve[h] = math.log(v / v0)
        if len(curve) < 6:
            continue
        pool.append({
            'name': c.get('name') or '',
            'size': math.log(v0),
            'state': math.log(v0 / max(1.0, prelaunch_level(series))),
            'pat': patterns(c.get('unlock') or ''),
            'w': tier_weight((c.get('unlock') or '') + ' ' + (c.get('platforms') or '')),
            'curve': curve,
        })
    return pool


def fallback(p1, spread=0.25):
    hours = []
    for h in HOURS:
        sd = spread * math.sqrt(h - 1)
        hours.append({'hour': h, 'q': quantiles(p1, 0.0, sd)})
    return hours


def scale(hours, p1, old, new):
    """Re-anchor a spread set to a new hour-1 print, keeping each hour's ratio."""
    return [{'hour': r['hour'], 'q': [round(max(0.01, q * new / old), 2) for q in r['q']]} for r in hours]


def quantiles(p1, mu, sd):
    qs = [max(0.01, min(1e9, p1 * math.exp(mu + z * sd))) for z in Z]
    out = [round(q, 2) for q in qs]
    for i in range(1, len(out)):
        if out[i] < out[i - 1]:
            out[i] = out[i - 1]
    return out


def forecast(hour1, notes, comparables, params=None):
    """The 23 forecast rows for one case."""
    p = dict(PARAMS)
    if params:
        p.update(params)
    p1 = max(1.0, float(hour1))
    pool = build_pool(comparables)
    size_t = math.log(p1)
    pat_t = patterns(notes or '')

    rows = []
    for c in pool:
        mism = sum(1.0 for a, b in zip(pat_t, c['pat']) if a != b)
        w = c['w'] * math.exp(-abs(size_t - c['size']) / p['tau_size'])
        w *= math.exp(-mism / p['tau_pat'])
        if w > 1e-4:
            rows.append((w, c['size'] - size_t, c))
    if len(rows) < p['min_neighbours']:
        return fallback(p1)

    biggest = max(c['size'] for _, _, c in rows)
    far = max(0.0, size_t - biggest)
    lim = p['size_clamp']

    out = []
    last = None
    for h in HOURS:
        obs = [(w, dz, c['curve'][h]) for w, dz, c in rows if h in c['curve']]
        if len(obs) < p['min_neighbours']:
            if last is None:
                return fallback(p1)
            mu, sd = last[0], last[1] * 1.04
        else:
            a = [[0.0, 0.0], [0.0, 0.0]]
            bb = [0.0, 0.0]
            tot = 0.0
            for w, dz, y in obs:
                z = max(-lim, min(lim, dz))
                tot += w
                bb[0] += w * y
                bb[1] += w * z * y
                a[0][0] += w
                a[0][1] += w * z
                a[1][1] += w * z * z
            a[1][0] = a[0][1]
            a[1][1] += p['ridge']
            sol = solve(a, bb)
            if sol is None:
                mu, slope = bb[0] / tot, 0.0
            else:
                mu, slope = sol
            resid = sum(w * (y - mu - slope * max(-lim, min(lim, dz))) ** 2 for w, dz, y in obs) / tot
            sd = math.sqrt(resid + p['sig_floor'] ** 2) * p['k']
            sd *= 1.0 + p['extra_spread'] * far
            last = (mu, sd)
        out.append({'hour': h, 'q': quantiles(p1, mu, sd), '_mu': mu, '_sd': sd})
    if p['smooth'] and len(out) == 23:
        mus = [r['_mu'] for r in out]
        sds = [r['_sd'] for r in out]
        for i in range(len(out)):
            lo, hi = max(0, i - 1), min(len(out) - 1, i + 1)
            w = [0.25, 0.5, 0.25] if (lo < i < hi) else [0.5, 0.5]
            m = sum(wj * mus[j] for wj, j in zip(w, range(lo, hi + 1)))
            s = sum(wj * sds[j] for wj, j in zip(w, range(lo, hi + 1)))
            out[i]['q'] = quantiles(p1, m, s)
    if p['bias']:
        shift = math.exp(p['bias'])
        for r in out:
            r['q'] = [max(0.01, min(1e9, q * shift)) for q in r['q']]
    for r in out:
        r.pop('_mu', None)
        r.pop('_sd', None)
    return out
