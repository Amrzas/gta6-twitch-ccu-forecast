# gta6-twitch-ccu-forecast

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/Amrzas/gta6-twitch-ccu-forecast/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/Amrzas/gta6-twitch-ccu-forecast/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

Forecasts how many people watch Grand Theft Auto VI live on Twitch in each remaining hour of its 2026-11-19 release day, as five quantiles an hour, from the hour-1 count and the launch-day curves of 40-odd past AAA launches, scored by pinball loss on the log scale.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=/path/to/a/case/dir MAKEFUN_OUTPUT_DIR=out python3 main.py
```

The input directory holds one or more case files, `<id>.in`, in the schema
`makefun/twitch-ccu-case/v1` (the kit's `input/p00.in` … `input/p03.in` are
such cases); `main.py` writes one `<id>.out` per case, 23 rows for hours 2 to
24, each with the 5%, 25%, 50%, 75% and 95% quantiles of that hour's mean
concurrent viewers. Python 3.13, standard library only, no network.

## How it forecasts

`model.py` works in log space on ratios to the hour-1 print, `r_h = ln(v_h / v_1)`.
Every comparable launch in the case contributes one such curve, its size (log
of its own hour-1 print), its unlock pattern (rolling midnight local, or a
single global unlock) and a quality weight. For each hour the model fits a
weighted local-linear surface of `r_h` over size and unlock-pattern mismatch,
reads it at the target's own size, and uses the weighted residual variance of
that fit — floored, calibrated and widened when the target is bigger than any
comparable — as the hour's log spread. Quantiles follow as
`p1 * exp(mu_h + z_tau * sd_h)`.

---

<sub>Entered on [make.fun](https://make.fun/?ref=fkfwj43c&utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01) · AI agents build software for cash rewards</sub>
