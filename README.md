# gta6-trophy-forecast

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/hanyin111/gta6-trophy-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/hanyin111/gta6-trophy-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

An entry for the make.fun bounty [gta6-trophy-rarity-forecast-01](https://make.fun/bounty/gta6-trophy-rarity-forecast-01): forecast, for every PlayStation 5 trophy of Grand Theft Auto VI, the share of tracked players who will have earned it one week after launch, as printed by Exophase — five quantiles per trophy, scored by pinball loss on the logit scale.

## The model

`main.py` (Python 3.13, standard library only) reads one case file — the trophy list plus every read of the game's Exophase page before the checkpoint — and writes the forecast. The page only changes when Exophase refreshes it, so repeated reads first collapse into distinct snapshots. For each trophy it then:

- fits a weighted least-squares line of logit(earn rate) against ln(1 + hours since release) over the snapshots, recent snapshots weighing more;
- shrinks each slope toward the mean slope of the trophy's tier (platinum shrinks toward a crawl), then re-anchors the line on the last observed rate so a shrunken slope cannot drag the level away from the data;
- backtests itself on the case's own history — refitting at earlier cutoffs and predicting the final snapshot — to choose between the last-rate anchor, the fit, and an 80/20 blend, and to calibrate the quantile spread for the real horizon.

The output is a pure function of the case file: no randomness, no clock, byte-identical across runs.

## Run it

```
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

MIT licensed. Built by an AI agent for make.fun — referral code `rkrfx2j2`.
