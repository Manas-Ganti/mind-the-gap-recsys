# Same-day ranker validation diagnostic (2026-10-13)

`same_day_validation.py` produced `same_day_validation.log` on the laptop: MIND-small, seed 42,
code at commit 567cb40. It needs that commit, because it uses `ranking.valid_fraction`, which
decision 7 removed. It reproduces the old stage-2 setup: rankers trained on 2019-11-14 and
early-stopped on held-out users from the same day. It reports NDCG@10 within candidates on
those held-out users and on the test day, the val→test feature drift, and GBDT feature
ablations.

The test-day ablation rows are diagnostic only. No shipped model was chosen from them; see
decision 7 in `CLAUDE.md`.
