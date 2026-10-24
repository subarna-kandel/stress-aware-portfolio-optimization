# Limitations

* One US industry universe and one 33-year sample; one full cycle of crises (2000-02, 2008, 2020, 2022) is a small number of independent stress episodes.
* French industry portfolios are research portfolios, not tradable instruments: no bid-ask spreads, shorting/financing costs, capacity, or market-impact; a flat 5 bps cost is a stylised assumption (sensitivity 0–50 bps reported).
* The forecast target is the volatility of the equal-weighted universe; the hedge ratio for other strategies is implicit.
* VIX is S&P 500 implied volatility, used as a market-condition feature only. VIX of day *t* is used in a decision at the close of *t* and executed the next day; its availability intraday is an assumption.
* Overlapping 5-day targets: DM tests use HAC corrections, but effective sample size is smaller than the number of daily forecasts.
* Random Forest cannot extrapolate beyond the target range seen in training, which matters in crises not represented in the 3-year window.
* Monthly rebalancing with a 5-day forecast horizon is a mismatch inherited from the specification; weekly/quarterly results are reported.
* Parameters of the main specification were fixed from the project specification before the backtest was run; the sensitivity grid is reported in full. Even so, this is a single historical path: results do not imply future performance, and nothing here is investment advice.
* Data redistribution rights are unverified; raw files are not committed.
