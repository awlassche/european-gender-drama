"""
Does gender classification accuracy rise, fall or stay flat over time?

Tests the trend in the predictions saved by train_temporal_fre.py
(--predictions_out), with the play as the unit of analysis rather than the
25-year window or the resampling iteration.

Why not test the window means or the iterations: the 50 iterations per
window are redraws from the same pool of plays, not independent samples,
so treating them as observations makes almost any difference "significant";
and with 5-7 window means a trend test has hardly any power.

The model, per method (TF-IDF, bge-m3):

    correct ~ decade              logistic regression (GEE, binomial)

  - correct: whether a test chunk's gender was predicted correctly
  - decade:  the play's year, centred on the mean year, in units of 10 years,
             so year is continuous and the result does not depend on the
             window boundaries
  - clustered by play: every chunk of a play, in every iteration it was in
    the test set, belongs to one cluster, and the standard errors are
    cluster-robust (sandwich). That accounts for chunks of the same play
    (and the same chunk in several iterations) not being independent. The
    working correlation is "independence", which keeps the slope a plain
    logistic-regression estimate; the robust standard errors do not depend
    on that choice being correct.

Reported: the slope as an odds ratio per decade with its 95% confidence
interval and p-value, the model's accuracy at the first and the last year,
and, as a simple check, the Spearman correlation between year and
per-play accuracy. A second model adds the chunk's gender, in case one
gender is recognised better and the gender mix of the test plays changes
over time.

python test_temporal_trend.py \\
  --predictions results/temporal_fre_predictions_v2_noreplace_heldout-plays_800.csv \\
                results/temporal_fre_predictions_v2_equal-speakers_300x1.csv \\
  --output results/temporal_trend_tests_fre_v2.txt
"""

import argparse
import os

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import spearmanr


def fit_trend(df, formula):
    """GEE logistic regression with play-clustered robust standard errors."""
    model = smf.gee(formula, groups="play", data=df,
                    family=sm.families.Binomial(), cov_struct=sm.cov_struct.Independence())
    return model.fit()


def test_method(df):
    df = df.copy()
    df["correct"] = (df["gender"] == df["predicted"]).astype(int)
    mean_year = df["year"].mean()
    df["decade"] = (df["year"] - mean_year) / 10

    res = fit_trend(df, "correct ~ decade")
    slope, (lo, hi), p = res.params["decade"], res.conf_int().loc["decade"], res.pvalues["decade"]

    # model-implied accuracy at the first and last year
    years = np.array([df["year"].min(), df["year"].max()])
    logit = res.params["Intercept"] + slope * (years - mean_year) / 10
    acc_first, acc_last = 1 / (1 + np.exp(-logit))

    res_g = fit_trend(df, "correct ~ decade + C(gender)")

    per_play = df.groupby("play").agg(year=("year", "first"), accuracy=("correct", "mean"))
    rho, rho_p = spearmanr(per_play["year"], per_play["accuracy"])

    return {
        "plays": df["play"].nunique(),
        "predictions": len(df),
        "years": f"{int(years[0])}-{int(years[1])}",
        "odds_ratio_per_decade": np.exp(slope),
        "ci_low": np.exp(lo),
        "ci_high": np.exp(hi),
        "p": p,
        "accuracy_first_year": acc_first,
        "accuracy_last_year": acc_last,
        "odds_ratio_per_decade_gender_adjusted": np.exp(res_g.params["decade"]),
        "p_gender_adjusted": res_g.pvalues["decade"],
        "spearman_play_accuracy_year": rho,
        "spearman_p": rho_p,
    }


def describe(r):
    direction = ("rises" if r["ci_low"] > 1 else "falls" if r["ci_high"] < 1 else "no significant trend")
    return (f"    plays: {r['plays']}, predictions: {r['predictions']}, years {r['years']}\n"
            f"    odds ratio per decade: {r['odds_ratio_per_decade']:.3f} "
            f"[95% CI {r['ci_low']:.3f}-{r['ci_high']:.3f}], p = {r['p']:.4f}  -> {direction}\n"
            f"    model accuracy: {r['accuracy_first_year']:.3f} (first year) -> "
            f"{r['accuracy_last_year']:.3f} (last year)\n"
            f"    adjusted for gender: odds ratio {r['odds_ratio_per_decade_gender_adjusted']:.3f}, "
            f"p = {r['p_gender_adjusted']:.4f}\n"
            f"    per-play accuracy vs year (Spearman): rho = {r['spearman_play_accuracy_year']:+.3f}, "
            f"p = {r['spearman_p']:.4f}\n")


def main():
    parser = argparse.ArgumentParser(description="Trend test on saved temporal predictions.")
    parser.add_argument("--predictions", nargs="+", required=True,
                        help="CSV files written by train_temporal_fre.py --predictions_out")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    lines = []
    for path in args.predictions:
        df = pd.read_csv(path)
        lines.append(f"{os.path.basename(path)}\n")
        for method, d in df.groupby("method", sort=False):
            lines.append(f"  {method}\n")
            lines.append(describe(test_method(d)))
        lines.append("\n")

    text = "".join(lines)
    print(text)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        f.write(text)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
