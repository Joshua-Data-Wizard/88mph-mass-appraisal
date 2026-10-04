"""
ratio.py — the IAAO ratio study, in about a hundred lines of arithmetic.

This is the heart of Section 9, and it deliberately contains no model, no
agent and no database. It takes sales, it returns statistics, and every number
in it can be checked by hand. Everything else in Section 9 — the tool the
agent calls, the scorecard the app draws, the notebook's tables — reads from
here, so there is exactly one place the maths can be wrong.

A ratio study asks the one question an appraiser lives by:

    How close is what we SAID a property was worth
    to what it ACTUALLY sold for?

That fraction is the *ratio*:  assessed value / sale price.

Do it for one sale and you know nothing. Do it for every arm's length sale in
a neighborhood and three separate problems become visible. IAAO gives each one
its own test:

    LEVEL       Are we high or low overall?          median ratio
    UNIFORMITY  Are we consistent, or scattered?     COD
    EQUITY      Do cheap and expensive homes get     PRD, PRB
                treated the same?

They are independent. A neighborhood can pass all three, or fail exactly one,
and which one it fails tells you what to go and fix. That is why Hill Valley
County has three neighborhoods each failing a different test — it gives an
appraiser three genuinely different afternoons of work.

The one that catches people: **the median, not the average.** A handful of odd
sales drag an average around; the median barely moves. Every number here that
could be either is the median on purpose, and Section 9 makes a point of it,
because a language model asked to do this in SQL reaches for AVG() every time.

ALL OF THIS DATA IS SYNTHETIC. See data/README.md.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd


# --------------------------------------------------------------------------
#  The standards themselves
# --------------------------------------------------------------------------
#
# From the IAAO *Standard on Ratio Studies*. These are the acceptance ranges
# for improved residential property, which is what Hill Valley County's demo
# roll is. An office valuing farmland or industrial property uses wider bands
# -- they are listed in the standard, and an attendee should edit these numbers
# rather than work around them.

#: Fewer sales than this and the statistics mean nothing at all.
MIN_SALES_TO_JUDGE = 5

#: Below this, report the numbers but say out loud that the sample is thin.
SMALL_SAMPLE = 30

#: The three graded tests. One neighborhood, one verdict per test.
STANDARDS: dict[str, dict[str, Any]] = {
    "level": {
        "label": "Level",
        "metric": "Median ratio",
        "low": 0.90, "high": 1.10,
        "asks": "Are we assessing high or low overall?",
        "fails_low": "assessing BELOW market",
        "fails_high": "assessing ABOVE market",
    },
    "uniformity": {
        "label": "Uniformity",
        "metric": "COD",
        "low": 5.0, "high": 15.0,
        "asks": "Are we consistent from one property to the next?",
        "fails_low": "suspiciously tight - check for sales chasing",
        "fails_high": "values are scattered; similar homes treated differently",
    },
    "equity": {
        "label": "Equity",
        "metric": "PRD",
        "low": 0.98, "high": 1.03,
        "asks": "Do cheap and expensive homes get the same treatment?",
        "fails_low": "progressive - expensive homes assessed relatively higher",
        "fails_high": "regressive - cheaper homes assessed relatively higher",
    },
}

#: PRB is computed and reported, but NOT graded, and that is a deliberate
#: choice worth explaining because it is the sort of thing an attendee will
#: reasonably want to change.
#:
#: PRB is the better statistic. IAAO added it precisely because PRD can
#: understate regressivity, and Hill Valley County demonstrates that beautifully
#: by accident: Hilldale's PRD reads a mild 1.04 while its PRB is a brutal
#: -0.42, because its ratios really do slide from 1.35 on a $160,000 home to
#: 0.66 on a $666,000 one.
#:
#: It stays off the scorecard for a teaching reason, not a statistical one.
#: Graded, PRB fails in three of six neighborhoods -- including two that pass
#: PRD -- and every neighborhood ends up failing two tests. Section 9 rests on
#: each troubled neighborhood failing exactly ONE test, so an appraiser can see
#: three distinct problems needing three distinct fixes. A fourth grade blurs
#: that into noise.
#:
#: So: report it, explain it, do not grade it. An office running this for real
#: should absolutely grade it -- move this entry into STANDARDS above.
ALSO_REPORTED: dict[str, dict[str, Any]] = {
    "equity_prb": {
        "label": "Equity",
        "metric": "PRB",
        "low": -0.05, "high": 0.05,
        "asks": "How far does the ratio move each time value doubles?",
        "fails_low": "regressive - ratio falls as value rises",
        "fails_high": "progressive - ratio rises as value rises",
    },
}


def _judge(value: float, low: float, high: float) -> str:
    """PASS if the value sits inside the standard's range, FAIL if not."""
    if value is None:
        return "N/A"
    return "PASS" if low <= value <= high else "FAIL"


# --------------------------------------------------------------------------
#  The four statistics
# --------------------------------------------------------------------------

def median_ratio(ratios: "pd.Series") -> float:
    """LEVEL. The middle ratio, once they are lined up smallest to largest.

    Half the sales came in above this and half below. Unlike an average, one
    four-million-dollar outlier cannot drag it anywhere.
    """
    return float(ratios.median())


def cod(ratios: "pd.Series") -> float:
    """UNIFORMITY. Coefficient of Dispersion, as a percentage.

    Read it as: "a typical sale misses the median by this percent."

        COD = 100 x (average distance from the median) / median

    A COD of 12 means a typical property in this group sits about 12% away
    from the middle of the group -- some above, some below. Two neighborhoods
    can share an identical median and have wildly different CODs, and the one
    with the high COD is the one generating appeals.
    """
    middle = ratios.median()
    if middle == 0:
        return float("nan")
    average_miss = (ratios - middle).abs().mean()
    return float(100.0 * average_miss / middle)


def prd(assessed: "pd.Series", sale: "pd.Series") -> float:
    """EQUITY. Price-Related Differential.

        PRD = (plain average of the ratios) / (dollar-weighted average)

    The trick is in what the two averages count. The plain average gives every
    SALE one vote, so a modest bungalow counts as much as a mansion. The
    weighted average gives every DOLLAR one vote, so the mansion dominates.

    If both kinds of property are treated alike, the two averages agree and
    the PRD sits at 1.00. Above 1.03 the cheaper homes are carrying relatively
    more assessment than the expensive ones -- *regressive*, and the single
    most common finding in a real ratio study.
    """
    ratios = assessed / sale
    dollar_weighted = assessed.sum() / sale.sum()
    if dollar_weighted == 0:
        return float("nan")
    return float(ratios.mean() / dollar_weighted)


def prb(assessed: "pd.Series", sale: "pd.Series") -> float:
    """EQUITY, the modern measure. Price-Related Bias.

    PRD has a known weakness: a couple of extreme sales can swing it. PRB was
    added to the standard to answer the same question more robustly, and it
    reports something an appraiser can say out loud in a commission meeting:

        "For every doubling of property value, our assessment ratio
         moves by PRB x 100 percent."

    So a PRB of -0.08 means that each time you double the price, the ratio
    drops another 8% -- cheaper homes are being over-assessed relative to
    expensive ones. The standard wants it inside plus or minus 0.05.

    It is an ordinary least-squares slope, written out rather than pulled from
    scikit-learn, because two lines of algebra are easier to check than an
    import.
    """
    import numpy as np

    ratios = assessed / sale
    middle = ratios.median()
    if middle == 0 or len(ratios) < 2:
        return float("nan")

    # PRB asks how the ratio moves AS VALUE CHANGES. If every property sold
    # for the same price there is no value axis to move along, and the slope
    # you would get back is an artifact of the assessed values alone. Say so
    # instead of returning a confident nonsense number.
    if sale.nunique() < 2:
        return float("nan")

    # How far off the middle is this sale, as a fraction of the middle?
    y = (ratios - middle) / middle

    # A value scale blending what the market said with what we said, then log
    # base 2 so one step to the right means "twice as expensive".
    value = 0.5 * sale + 0.5 * (assessed / middle)
    x = np.log(value) / np.log(2.0)

    spread = x - x.mean()
    denominator = float((spread ** 2).sum())
    if denominator == 0:
        return float("nan")
    return float((spread * (y - y.mean())).sum() / denominator)


# --------------------------------------------------------------------------
#  A whole study
# --------------------------------------------------------------------------

def ratio_stats(sales: "pd.DataFrame", *,
                arms_length_only: bool = True) -> dict[str, Any]:
    """Run all four tests over a set of sales and report the verdicts.

    Expects columns `assessed_value` and `sale_price`. An `arms_length`
    column, if present, is honoured.

    Returns a plain dictionary rather than an object, because this result is
    JSON-encoded straight back into a language model's context in Section 7.
    It has to read well to something that has never seen this codebase.

    Non-arm's length sales are dropped BY DEFAULT and the count is reported.
    Foreclosures, sheriff's deeds and transfers between family members are not
    evidence of market value, and IAAO excludes them. The tool doing that on
    the appraiser's behalf -- and saying so in its answer -- is the whole
    argument for putting professional judgment inside a tool rather than
    hoping the model remembers.
    """
    total = len(sales)
    excluded_not_arms_length = 0

    if arms_length_only and "arms_length" in sales.columns:
        keep = sales["arms_length"] == 1
        excluded_not_arms_length = int((~keep).sum())
        sales = sales[keep]

    # A sale price of zero is a recording artifact, not a sale. Dividing by it
    # produces infinities that quietly poison every statistic downstream.
    usable = sales[(sales["sale_price"] > 0) & sales["assessed_value"].notna()]
    excluded_unusable = len(sales) - len(usable)

    n = len(usable)
    result: dict[str, Any] = {
        "sales_examined": total,
        "sales_used": n,
        "sales_excluded_not_arms_length": excluded_not_arms_length,
        "sales_excluded_unusable": excluded_unusable,
    }

    if n < MIN_SALES_TO_JUDGE:
        result["verdict"] = "NOT ENOUGH SALES"
        result["tests"] = []
        result["note"] = (
            f"Only {n} usable sale(s). IAAO asks for at least "
            f"{MIN_SALES_TO_JUDGE} before a ratio study means anything, and "
            f"prefers {SMALL_SAMPLE} or more. Widen the query - a longer date "
            f"range, or several similar neighborhoods together."
        )
        result["summary"] = result["note"]
        return result

    assessed = usable["assessed_value"].astype(float)
    sale = usable["sale_price"].astype(float)
    ratios = assessed / sale

    values = {
        "level": median_ratio(ratios),
        "uniformity": cod(ratios),
        "equity": prd(assessed, sale),
        "equity_prb": prb(assessed, sale),
    }

    def describe(key: str, standard: dict[str, Any]) -> dict[str, Any]:
        value = values[key]
        verdict = _judge(value, standard["low"], standard["high"])
        entry: dict[str, Any] = {
            "test": standard["label"],
            "metric": standard["metric"],
            "value": round(value, 4),
            "standard": f"{standard['low']} to {standard['high']}",
            "verdict": verdict,
            "asks": standard["asks"],
        }
        if verdict == "FAIL":
            entry["meaning"] = (standard["fails_low"] if value < standard["low"]
                                else standard["fails_high"])
        return entry

    tests = [describe(key, std) for key, std in STANDARDS.items()]

    # Reported, not graded -- see ALSO_REPORTED for why. The verdict field is
    # kept so a reader can see where it sits against the standard, but it does
    # not count toward the overall PASS/FAIL.
    result["also_reported"] = [describe(key, std)
                               for key, std in ALSO_REPORTED.items()]

    failed = [t for t in tests if t["verdict"] == "FAIL"]
    result["tests"] = tests
    result["verdict"] = "PASS" if not failed else "FAIL"
    result["failed_tests"] = [f"{t['test']} ({t['metric']})" for t in failed]

    notes = []
    if excluded_not_arms_length:
        notes.append(
            f"{excluded_not_arms_length} sale(s) excluded as not arm's length, "
            f"per IAAO - foreclosures and family transfers are not market "
            f"evidence.")
    if n < SMALL_SAMPLE:
        notes.append(
            f"Only {n} sales. IAAO prefers {SMALL_SAMPLE} or more; treat these "
            f"figures as indicative rather than conclusive.")
    if notes:
        result["note"] = " ".join(notes)

    result["summary"] = summarize(result)
    return result


def summarize(result: dict[str, Any]) -> str:
    """One sentence a person could read out in a meeting.

    Lives here rather than in the tool that calls it, because the sentence is
    part of knowing how to report a ratio study -- an attendee running
    `ratio_stats()` on their own sales should get the same wording the agent
    gives, not have to write it themselves.
    """
    if result.get("verdict") == "NOT ENOUGH SALES":
        return result.get("note", "Not enough sales to judge.")

    parts = [f"{t['metric']} {t['value']} ({t['verdict']})"
             for t in result.get("tests", [])]

    if result.get("verdict") == "PASS":
        return (f"Meets all three IAAO standards on {result['sales_used']} "
                f"sales: " + "; ".join(parts) + ".")

    return (f"FAILS {', '.join(result.get('failed_tests', []))} on "
            f"{result['sales_used']} sales: " + "; ".join(parts) + ".")


def ratio_by_group(sales: "pd.DataFrame", group_by: str, *,
                   arms_length_only: bool = True) -> "pd.DataFrame":
    """Run a separate ratio study for each value of `group_by`.

    This is the function behind Section 9's whole point. Run the county as one
    lump and Hill Valley looks healthy. Run it grouped by neighborhood and
    three of the six fail -- each one a different test.

    Returns one row per group, sorted worst-first, so the neighborhoods that
    need an appraiser's attention are at the top where they belong.
    """
    import pandas as pd

    if group_by not in sales.columns:
        raise KeyError(
            f"Cannot group by {group_by!r} - the sales have no such column. "
            f"Available: {', '.join(sales.columns)}")

    rows = []
    for name, group in sales.groupby(group_by, dropna=False):
        stats = ratio_stats(group, arms_length_only=arms_length_only)
        row: dict[str, Any] = {
            group_by: name,
            "sales_used": stats["sales_used"],
            "verdict": stats["verdict"],
        }
        for test in stats["tests"]:
            row[test["metric"]] = test["value"]
            row[test["metric"] + " verdict"] = test["verdict"]
        for test in stats.get("also_reported", []):
            row[test["metric"]] = test["value"]
        row["failed"] = ", ".join(stats.get("failed_tests", [])) or "-"
        rows.append(row)

    frame = pd.DataFrame(rows)

    # Worst first: whatever failed, ordered by how many tests it failed.
    if "failed" in frame.columns:
        frame["_failures"] = frame["failed"].apply(
            lambda s: 0 if s == "-" else s.count(",") + 1)
        frame = frame.sort_values(
            ["_failures", group_by], ascending=[False, True]
        ).drop(columns="_failures").reset_index(drop=True)

    return frame
