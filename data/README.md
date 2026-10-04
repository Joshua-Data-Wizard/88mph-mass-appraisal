# 🏘️ Hill Valley County — the synthetic dataset

> ## ⚠️ Every record in this folder is invented.
>
> No real parcel, owner, address, sale, or tax account appears anywhere in this
> repository. The names are drawn from a fictional town in a 1985 film. The
> coordinates are real Calgary-area coordinates — chosen so the maps in Section 9
> render over actual streets — but **every property sitting on them is fictional**.
>
> This is a public repository about property records. It is treated accordingly.

---

## Why three databases instead of one

```
data/cama.db     🏛️  The Appraiser's Office     1,200 parcels
data/clerk.db    📜  The Clerk of Courts           244 recorded deeds
data/tax.db      💵  The Tax Collector           1,200 tax accounts
```

Three separate files. No foreign keys between them. No shared connection. You
cannot write a SQL join across them, and that is deliberate.

This is the actual situation in a real county: three separately elected offices,
three software vendors, three databases, and no integration anyone ever funded.
It is the reason **Section 10** exists. A question like *"may this parcel be
split?"* cannot be answered from any single one of them — it needs facts from
all three plus a rule from a policy document, which is exactly the case MCP is
built to solve.

Each office knows something the others do not:

| Office | Knows | Cannot tell you |
|---|---|---|
| **Appraiser** | characteristics, ownership of record, assessed value, location | what anything sold for, or whether taxes were paid |
| **Clerk of Courts** | recorded deeds, sale prices, grantor/grantee, instrument numbers | what the property looks like, or what it is assessed at |
| **Tax Collector** | amounts levied and paid, balances, delinquency | why the value is what it is, or who sold it to whom |

---

## Rebuilding it

```powershell
uv run python scripts/generate_hill_valley_data.py
```

The generator is **deterministic** — seeded on `1985`. Every machine that runs
it produces byte-identical databases, which is why the notebook outputs
committed to this repo always match what you get locally. Safe to re-run at any
time; it overwrites cleanly.

CSV mirrors of all three tables live in [`csv/`](csv/) so you can read the data
on GitHub without opening SQLite.

---

## 🎯 The flaws are on purpose

**A dataset with no problems in it makes Section 9 pointless.** If every
neighborhood were assessed perfectly, the ratio study would find nothing and
there would be no reason for an appraiser to want an assistant.

So each neighborhood is built to fail a *different* IAAO test. These are the
numbers the generator actually produces — not targets, measured output.

**Qualified residential sales only** — `arms_length = 1` and
`property_class IN ('R01','R02')` — which is how IAAO stratifies and how
`verify()` in the generator reports:

| Neighborhood | n | Median | COD | PRD | Built to demonstrate |
|---|---:|---:|---:|---:|---|
| Lyon Estates | 40 | 0.948 | 6.07 | 1.002 | ✅ a healthy baseline |
| Twin Pines Ranch | 28 | 0.935 | 9.60 | 1.002 | ✅ passes |
| Eastwood Ravine | 27 | 0.984 | 6.06 | 1.001 | ✅ passes |
| **Riverside Heights** | 31 | **0.807** | 6.27 | 1.002 | ❌ **LEVEL** — assessed 19% below market |
| **Courthouse Square** | 29 | 0.938 | **22.20** | 1.026 | ❌ **UNIFORMITY** — inconsistent, COD over 15 |
| **Hilldale** | 47 | 0.930 | 13.59 | **1.042** | ❌ **EQUITY** — regressive, PRD over 1.03 |
| **COUNTYWIDE** | 202 | 0.927 | 12.37 | 1.027 | ⚠️ **every measure passes** |

> ### ⚠️ Section 9 shows slightly different figures, and both are right
>
> The §9 agent writes its own SQL and **does not filter by property class** —
> it is handed the sales feed and asked a question. Across all classes every
> verdict is identical and the numbers move in the third decimal:
>
> | | residential only | all classes |
> |---|---:|---:|
> | Hilldale PRD | 1.042 | **1.043** |
> | Courthouse Square COD | 22.20 | **22.08** |
> | Countywide *n* | 202 | **210** |
>
> **[DEMOS.md](../DEMOS.md) prints the all-classes table**, because that is
> what the room actually sees on screen. Neither table is wrong — they answer
> slightly different questions. **If you quote a figure anywhere, say which
> one it is.** Not doing that is what let a third set of numbers survive in
> this file until 2026-10-04.

> ### 🔁 Re-verify these rather than trusting them
>
> `verify()` in
> [`scripts/generate_hill_valley_data.py`](../scripts/generate_hill_valley_data.py)
> prints this exact table on every run. ⚠️ **Anything that changes how often
> the generator calls `rng` moves every number after it.** That has already
> happened once — bounding the map scatter shifted Hilldale's median from
> 0.902 to 0.930 — and this table kept the old figures until somebody
> re-derived them. **Regenerate, read the output, update the table.**

### That last row is the whole lesson

Look at it again. Countywide, Hill Valley County passes all three IAAO
standards — median inside 0.90–1.10, COD under 15, PRD inside 0.98–1.03. An
office reporting only that number would call the year a success.

Three of its six neighborhoods have real, actionable problems. The errors
offset each other in aggregate and vanish.

**This is why stratification is mandatory** (Policy 9.5), and it is the finding
the Section 9 agent is built to surface.

### Each failure is a genuinely different problem

- **Riverside Heights** — the market ran away and the roll did not follow.
  Everyone there is under-assessed *together*. Uniformity is fine; the level is
  wrong. Fixed with a reappraisal.
- **Courthouse Square** — historic housing stock, 1901–1946, wildly varied.
  The *average* is right, so nothing looks wrong until you stratify. Individual
  neighbors in similar houses get very different values. Some are right and some
  are badly wrong, and the aggregate cannot tell you which.
- **Hilldale** — the modest homes are assessed at a higher percentage of their
  value than the expensive ones. Median and COD both pass. Only PRD catches it.
  This is the most serious failure of the three, because it falls hardest on the
  owners least able to absorb it.

### Other deliberate content

| What | How much | Why it is there |
|---|---|---|
| Non-arm's-length deeds | 35 of 244 | Quit claims, sheriff's deeds, estate transfers, $1 interfamily gifts. **Section 9 must filter these** before computing anything — Policy 7.3. |
| Delinquent tax accounts | 37 of 1,200 (3.1%) | Section 10 needs real delinquencies, not just the one hero parcel. |
| Partial payments | ~3.4% | `status = 'PARTIAL'` — a balance that is not zero but not a full delinquency either. An agent that only checks `status = 'DELINQUENT'` gets the wrong answer. |
| Commercial + vacant parcels | 91 of 1,200 (7.6%) | A ratio study is residential-improved only. Mixing classes is a classic beginner error, so the data makes it possible to commit. |

---

## ⭐ The hero parcel: `04-127-3891`

One parcel is hand-built and threads the entire talk.

**Section 2** already showed the audience this parcel being chopped into
tokens — `04-127-3891` on its own is **11**, and `parcel 04-127-3891` is
**13** — proof that the model sees fragments, not an identifier. **Section 10**
then asks whether it can be split. Same parcel, 80 minutes apart.

| System | What it says |
|---|---|
| 🏛️ **Appraiser** | Biff Tannen · 1955 Hilldale Ct · 1,640 sq ft on a **15,800 sq ft lot** · assessed **$412,500** |
| 📜 **Clerk** | Warranty Deed `2025-000417`, recorded 2025-03-14, **$437,000** from Gerald Strickland |
| 💵 **Tax Collector** | **DELINQUENT** — $5,133.72 for 2025 **plus** $3,298.41 still owing from 2024 = **$8,432.13** |

### Why it makes a good demo

Walk the logic the way the agent has to:

1. **Is the lot big enough to split?** Hilldale is zoned R-1, minimum 6,000 sq ft
   per resulting parcel (Policy 4.3). At 15,800 sq ft it splits into two 7,900
   sq ft lots. ✅ **Passes.**
2. **Does he own it?** The Clerk confirms he took title in March 2025 under a
   Warranty Deed. ✅ **Passes.**
3. **Are the taxes clear?** ❌ **No.** $8,432.13 outstanding across two years.

**Policy 4.2 bars the split.** Not because of the lot, not because of ownership
— only the taxes. The agent has to check all three systems *and* read the policy
to get there, and it has to report the specific blocking reason rather than a
bare yes/no.

Note also that **the assessed value ($412,500) is the same number Section 2's
model does arithmetic on** — and gets wrong by $2,000. That is not a coincidence
either.

---

## 📋 Schema

### `cama.db` → `parcels`
`parcel_id` · `situs_address` · `owner_name` · `neighborhood_code` ·
`neighborhood` · `property_class` · `property_class_desc` · `year_built` ·
`living_area_sqft` · `lot_size_sqft` · `bedrooms` · `bathrooms` ·
`assessed_value` · `assessment_year` · `latitude` · `longitude`

### `clerk.db` → `sales`
`instrument_number` · `parcel_id` · `recorded_date` · `sale_price` · `grantor` ·
`grantee` · `deed_type` · `arms_length` · `book` · `page`

### `tax.db` → `tax_accounts`
`parcel_id` · `tax_year` · `owner_name` · `taxable_value` · `levy_amount` ·
`amount_paid` · `balance_due` · `status` · `last_payment_date` ·
`prior_year_delinquent` · `prior_year_balance`

> **Note:** the generator computes a true `market_value` for each parcel in
> order to produce honest sale prices and reproducible flaws. **It is
> deliberately not written to any database.** A real assessor does not have a
> column containing the right answer, and neither should the demos.

---

## 📄 Office policy documents → [`policy/`](policy/)

The corpus for **Section 8's** RAG demo. Four documents, written as a real
office would write them:

| Document | Contains | Used by |
|---|---|---|
| [`parcel_split_and_combination.md`](policy/parcel_split_and_combination.md) | **Policy 4.2 — the tax balance rule that blocks Biff's split**, minimum lot sizes, effective dates | §8, §10 |
| [`sales_qualification.md`](policy/sales_qualification.md) | Which deed types count as arm's length, disqualifying conditions, minimum sample size | §9, §8 |
| [`ratio_study_standards.md`](policy/ratio_study_standards.md) | COD, PRD and median ratio standards; why stratification is mandatory; corrective action | §9, §8 |
| [`assessment_calendar.md`](policy/assessment_calendar.md) | Assessment date, notice mailing, **the July 25 appeal deadline**, roll certification | §8 |

**Markdown, not PDF** — deliberately. Markdown renders on GitHub, so an attendee
can read the corpus without downloading anything, and diffs are legible. The
Section 8 loader accepts `.md` *and* `.pdf`, so you can drop your own office's
real policy PDFs into this folder and re-index without touching a line of code.

These documents interlock with the databases on purpose: **§8 retrieves the
rule, §10 retrieves the facts, and the agent combines them into a decision.**

---

## Reading it from Python

```python
from delorean import data

data.check()          # generates the databases if they are missing
data.systems()        # print the three offices and what each one knows

with data.cama() as con:
    row = con.execute(
        "SELECT * FROM parcels WHERE parcel_id = ?", ("04-127-3891",)
    ).fetchone()
    print(row["owner_name"], row["assessed_value"])

df = data.parcels_df()     # or as a DataFrame, for the ratio study
```

There is no function that queries all three at once. That is not an oversight.
