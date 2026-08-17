"""
generate_hill_valley_data.py — build the synthetic Hill Valley County dataset.

    uv run python scripts/generate_hill_valley_data.py

This writes THREE separate SQLite databases into data/, on purpose:

    data/cama.db     The Appraiser's office     -> parcels, characteristics, values
    data/clerk.db    The Clerk of Courts        -> recorded deeds and sale prices
    data/tax.db      The Tax Collector          -> billing, payments, balances

They are separate files with no foreign keys between them because that is the
actual situation in a real county. Three elected offices, three vendors, three
databases, and no join. You cannot answer "may this parcel be split?" from any
one of them — which is exactly the point Section 10 makes about MCP.

EVERYTHING HERE IS INVENTED. No real parcel, owner, address, or sale appears in
this repository. The coordinates are real Calgary-area coordinates so the maps
in Section 7 render over actual streets, but every property sitting on them is
fictional.

The data is DETERMINISTIC. Same seed, same database, every time — so the outputs
committed in the notebooks always match what your machine produces.

---------------------------------------------------------------------------
The flaws in this data are deliberate. See data/README.md for the full list.
Assessment data with no problems in it makes Section 7's ratio study pointless:
there would be nothing for the agent to find. Each neighborhood below is built
to fail (or pass) a different IAAO test.
---------------------------------------------------------------------------
"""

from __future__ import annotations

import csv
import math
import random
import sqlite3
from datetime import date, timedelta
from pathlib import Path

# Everything is generated from this one number. Change it and you get a
# different-but-equally-valid county; leave it alone and everyone who clones
# this repo gets byte-identical data.
SEED = 1985

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
CSV_DIR = DATA_DIR / "csv"

N_PARCELS = 1_200
TAX_YEAR = 2025

# The parcel the audience already met. Section 2 chops "parcel 04-127-3891"
# into 13 tokens to show that the model sees fragments, not an identifier.
# It becomes the star of Section 10, so it has to actually exist here.
HERO_PARCEL = "04-127-3891"


# ─────────────────────────────────────────────────────────────────────────
#  The six neighborhoods, and the appraisal problem each one is built to have
#
#  IAAO measures three different things, and they are genuinely different
#  questions. This dataset gives Section 7 one clear example of each:
#
#    LEVEL       Are we assessing at the right percentage of market value?
#                -> measured by the MEDIAN RATIO (assessed / sale price)
#    UNIFORMITY  Are similar properties assessed consistently with each other?
#                -> measured by COD. IAAO: 15 or under for improved residential.
#    EQUITY      Are cheap and expensive homes treated the same?
#                -> measured by PRD. IAAO: 0.98 to 1.03.
#                   Above 1.03 is REGRESSIVE: the cheap homes carry too much.
#
#  `assess_sigma` is how inconsistent the assessments are (drives COD).
#  `regressivity` tilts ratios against lower-value homes (drives PRD).
# ─────────────────────────────────────────────────────────────────────────
NEIGHBORHOODS = [
    {
        "code": "LYON",
        "name": "Lyon Estates",
        "blurb": "1950s-60s tract housing. The healthy baseline.",
        "lat": 51.0385, "lon": -114.0330,
        "price_per_sqft": 385, "year_range": (1952, 1968),
        "sqft_range": (1_050, 1_950), "lot_range": (5_500, 8_500),
        "share": 0.22,
        "level": 0.955,       # right on target
        "assess_sigma": 0.072,  # -> COD near 7. Comfortably passing.
        "regressivity": 0.0,
        "planted_issue": None,
    },
    {
        "code": "TWIN",
        "name": "Twin Pines Ranch",
        "blurb": "Large-lot acreage on the west edge. Highest values.",
        "lat": 51.0455, "lon": -114.0520,
        "price_per_sqft": 505, "year_range": (1978, 1996),
        "sqft_range": (1_900, 3_600), "lot_range": (14_000, 40_000),
        "share": 0.12,
        "level": 0.948,
        "assess_sigma": 0.081,
        "regressivity": 0.0,
        "planted_issue": None,
    },
    {
        "code": "EAST",
        "name": "Eastwood Ravine",
        "blurb": "Newest subdivision. Recent construction, easy to value.",
        "lat": 51.0250, "lon": -113.9860,
        "share": 0.18,
        "price_per_sqft": 442, "year_range": (2004, 2021),
        "sqft_range": (1_500, 2_900), "lot_range": (4_200, 7_000),
        "level": 0.962,
        "assess_sigma": 0.066,  # newest + most uniform -> lowest COD
        "regressivity": 0.0,
        "planted_issue": None,
    },
    {
        "code": "RIVR",
        "name": "Riverside Heights",
        "blurb": "PLANTED LEVEL PROBLEM - assessed well below market.",
        "lat": 51.0180, "lon": -114.0105,
        "share": 0.16,
        "price_per_sqft": 610, "year_range": (1988, 2012),
        "sqft_range": (1_700, 3_200), "lot_range": (6_000, 11_000),
        "level": 0.842,       # <- the flaw. Values ran away from the roll.
        "assess_sigma": 0.078,
        "regressivity": 0.0,
        "planted_issue": "level",
    },
    {
        "code": "CTSQ",
        "name": "Courthouse Square",
        "blurb": "PLANTED UNIFORMITY PROBLEM - historic stock, wildly inconsistent.",
        "lat": 51.0455, "lon": -114.0080,
        "share": 0.14,
        "price_per_sqft": 428, "year_range": (1901, 1946),
        "sqft_range": (900, 2_600), "lot_range": (3_000, 7_500),
        "level": 0.951,       # level is FINE - that is what makes it instructive
        "assess_sigma": 0.300,  # <- the flaw. -> COD near 20, well over 15.
        "regressivity": 0.0,
        "planted_issue": "uniformity",
    },
    {
        "code": "HILL",
        "name": "Hilldale",
        "blurb": "PLANTED EQUITY PROBLEM - regressive. Modest homes over-assessed.",
        "lat": 51.0295, "lon": -113.9975,
        "share": 0.18,
        "price_per_sqft": 336, "year_range": (1962, 1984),
        "sqft_range": (860, 2_100), "lot_range": (4_000, 7_200),
        "level": 0.938,
        "assess_sigma": 0.084,
        "regressivity": 0.430,  # <- the flaw. -> PRD above 1.03.
        "planted_issue": "equity",
    },
]

# Property classes. The commercial and vacant parcels exist so that Section 7
# has to FILTER before it computes anything -- a ratio study is residential
# improved only, and mixing classes is a classic beginner error.
# (code, weight, description) -- weight second, so weighted_choice() can read
# option[1] the same way for every table in this file.
CLASSES = [
    ("R01", 0.86, "Single Family Residential"),
    ("R02", 0.07, "Residential Condominium"),
    ("C01", 0.04, "Commercial Improved"),
    ("V01", 0.03, "Vacant Residential Land"),
]

STREETS = [
    "Riverside Dr", "Hill Valley Rd", "Lyon Dr", "Oak Park Ave", "Maple Ave",
    "Elm St", "John F Kennedy Dr", "Courthouse Sq", "Clayton Ridge Rd",
    "Twin Pines Way", "Eastwood Ln", "Hilldale Ct", "Chester Ave",
    "Strickland Blvd", "Parkview Ter", "Gale Cres", "Baines Way", "Peabody Farm Rd",
]

FIRST_NAMES = [
    "George", "Lorraine", "Marty", "Jennifer", "Emmett", "Clara", "Dave", "Linda",
    "Goldie", "Marvin", "Otis", "Gerald", "Stella", "Sam", "Milton", "Joey",
    "Arthur", "Beverly", "Carol", "Daniel", "Elaine", "Frank", "Grace", "Harold",
    "Irene", "Jack", "Karen", "Leonard", "Marcia", "Neil", "Olive", "Patrick",
    "Rita", "Stephen", "Theresa", "Victor", "Wendy", "Alan", "Bonnie", "Curtis",
    "Diane", "Edward", "Fiona", "Gordon", "Helen", "Ivan", "Janet", "Keith",
]

LAST_NAMES = [
    "McFly", "Tannen", "Brown", "Baines", "Strickland", "Clayton", "Parker",
    "Sanchez", "Peabody", "Wilson", "Statler", "Caruthers", "Von Braun",
    "Anderson", "Bennett", "Chen", "Dubois", "Evans", "Fletcher", "Gagnon",
    "Hendricks", "Ibrahim", "Johansson", "Kowalski", "Lindqvist", "Moreau",
    "Nakamura", "O'Brien", "Patel", "Quinn", "Rossi", "Silva", "Thompson",
    "Ueda", "Vasquez", "Whitfield", "Yamada", "Zielinski", "Achebe", "Bergeron",
]

# Deed types the Clerk actually records. Only some of them represent an
# arm's-length sale, which is why Section 7 has to qualify sales before
# computing a ratio study.
DEED_TYPES = [
    # (type, weight, is_arms_length)
    ("Warranty Deed", 0.74, True),
    ("Grant Deed", 0.11, True),
    ("Quit Claim Deed", 0.07, False),      # usually family, not market
    ("Sheriff's Deed", 0.03, False),       # foreclosure
    ("Executor's Deed", 0.03, False),      # estate
    ("Interfamily Transfer", 0.02, False),  # gift / $1
]


# ─────────────────────────────────────────────────────────────────────────
#  Small helpers
# ─────────────────────────────────────────────────────────────────────────

def weighted_choice(rng: random.Random, options: list[tuple]) -> tuple:
    """Pick one (value, weight, ...) tuple, respecting the weights."""
    total = sum(o[1] for o in options)
    roll = rng.uniform(0, total)
    running = 0.0
    for opt in options:
        running += opt[1]
        if roll <= running:
            return opt
    return options[-1]


def jitter(rng: random.Random, sigma: float) -> float:
    """A multiplicative wobble centered on 1.0.

    Uses a lognormal so the result is never negative and so being 20% high is
    the same size of error as being 20% low -- which is how property value
    error actually behaves.
    """
    return math.exp(rng.gauss(0.0, sigma))


def money(value: float, nearest: int = 100) -> int:
    """Round to whole dollars the way an assessment roll would."""
    return int(round(value / nearest) * nearest)


# ─────────────────────────────────────────────────────────────────────────
#  1. The Appraiser's office (CAMA)
# ─────────────────────────────────────────────────────────────────────────

def build_parcels(rng: random.Random) -> list[dict]:
    """Generate the parcel roll: characteristics, market value, assessed value.

    `market_value` is the TRUTH -- what the property is really worth. Real
    assessors never see this column; it exists here so we can generate honest
    sale prices from it and so the planted flaws are reproducible.
    `assessed_value` is what the office has on the books, flaws included.
    """
    parcels: list[dict] = []
    used_ids: set[str] = set()

    # How many parcels land in each neighborhood.
    counts = {n["code"]: int(N_PARCELS * n["share"]) for n in NEIGHBORHOODS}
    counts[NEIGHBORHOODS[0]["code"]] += N_PARCELS - sum(counts.values())

    # First pass: characteristics and true market value.
    for hood in NEIGHBORHOODS:
        for _ in range(counts[hood["code"]]):
            # Parcel id, format NN-NNN-NNNN.
            while True:
                pid = (f"{rng.randint(1, 12):02d}-{rng.randint(100, 499):03d}"
                       f"-{rng.randint(1000, 9999):04d}")
                if pid not in used_ids:
                    used_ids.add(pid)
                    break

            cls_code, _weight, cls_name = weighted_choice(rng, CLASSES)

            year_built = rng.randint(*hood["year_range"])
            if cls_code == "V01":
                living_area = 0
                beds = baths = 0
                year_built = 0
            else:
                living_area = rng.randint(*hood["sqft_range"])
                beds = max(1, min(6, round(living_area / 520)))
                baths = round(max(1.0, min(4.5, living_area / 900)) * 2) / 2

            lot_size = rng.randint(*hood["lot_range"])

            # Market value: base rate x size, adjusted for age and lot.
            if cls_code == "V01":
                market = lot_size * rng.uniform(11.0, 19.0)
            else:
                age = TAX_YEAR - year_built
                # Older costs less, but the curve flattens -- a 90-year-old
                # house is not worth half a 45-year-old one.
                age_factor = 1.0 - 0.0042 * min(age, 85)
                lot_factor = 1.0 + 0.10 * (lot_size / sum(hood["lot_range"]) * 2 - 1.0)
                market = (living_area * hood["price_per_sqft"]
                          * age_factor * lot_factor * jitter(rng, 0.11))
                if cls_code == "R02":
                    market *= 0.78          # condos trade below detached
                elif cls_code == "C01":
                    market *= 1.55

            # Address and owner.
            street = rng.choice(STREETS)
            number = rng.randint(100, 3_499)
            owner = f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}"

            # Coordinates: scatter around the neighborhood center. About
            # 0.012 degrees of latitude is a little over a kilometre.
            lat = hood["lat"] + rng.gauss(0, 0.0075)
            lon = hood["lon"] + rng.gauss(0, 0.0110)

            parcels.append({
                "parcel_id": pid,
                "situs_address": f"{number} {street}",
                "owner_name": owner,
                "neighborhood_code": hood["code"],
                "neighborhood": hood["name"],
                "property_class": cls_code,
                "property_class_desc": cls_name,
                "year_built": year_built,
                "living_area_sqft": living_area,
                "lot_size_sqft": lot_size,
                "bedrooms": beds,
                "bathrooms": baths,
                "latitude": round(lat, 6),
                "longitude": round(lon, 6),
                "_market_value": market,
                "_hood": hood,
            })

    # Second pass: assessed value. This is where the flaws get planted, and it
    # has to be a second pass because regressivity is measured against the
    # neighborhood MEDIAN, which we only know once every parcel exists.
    for hood in NEIGHBORHOODS:
        in_hood = [p for p in parcels if p["neighborhood_code"] == hood["code"]]
        median_mv = sorted(p["_market_value"] for p in in_hood)[len(in_hood) // 2]

        for p in in_hood:
            ratio = hood["level"] * jitter(rng, hood["assess_sigma"])

            # Regressivity: push the ratio UP on homes below the neighborhood
            # median and DOWN on homes above it. That is what makes a modest
            # house carry more than its share of the levy, and it is precisely
            # what PRD is designed to detect.
            if hood["regressivity"]:
                tilt = (median_mv / p["_market_value"]) ** hood["regressivity"]
                ratio *= tilt

            p["assessed_value"] = money(p["_market_value"] * ratio, 100)
            p["assessment_year"] = TAX_YEAR

    return parcels


def install_hero_parcel(parcels: list[dict], rng: random.Random) -> dict:
    """Make parcel 04-127-3891 real, and hand it to Biff Tannen.

    Section 2 already showed the audience this exact string being chopped into
    13 tokens. Section 10 then asks whether it may be split. Reusing the same
    parcel across 80 minutes of talk is free continuity, so it is worth
    reserving one record for it by hand.
    """
    hood = next(h for h in NEIGHBORHOODS if h["code"] == "HILL")

    # If the random generator happened to mint this id already, drop that one
    # first. Otherwise we would end up with two rows sharing a primary key the
    # moment somebody changes SEED.
    for dup in [p for p in parcels if p["parcel_id"] == HERO_PARCEL]:
        parcels.remove(dup)

    victim = next(p for p in parcels if p["neighborhood_code"] == "HILL"
                  and p["property_class"] == "R01")

    victim.update({
        "parcel_id": HERO_PARCEL,
        # 1955 is not an accident.
        "situs_address": "1955 Hilldale Ct",
        "owner_name": "Biff Tannen",
        "year_built": 1971,
        "living_area_sqft": 1_640,
        "lot_size_sqft": 15_800,     # big lot -- which is WHY a split is plausible
        "bedrooms": 3,
        "bathrooms": 2.0,
        "latitude": round(hood["lat"] + 0.0031, 6),
        "longitude": round(hood["lon"] - 0.0042, 6),
        "assessed_value": 412_500,   # the number Section 2 does bad math on
        "assessment_year": TAX_YEAR,
    })
    victim["_market_value"] = 445_000.0
    return victim


# ─────────────────────────────────────────────────────────────────────────
#  2. The Clerk of Courts (recorded deeds)
# ─────────────────────────────────────────────────────────────────────────

def build_sales(rng: random.Random, parcels: list[dict]) -> list[dict]:
    """Generate recorded deeds for roughly a fifth of the roll.

    Note what is NOT here: no assessed value, no characteristics, no
    neighborhood. The Clerk records instruments, not appraisals. If you want
    to know whether a sale price was reasonable, you have to go get the
    assessment from a different office -- which is the whole problem.
    """
    sales: list[dict] = []
    window_start = date(2024, 1, 1)
    window_days = (date(2025, 12, 31) - window_start).days

    sellable = [p for p in parcels if p["property_class"] != "V01"]
    n_sales = int(len(sellable) * 0.21)

    for n, parcel in enumerate(rng.sample(sellable, n_sales), start=1):
        deed_type, _weight, arms_length = weighted_choice(rng, DEED_TYPES)
        sale_date = window_start + timedelta(days=rng.randint(0, window_days))

        if arms_length:
            # A real sale lands near market value, with normal market noise.
            price = money(parcel["_market_value"] * jitter(rng, 0.052), 500)
        elif deed_type == "Interfamily Transfer":
            price = rng.choice([1, 10, 100])          # nominal consideration
        elif deed_type == "Sheriff's Deed":
            price = money(parcel["_market_value"] * rng.uniform(0.52, 0.74), 500)
        else:
            price = money(parcel["_market_value"] * rng.uniform(0.60, 0.93), 500)

        grantee = parcel["owner_name"]
        grantor = f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}"

        sales.append({
            "instrument_number": f"{sale_date.year}-{n:06d}",
            "parcel_id": parcel["parcel_id"],
            "recorded_date": sale_date.isoformat(),
            "sale_price": price,
            "grantor": grantor,
            "grantee": grantee,
            "deed_type": deed_type,
            # The Clerk records the instrument. Whether it is a usable market
            # sale is the ASSESSOR's determination, so this column is the
            # office's own qualification flag, not the Clerk's.
            "arms_length": 1 if arms_length else 0,
            "book": rng.randint(1_100, 1_480),
            "page": rng.randint(1, 899),
        })

    sales.sort(key=lambda s: s["recorded_date"])
    return sales


def install_hero_deed(sales: list[dict], hero: dict) -> dict:
    """Biff took title in March 2025. Section 10 verifies this against CAMA."""
    deed = {
        "instrument_number": "2025-000417",
        "parcel_id": HERO_PARCEL,
        "recorded_date": "2025-03-14",
        "sale_price": 437_000,
        "grantor": "Gerald Strickland",
        "grantee": "Biff Tannen",
        "deed_type": "Warranty Deed",
        "arms_length": 1,
        "book": 1_402,
        "page": 318,
    }
    sales[:] = [s for s in sales if s["parcel_id"] != HERO_PARCEL]
    sales.append(deed)
    sales.sort(key=lambda s: s["recorded_date"])
    return deed


# ─────────────────────────────────────────────────────────────────────────
#  3. The Tax Collector (billing and balances)
# ─────────────────────────────────────────────────────────────────────────

def build_tax_accounts(rng: random.Random, parcels: list[dict]) -> list[dict]:
    """Generate the tax roll: what was billed, what was paid, what is owed.

    Again, note the absence: no characteristics, no sale history. The Tax
    Collector bills against a value handed over by the Assessor and then
    tracks money. That is all.
    """
    accounts: list[dict] = []
    mill_rate = 0.01142   # about $11.42 per $1,000 of assessed value

    for p in parcels:
        levy = round(p["assessed_value"] * mill_rate, 2)

        roll = rng.random()
        if roll < 0.938:
            status, paid = "PAID", levy
        elif roll < 0.972:
            status, paid = "PARTIAL", round(levy * rng.uniform(0.15, 0.80), 2)
        else:
            status, paid = "DELINQUENT", 0.0

        balance = round(levy - paid, 2)
        # Delinquent accounts accrue interest at 1% a month.
        if status == "DELINQUENT":
            balance = round(balance * 1.09, 2)

        if status == "PAID":
            last_payment = date(TAX_YEAR, rng.randint(1, 11), rng.randint(1, 28)).isoformat()
        elif status == "PARTIAL":
            last_payment = date(TAX_YEAR, rng.randint(1, 8), rng.randint(1, 28)).isoformat()
        else:
            last_payment = None

        accounts.append({
            "parcel_id": p["parcel_id"],
            "tax_year": TAX_YEAR,
            "owner_name": p["owner_name"],
            "taxable_value": p["assessed_value"],
            "levy_amount": levy,
            "amount_paid": paid,
            "balance_due": balance,
            "status": status,
            "last_payment_date": last_payment,
            "prior_year_delinquent": 0,
            "prior_year_balance": 0.0,
        })

    return accounts


def install_hero_tax(accounts: list[dict], hero: dict) -> dict:
    """Biff owes money. This single record is what blocks the parcel split.

    Two years delinquent, so the agent's answer is not merely "there is a
    balance" but "there is a balance going back to 2024" -- which is a
    materially different conversation with the taxpayer.
    """
    acct = next(a for a in accounts if a["parcel_id"] == HERO_PARCEL)
    acct.update({
        "levy_amount": 4_710.75,
        "amount_paid": 0.0,
        "balance_due": 5_133.72,        # 2025 levy plus interest
        "status": "DELINQUENT",
        "last_payment_date": None,
        "prior_year_delinquent": 1,
        "prior_year_balance": 3_298.41,  # 2024, still outstanding
    })
    acct["total_due"] = round(acct["balance_due"] + acct["prior_year_balance"], 2)
    return acct


# ─────────────────────────────────────────────────────────────────────────
#  Writing it out
# ─────────────────────────────────────────────────────────────────────────

def write_db(path: Path, table: str, rows: list[dict], schema: str,
             columns: list[str]) -> None:
    """Create one database with one table and fill it."""
    path.unlink(missing_ok=True)
    con = sqlite3.connect(path)
    con.execute(schema)
    placeholders = ",".join("?" for _ in columns)
    con.executemany(
        f"INSERT INTO {table} ({','.join(columns)}) VALUES ({placeholders})",
        [tuple(r.get(c) for c in columns) for r in rows],
    )
    con.execute(f"CREATE INDEX idx_{table}_parcel ON {table}(parcel_id)")
    con.commit()
    con.close()


def write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    """Mirror each table as CSV so GitHub can render it without SQLite."""
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


# ─────────────────────────────────────────────────────────────────────────
#  Verification — prove the planted flaws actually landed
# ─────────────────────────────────────────────────────────────────────────

def ratio_stats(pairs: list[tuple[float, float]]) -> dict:
    """Compute the three IAAO measures from (assessed, sale_price) pairs."""
    ratios = sorted(a / s for a, s in pairs)
    n = len(ratios)
    median = (ratios[n // 2] if n % 2
              else (ratios[n // 2 - 1] + ratios[n // 2]) / 2)
    mean = sum(ratios) / n
    cod = 100 * (sum(abs(r - median) for r in ratios) / n) / median
    weighted_mean = sum(a for a, _ in pairs) / sum(s for _, s in pairs)
    return {"n": n, "median": median, "mean": mean,
            "cod": cod, "prd": mean / weighted_mean}


def verify(parcels: list[dict], sales: list[dict]) -> None:
    """Print the ratio study so we can SEE that the flaws are where we put them."""
    by_id = {p["parcel_id"]: p for p in parcels}
    qualified = [s for s in sales
                 if s["arms_length"] == 1
                 and by_id[s["parcel_id"]]["property_class"] in ("R01", "R02")]

    print("\n  Ratio study by neighborhood (qualified residential sales only)")
    print("  " + "-" * 74)
    print(f"  {'Neighborhood':<20} {'n':>4} {'Median':>8} {'COD':>7} {'PRD':>7}   Finding")
    print("  " + "-" * 74)

    for hood in NEIGHBORHOODS:
        pairs = [(by_id[s["parcel_id"]]["assessed_value"], s["sale_price"])
                 for s in qualified
                 if by_id[s["parcel_id"]]["neighborhood_code"] == hood["code"]]
        if len(pairs) < 5:
            continue
        st = ratio_stats(pairs)

        flags = []
        if not 0.90 <= st["median"] <= 1.10:
            flags.append("LEVEL")
        if st["cod"] > 15.0:
            flags.append("COD")
        if not 0.98 <= st["prd"] <= 1.03:
            flags.append("PRD")
        finding = ", ".join(flags) if flags else "passes"

        print(f"  {hood['name']:<20} {st['n']:>4} {st['median']:>8.3f} "
              f"{st['cod']:>7.2f} {st['prd']:>7.3f}   {finding}")

    print("  " + "-" * 74)
    overall = ratio_stats([(by_id[s["parcel_id"]]["assessed_value"], s["sale_price"])
                           for s in qualified])
    print(f"  {'COUNTYWIDE':<20} {overall['n']:>4} {overall['median']:>8.3f} "
          f"{overall['cod']:>7.2f} {overall['prd']:>7.3f}")
    print("\n  IAAO: median 0.90-1.10 | COD <= 15 (improved residential) | PRD 0.98-1.03")


# ─────────────────────────────────────────────────────────────────────────

PARCEL_COLS = [
    "parcel_id", "situs_address", "owner_name", "neighborhood_code", "neighborhood",
    "property_class", "property_class_desc", "year_built", "living_area_sqft",
    "lot_size_sqft", "bedrooms", "bathrooms", "assessed_value", "assessment_year",
    "latitude", "longitude",
]

SALE_COLS = [
    "instrument_number", "parcel_id", "recorded_date", "sale_price", "grantor",
    "grantee", "deed_type", "arms_length", "book", "page",
]

TAX_COLS = [
    "parcel_id", "tax_year", "owner_name", "taxable_value", "levy_amount",
    "amount_paid", "balance_due", "status", "last_payment_date",
    "prior_year_delinquent", "prior_year_balance",
]


def main() -> None:
    rng = random.Random(SEED)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    CSV_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    # ASCII only in console output: the default Windows terminal codepage
    # mangles em-dashes, and attendees will run this on Windows.
    print(" Hill Valley County - generating three independent office systems")
    print("=" * 78)

    parcels = build_parcels(rng)
    hero = install_hero_parcel(parcels, rng)

    sales = build_sales(rng, parcels)
    install_hero_deed(sales, hero)

    accounts = build_tax_accounts(rng, parcels)
    install_hero_tax(accounts, hero)

    write_db(
        DATA_DIR / "cama.db", "parcels", parcels,
        """CREATE TABLE parcels (
               parcel_id TEXT PRIMARY KEY, situs_address TEXT, owner_name TEXT,
               neighborhood_code TEXT, neighborhood TEXT, property_class TEXT,
               property_class_desc TEXT, year_built INTEGER,
               living_area_sqft INTEGER, lot_size_sqft INTEGER,
               bedrooms INTEGER, bathrooms REAL, assessed_value INTEGER,
               assessment_year INTEGER, latitude REAL, longitude REAL)""",
        PARCEL_COLS,
    )
    write_db(
        DATA_DIR / "clerk.db", "sales", sales,
        """CREATE TABLE sales (
               instrument_number TEXT PRIMARY KEY, parcel_id TEXT,
               recorded_date TEXT, sale_price INTEGER, grantor TEXT,
               grantee TEXT, deed_type TEXT, arms_length INTEGER,
               book INTEGER, page INTEGER)""",
        SALE_COLS,
    )
    write_db(
        DATA_DIR / "tax.db", "tax_accounts", accounts,
        """CREATE TABLE tax_accounts (
               parcel_id TEXT PRIMARY KEY, tax_year INTEGER, owner_name TEXT,
               taxable_value INTEGER, levy_amount REAL, amount_paid REAL,
               balance_due REAL, status TEXT, last_payment_date TEXT,
               prior_year_delinquent INTEGER, prior_year_balance REAL)""",
        TAX_COLS,
    )

    write_csv(CSV_DIR / "parcels.csv", parcels, PARCEL_COLS)
    write_csv(CSV_DIR / "sales.csv", sales, SALE_COLS)
    write_csv(CSV_DIR / "tax_accounts.csv", accounts, TAX_COLS)

    print(f"\n  cama.db   {len(parcels):>5} parcels        (Appraiser's office)")
    print(f"  clerk.db  {len(sales):>5} recorded deeds (Clerk of Courts)")
    print(f"  tax.db    {len(accounts):>5} tax accounts   (Tax Collector)")

    delinquent = [a for a in accounts if a["status"] == "DELINQUENT"]
    print(f"\n  Delinquent accounts: {len(delinquent)} "
          f"({100 * len(delinquent) / len(accounts):.1f}%)")

    verify(parcels, sales)

    print(f"\n  Hero parcel {HERO_PARCEL} - the Section 10 split scenario")
    print(f"    CAMA  : {hero['owner_name']}, {hero['situs_address']}, "
          f"{hero['lot_size_sqft']:,} sqft lot, assessed ${hero['assessed_value']:,}")
    deed = next(s for s in sales if s["parcel_id"] == HERO_PARCEL)
    print(f"    Clerk : {deed['deed_type']} {deed['recorded_date']}, "
          f"${deed['sale_price']:,} from {deed['grantor']}")
    acct = next(a for a in accounts if a["parcel_id"] == HERO_PARCEL)
    total = acct["balance_due"] + acct["prior_year_balance"]
    print(f"    Tax   : {acct['status']}, ${acct['balance_due']:,.2f} for {TAX_YEAR} "
          f"+ ${acct['prior_year_balance']:,.2f} prior = ${total:,.2f} TOTAL DUE")
    print("\n  Done.\n")


if __name__ == "__main__":
    main()
