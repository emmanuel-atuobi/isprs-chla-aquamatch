"""
Spatial (HUC4) station-holdout validation: 35 repeated splits with a fixed XGBoost
configuration, grouping stations by HUC4 watershed so no station appears in both the
training and test set of a given split. Produces HUC4_fixed_35seed_metrics.csv and the
35 station manifests that the HUC4 feature-group ablation and Fig. 5 station map reuse.
"""
import os
from pathlib import Path
import json
import time
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from xgboost import XGBRegressor

from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
)

from scipy.stats import linregress, t

warnings.filterwarnings("ignore")


# =============================================================================
# CONFIGURATION
# =============================================================================

SEEDS = list(range(1, 36))

MODEL_SEED = 42
TEST_FRACTION = 0.25
COORD_DECIMALS = 6

DATA_DIR = Path(os.environ.get("AQUAMATCH_DATA_DIR", "./data"))
RESULTS_DIR = Path(os.environ.get("AQUAMATCH_RESULTS_DIR", "./results"))

INPUT = (
    DATA_DIR
    / "spatial_validation_diagnostics"
    / "huc8_field_comparison.csv"
)

OUT = (
    RESULTS_DIR
    / "spatial_validation_diagnostics"
    / "repeated_HUC4_fixed_35seeds"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)

(OUT / "splits").mkdir(
    parents=True,
    exist_ok=True,
)

(OUT / "predictions").mkdir(
    parents=True,
    exist_ok=True,
)

(OUT / "plots").mkdir(
    parents=True,
    exist_ok=True,
)


# =============================================================================
# FINAL 22 FEATURES
# =============================================================================

FEATURES = [

    "red",
    "nir",
    "blue",

    "lat",
    "long",

    "NDVI",
    "NDTI",
    "RNI",
    "GBI",
    "BLRDGR",
    "GNRI",
    "RBI",
    "NIRGI",
    "GDVI",
    "NDAVI",
    "FAI",
    "MNDWI",
    "SWI",
    "TGI",
    "AFAI",

    "sin_doy",
    "cos_doy",
]


# =============================================================================
# FIXED XGBOOST CONFIGURATION
#
# Fixed configuration used for the reported HUC4-stratified station-level
# validation (not retuned per split).
# =============================================================================

FIXED_PARAMS = {

    "colsample_bytree":
        0.8,

    "gamma":
        0.5,

    "learning_rate":
        0.02,

    "max_bin":
        512,

    "max_depth":
        7,

    "min_child_weight":
        1,

    "n_estimators":
        2200,

    "reg_alpha":
        0.1,

    "reg_lambda":
        4.0,

    "subsample":
        0.8,

    "objective":
        "reg:squarederror",

    "eval_metric":
        "rmse",

    "random_state":
        MODEL_SEED,

    "tree_method":
        "hist",

    "n_jobs":
        16,

    "verbosity":
        0,
}


# =============================================================================
# HELPERS
# =============================================================================

def banner(text):

    print(
        "\n"
        + "=" * 110
    )

    print(text)

    print(
        "=" * 110
    )


def normalize_huc8(series):

    s = (
        series
        .astype("string")
        .str.strip()
        .str.replace(
            r"\.0$",
            "",
            regex=True,
        )
    )

    s = s.str.extract(
        r"(\d+)",
        expand=False,
    )

    return s.str.zfill(8)


def calculate_metrics(
    y_true,
    y_pred,
):

    y_true = np.asarray(
        y_true,
        dtype=float,
    )

    y_pred = np.asarray(
        y_pred,
        dtype=float,
    )

    fit = linregress(
        y_true,
        y_pred,
    )

    return {

        "r2":
            float(
                r2_score(
                    y_true,
                    y_pred,
                )
            ),

        "rmse":
            float(
                np.sqrt(
                    mean_squared_error(
                        y_true,
                        y_pred,
                    )
                )
            ),

        "mae":
            float(
                mean_absolute_error(
                    y_true,
                    y_pred,
                )
            ),

        "bias":
            float(
                np.mean(
                    y_pred
                    - y_true
                )
            ),

        "slope":
            float(
                fit.slope
            ),

        "intercept":
            float(
                fit.intercept
            ),

        "pearson_r":
            float(
                fit.rvalue
            ),
    }


def describe_response(x):

    x = pd.Series(
        x,
        dtype=float,
    )

    return {

        "mean":
            float(
                x.mean()
            ),

        "std":
            float(
                x.std()
            ),

        "median":
            float(
                x.median()
            ),

        "p25":
            float(
                x.quantile(0.25)
            ),

        "p75":
            float(
                x.quantile(0.75)
            ),

        "p95":
            float(
                x.quantile(0.95)
            ),
    }


# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

def add_features(df):

    df = df.copy()


    rename_map = {

        "harmonized_value":
            "chl_a",

        "med_Blue":
            "blue",

        "med_Green":
            "green",

        "med_Red":
            "red",

        "med_Nir":
            "nir",

        "med_Swir1":
            "swir1",

        "med_Swir2":
            "swir2",

        "lon":
            "long",
    }


    for old, new in rename_map.items():

        if (
            old in df.columns
            and
            new not in df.columns
        ):

            df = df.rename(
                columns={
                    old:
                        new
                }
            )


    for col in [

        "chl_a",

        "blue",
        "green",
        "red",
        "nir",
        "swir1",
        "swir2",

        "lat",
        "long",

    ]:

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce",
        )


    eps = np.finfo(float).eps


    # -------------------------------------------------------------------------
    # Feature definitions
    # -------------------------------------------------------------------------

    df["NDVI"] = (
        (df["nir"] - df["red"])
        /
        (
            df["nir"]
            + df["red"]
            + eps
        )
    )


    df["NDTI"] = (
        (df["red"] - df["green"])
        /
        (
            df["red"]
            + df["green"]
            + eps
        )
    )


    df["RNI"] = (
        df["red"]
        /
        (
            df["nir"]
            + eps
        )
    )


    df["GBI"] = (
        df["green"]
        /
        (
            df["blue"]
            + eps
        )
    )


    df["BLRDGR"] = (
        (df["blue"] - df["red"])
        /
        (
            df["green"]
            + eps
        )
    )


    df["GNRI"] = (
        df["green"]
        -
        (
            df["green"]
            /
            (
                df["red"]
                + eps
            )
        )
    )


    df["RBI"] = (
        df["red"]
        /
        (
            df["blue"]
            + eps
        )
    )


    df["NIRGI"] = (
        df["nir"]
        /
        (
            df["green"]
            + eps
        )
    )


    df["GDVI"] = (
        df["nir"]
        -
        df["green"]
    )


    df["NDAVI"] = (
        (df["nir"] - df["blue"])
        /
        (
            df["nir"]
            + df["blue"]
            + eps
        )
    )


    baseline = (

        df["red"]

        +

        (
            (
                df["swir1"]
                - df["red"]
            )

            *

            (0.86 - 0.66)

            /

            (1.60 - 0.66)
        )
    )


    df["FAI"] = (
        df["nir"]
        -
        baseline
    )


    df["MNDWI"] = (
        (
            df["green"]
            - df["swir1"]
        )
        /
        (
            df["green"]
            + df["swir1"]
            + eps
        )
    )


    df["SWI"] = (
        (
            df["nir"]
            - df["swir1"]
        )
        /
        (
            df["nir"]
            + df["swir1"]
            + eps
        )
    )


    df["TGI"] = (
        -0.5
        *
        (
            120
            *
            (
                df["red"]
                - df["green"]
            )
            -
            190
            *
            (
                df["red"]
                - df["blue"]
            )
        )
    )


    df["AFAI"] = (
        (
            df["nir"]
            - df["red"]
        )
        +
        0.5
        *
        (
            df["swir1"]
            - df["red"]
        )
    )


    # -------------------------------------------------------------------------
    # Correct cyclical DOY
    # -------------------------------------------------------------------------

    if "ActivityStartDate" in df.columns:

        date_col = (
            "ActivityStartDate"
        )

    elif "date" in df.columns:

        date_col = (
            "date"
        )

    else:

        raise KeyError(
            "Could not identify date column."
        )


    date = pd.to_datetime(
        df[date_col],
        errors="coerce",
        format="mixed",
        dayfirst=False,
    )


    doy = (
        date.dt.dayofyear
    )


    df["sin_doy"] = np.sin(
        2
        * np.pi
        * (doy - 1)
        / 365.0
    )


    df["cos_doy"] = np.cos(
        2
        * np.pi
        * (doy - 1)
        / 365.0
    )


    df = df.replace(
        [np.inf, -np.inf],
        np.nan,
    )


    return df


# =============================================================================
# EXACT STANDARDIZED HUC4 SPLITTER
#
# Same design as the HUC2/HUC4/HUC6 comparison and nested HUC4 experiment.
#
# 5,147 stations × 0.25 = 1,286.75 -> EXACTLY 1,287 validation stations.
# Allocation is proportional across HUC4s while retaining at least one
# training station in every HUC4 represented in validation.
# =============================================================================

def make_huc4_split(
    station_table,
    seed,
):

    manifest = (
        station_table
        .copy()
        .reset_index(
            drop=True
        )
    )


    target_test = int(
        np.round(
            TEST_FRACTION
            * len(manifest)
        )
    )


    assert target_test == 1287


    manifest[
        "split"
    ] = "train"


    counts = (
        manifest
        .groupby(
            "HUC4",
            dropna=False,
        )
        .size()
        .rename(
            "n_stations"
        )
        .reset_index()
    )


    counts[
        "raw_target"
    ] = (
        TEST_FRACTION
        * counts[
            "n_stations"
        ]
    )


    counts[
        "max_test"
    ] = np.maximum(
        counts[
            "n_stations"
        ]
        - 1,
        0,
    )


    counts[
        "n_test"
    ] = np.floor(
        counts[
            "raw_target"
        ]
    ).astype(int)


    counts[
        "n_test"
    ] = np.minimum(
        counts[
            "n_test"
        ],
        counts[
            "max_test"
        ],
    )


    counts[
        "remainder"
    ] = (
        counts[
            "raw_target"
        ]
        -
        np.floor(
            counts[
                "raw_target"
            ]
        )
    )


    current = int(
        counts[
            "n_test"
        ]
        .sum()
    )


    # -------------------------------------------------------------------------
    # Add stations until exact global target is reached
    # -------------------------------------------------------------------------

    if current < target_test:

        need = (
            target_test
            - current
        )


        while need > 0:

            eligible = counts[
                counts[
                    "n_test"
                ]
                <
                counts[
                    "max_test"
                ]
            ].copy()


            if len(
                eligible
            ) == 0:

                raise RuntimeError(
                    "Unable to reach HUC4 validation target."
                )


            eligible = (
                eligible
                .sort_values(
                    [
                        "remainder",
                        "n_stations",
                        "HUC4",
                    ],
                    ascending=[
                        False,
                        False,
                        True,
                    ],
                )
            )


            for idx in eligible.index:

                if need <= 0:

                    break


                counts.loc[
                    idx,
                    "n_test"
                ] += 1


                need -= 1


    # -------------------------------------------------------------------------
    # Remove stations if above exact target
    # -------------------------------------------------------------------------

    elif current > target_test:

        excess = (
            current
            - target_test
        )


        while excess > 0:

            eligible = counts[
                counts[
                    "n_test"
                ]
                > 0
            ].copy()


            eligible = (
                eligible
                .sort_values(
                    [
                        "remainder",
                        "n_stations",
                        "HUC4",
                    ],
                    ascending=[
                        True,
                        True,
                        True,
                    ],
                )
            )


            for idx in eligible.index:

                if excess <= 0:

                    break


                counts.loc[
                    idx,
                    "n_test"
                ] -= 1


                excess -= 1


    assert int(
        counts[
            "n_test"
        ].sum()
    ) == target_test


    # -------------------------------------------------------------------------
    # Random station selection within HUC4
    # -------------------------------------------------------------------------

    rng = np.random.default_rng(
        seed
    )


    allocation = (
        counts
        .set_index(
            "HUC4"
        )[
            "n_test"
        ]
        .to_dict()
    )


    for huc4, group in manifest.groupby(
        "HUC4",
        sort=True,
    ):

        n_test = int(
            allocation[
                huc4
            ]
        )


        if n_test == 0:

            continue


        chosen = rng.choice(

            group.index.to_numpy(),

            size=
                n_test,

            replace=
                False,
        )


        manifest.loc[
            chosen,
            "split"
        ] = "test"


    assert (
        manifest[
            "split"
        ]
        .eq(
            "test"
        )
        .sum()
        == 1287
    )


    return manifest


# =============================================================================
# LOAD QUALITY-FILTERED DATASET
# =============================================================================

banner(
    "LOADING FINAL QUALITY-FILTERED DATASET"
)


df = pd.read_csv(
    INPUT,
    low_memory=False,
    dtype={
        "HUC8_assigned":
            "string",
    },
)


print(
    f"Rows: "
    f"{len(df):,}"
)


if len(df) != 14125:

    raise RuntimeError(
        f"Expected 14,125 quality-filtered observations; "
        f"found {len(df):,}."
    )


if (
    "lon" in df.columns
    and
    "long" not in df.columns
):

    df = df.rename(
        columns={
            "lon":
                "long"
        }
    )


if (
    "harmonized_value"
    in df.columns
    and
    "chl_a" not in df.columns
):

    df = df.rename(
        columns={
            "harmonized_value":
                "chl_a"
        }
    )


df[
    "lat"
] = pd.to_numeric(
    df[
        "lat"
    ],
    errors="coerce",
)


df[
    "long"
] = pd.to_numeric(
    df[
        "long"
    ],
    errors="coerce",
)


# =============================================================================
# HUC8
# =============================================================================

huc_candidates = [

    "HUC8_assigned",
    "assigned_HUC",
    "HUC8",
    "HUCEightDigitCode",
]


HUC8_FIELD = None


for candidate in huc_candidates:

    if candidate in df.columns:

        HUC8_FIELD = candidate
        break


if HUC8_FIELD is None:

    raise KeyError(
        "Could not find HUC8 field."
    )


print(
    f"HUC8 field: "
    f"{HUC8_FIELD}"
)


df[
    "_HUC8"
] = normalize_huc8(
    df[
        HUC8_FIELD
    ]
)


# =============================================================================
# RECONSTRUCT ATOMIC STATION GROUPS
#
# Same MonitoringLocationIdentifier -> same atomic station.
# Same exact rounded coordinate -> same atomic station.
# =============================================================================

banner(
    "RECONSTRUCTING ATOMIC STATIONS"
)


df[
    "_station_id"
] = (
    df[
        "MonitoringLocationIdentifier"
    ]
    .astype("string")
)


missing_id = (
    df[
        "_station_id"
    ]
    .isna()
)


if missing_id.any():

    df.loc[
        missing_id,
        "_station_id"
    ] = [

        f"MISSING_ID_ROW_{i}"

        for i in df.index[
            missing_id
        ]
    ]


df[
    "_coord_key"
] = (
    df[
        "lat"
    ]
    .round(
        COORD_DECIMALS
    )
    .map(
        lambda x:
            f"{x:.{COORD_DECIMALS}f}"
    )

    +

    "_"

    +

    df[
        "long"
    ]
    .round(
        COORD_DECIMALS
    )
    .map(
        lambda x:
            f"{x:.{COORD_DECIMALS}f}"
    )
)


n = len(
    df
)


parent = np.arange(
    n,
    dtype=int,
)


rank = np.zeros(
    n,
    dtype=int,
)


def find(x):

    while parent[x] != x:

        parent[x] = parent[
            parent[x]
        ]

        x = parent[x]

    return x


def union(a, b):

    ra = find(a)
    rb = find(b)


    if ra == rb:

        return


    if rank[ra] < rank[rb]:

        parent[ra] = rb

    elif rank[ra] > rank[rb]:

        parent[rb] = ra

    else:

        parent[rb] = ra

        rank[ra] += 1


first_id = {}
first_coord = {}


for pos, (
    station_id,
    coord_key,
) in enumerate(
    zip(
        df[
            "_station_id"
        ],
        df[
            "_coord_key"
        ],
    )
):

    station_id = str(
        station_id
    )

    coord_key = str(
        coord_key
    )


    if station_id in first_id:

        union(
            pos,
            first_id[
                station_id
            ],
        )

    else:

        first_id[
            station_id
        ] = pos


    if coord_key in first_coord:

        union(
            pos,
            first_coord[
                coord_key
            ],
        )

    else:

        first_coord[
            coord_key
        ] = pos


roots = np.array(
    [
        find(i)
        for i in range(n)
    ]
)


unique_roots = {

    root:
        number

    for number, root in enumerate(
        np.unique(
            roots
        ),
        start=1,
    )
}


df[
    "station_group"
] = [

    f"STATION_{unique_roots[root]:05d}"

    for root in roots
]


n_stations = (
    df[
        "station_group"
    ]
    .nunique()
)


print(
    f"Atomic station groups: "
    f"{n_stations:,}"
)


if n_stations != 5147:

    raise RuntimeError(
        f"Expected 5,147 stations; found {n_stations:,}."
    )


# =============================================================================
# VERIFY EACH ATOMIC STATION BELONGS TO ONE HUC8
# =============================================================================

station_huc_count = (
    df
    .groupby(
        "station_group"
    )[
        "_HUC8"
    ]
    .nunique()
)


n_crossing = int(
    (
        station_huc_count
        > 1
    )
    .sum()
)


print(
    f"Stations crossing HUC8: "
    f"{n_crossing}"
)


if n_crossing != 0:

    raise RuntimeError(
        "Atomic stations cross HUC8 boundaries."
    )


# =============================================================================
# BUILD STATION TABLE
# =============================================================================

station_table = (

    df
    .groupby(
        "station_group",
        as_index=False,
    )
    .agg(

        HUC8=(
            "_HUC8",
            "first",
        ),

        lat=(
            "lat",
            "median",
        ),

        long=(
            "long",
            "median",
        ),

        n_observations=(
            "chl_a",
            "size",
        ),
    )
)


station_table[
    "HUC8"
] = (
    station_table[
        "HUC8"
    ]
    .astype("string")
    .str.zfill(8)
)


station_table[
    "HUC4"
] = (
    station_table[
        "HUC8"
    ]
    .str[:4]
)


print(
    f"HUC4 strata: "
    f"{station_table['HUC4'].nunique():,}"
)


if (
    station_table[
        "HUC4"
    ]
    .nunique()
    != 147
):

    raise RuntimeError(
        "Expected 147 HUC4 strata."
    )


# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

banner(
    "ENGINEERING FINAL 22 FEATURES"
)


df = add_features(
    df
)


df[
    "HUC4"
] = (
    df[
        "station_group"
    ]
    .map(
        station_table
        .set_index(
            "station_group"
        )[
            "HUC4"
        ]
    )
)


df = (
    df
    .dropna(
        subset=
            FEATURES
            +
            [
                "chl_a",
                "station_group",
                "HUC4",
            ]
    )
    .reset_index(
        drop=True
    )
)


print(
    f"Model-ready observations: "
    f"{len(df):,}"
)


if len(df) != 14125:

    raise RuntimeError(
        "Feature engineering changed the expected sample size."
    )


# =============================================================================
# SAVE CONFIGURATION
# =============================================================================

with open(
    OUT
    / "fixed_xgboost_configuration.json",
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        {
            "model_seed":
                MODEL_SEED,

            "outer_split_seeds":
                SEEDS,

            "n_outer_splits":
                len(SEEDS),

            "test_fraction":
                TEST_FRACTION,

            "target_test_stations":
                1287,

            "features":
                FEATURES,

            "parameters":
                FIXED_PARAMS,
        },
        f,
        indent=2,
    )


# =============================================================================
# RUN 35 FIXED-PARAMETER HUC4 HOLDOUTS
# =============================================================================

banner(
    "RUNNING 35 FIXED-PARAMETER HUC4 STATION HOLDOUTS"
)


all_results = []
all_predictions = []
all_manifests = []


overall_start = time.perf_counter()


for seed in SEEDS:

    print(
        "\n"
        + "-" * 110
    )

    print(
        f"SEED {seed:02d}/35"
    )

    print(
        "-" * 110
    )


    # -------------------------------------------------------------------------
    # Exact HUC4-stratified station split
    # -------------------------------------------------------------------------

    manifest = make_huc4_split(
        station_table=
            station_table,

        seed=
            seed,
    )


    split_lookup = (
        manifest
        .set_index(
            "station_group"
        )[
            "split"
        ]
    )


    work = (
        df.copy()
    )


    work[
        "split"
    ] = (
        work[
            "station_group"
        ]
        .map(
            split_lookup
        )
    )


    train = (
        work[
            work[
                "split"
            ]
            == "train"
        ]
        .copy()
    )


    test = (
        work[
            work[
                "split"
            ]
            == "test"
        ]
        .copy()
    )


    # -------------------------------------------------------------------------
    # Counts
    # -------------------------------------------------------------------------

    train_stations = (
        train[
            "station_group"
        ]
        .nunique()
    )


    test_stations = (
        test[
            "station_group"
        ]
        .nunique()
    )


    assert train_stations == 3860
    assert test_stations == 1287


    # -------------------------------------------------------------------------
    # Leakage checks
    # -------------------------------------------------------------------------

    station_overlap = (
        set(
            train[
                "station_group"
            ]
        )
        &
        set(
            test[
                "station_group"
            ]
        )
    )


    id_overlap = (
        set(
            train[
                "_station_id"
            ]
            .astype(str)
        )
        &
        set(
            test[
                "_station_id"
            ]
            .astype(str)
        )
    )


    coord_overlap = (
        set(
            train[
                "_coord_key"
            ]
            .astype(str)
        )
        &
        set(
            test[
                "_coord_key"
            ]
            .astype(str)
        )
    )


    test_only_huc4 = (
        set(
            test[
                "HUC4"
            ]
        )
        -
        set(
            train[
                "HUC4"
            ]
        )
    )


    if (
        station_overlap
        or
        id_overlap
        or
        coord_overlap
        or
        test_only_huc4
    ):

        raise RuntimeError(
            f"Leakage detected for seed {seed}."
        )


    # -------------------------------------------------------------------------
    # Fit model
    # -------------------------------------------------------------------------

    X_train = (
        train[
            FEATURES
        ]
    )

    y_train = (
        train[
            "chl_a"
        ]
        .astype(float)
    )


    X_test = (
        test[
            FEATURES
        ]
    )

    y_test = (
        test[
            "chl_a"
        ]
        .astype(float)
    )


    model = XGBRegressor(
        **FIXED_PARAMS
    )


    start = time.perf_counter()


    model.fit(
        X_train,
        y_train,
    )


    fit_seconds = (
        time.perf_counter()
        -
        start
    )


    train_pred = (
        model.predict(
            X_train
        )
    )


    test_pred = (
        model.predict(
            X_test
        )
    )


    train_metrics = calculate_metrics(
        y_train,
        train_pred,
    )


    test_metrics = calculate_metrics(
        y_test,
        test_pred,
    )


    response = describe_response(
        y_test
    )


    print(
        f"Train obs/stations : "
        f"{len(train):,} / {train_stations:,}"
    )

    print(
        f"Test obs/stations  : "
        f"{len(test):,} / {test_stations:,}"
    )

    print(
        f"R²                 : "
        f"{test_metrics['r2']:.4f}"
    )

    print(
        f"RMSE               : "
        f"{test_metrics['rmse']:.3f}"
    )

    print(
        f"MAE                : "
        f"{test_metrics['mae']:.3f}"
    )

    print(
        f"Bias               : "
        f"{test_metrics['bias']:+.3f}"
    )

    print(
        f"Slope              : "
        f"{test_metrics['slope']:.4f}"
    )

    print(
        f"Fit time           : "
        f"{fit_seconds:.1f} s"
    )


    # -------------------------------------------------------------------------
    # Save station manifest
    # -------------------------------------------------------------------------

    manifest_save = manifest[
        [
            "station_group",
            "HUC8",
            "HUC4",
            "lat",
            "long",
            "n_observations",
            "split",
        ]
    ].copy()


    manifest_save[
        "seed"
    ] = seed


    manifest_save.to_csv(
        OUT
        / "splits"
        / f"HUC4_seed_{seed:02d}_station_manifest.csv",
        index=False,
    )


    all_manifests.append(
        manifest_save
    )


    # -------------------------------------------------------------------------
    # Save validation predictions
    # -------------------------------------------------------------------------

    pred = test[
        [
            "station_group",
            "MonitoringLocationIdentifier",
            "lat",
            "long",
            "HUC4",
        ]
    ].copy()


    pred[
        "seed"
    ] = seed


    pred[
        "observed_chl_a"
    ] = (
        y_test.to_numpy()
    )


    pred[
        "predicted_chl_a"
    ] = (
        test_pred
    )


    pred[
        "residual"
    ] = (
        test_pred
        -
        y_test.to_numpy()
    )


    pred.to_csv(
        OUT
        / "predictions"
        / f"HUC4_seed_{seed:02d}_validation_predictions.csv",
        index=False,
    )


    all_predictions.append(
        pred
    )


    # -------------------------------------------------------------------------
    # Collect metrics
    # -------------------------------------------------------------------------

    all_results.append({

        "seed":
            seed,

        "train_n":
            len(train),

        "test_n":
            len(test),

        "train_stations":
            train_stations,

        "test_stations":
            test_stations,

        "train_HUC4":
            train[
                "HUC4"
            ]
            .nunique(),

        "test_HUC4":
            test[
                "HUC4"
            ]
            .nunique(),

        "station_overlap":
            len(
                station_overlap
            ),

        "monitoring_id_overlap":
            len(
                id_overlap
            ),

        "coordinate_overlap":
            len(
                coord_overlap
            ),

        "test_only_HUC4":
            len(
                test_only_huc4
            ),

        "train_r2":
            train_metrics[
                "r2"
            ],

        "test_r2":
            test_metrics[
                "r2"
            ],

        "test_rmse":
            test_metrics[
                "rmse"
            ],

        "test_mae":
            test_metrics[
                "mae"
            ],

        "test_bias":
            test_metrics[
                "bias"
            ],

        "test_slope":
            test_metrics[
                "slope"
            ],

        "test_intercept":
            test_metrics[
                "intercept"
            ],

        "test_pearson_r":
            test_metrics[
                "pearson_r"
            ],

        "test_chl_mean":
            response[
                "mean"
            ],

        "test_chl_std":
            response[
                "std"
            ],

        "test_chl_median":
            response[
                "median"
            ],

        "test_chl_p25":
            response[
                "p25"
            ],

        "test_chl_p75":
            response[
                "p75"
            ],

        "test_chl_p95":
            response[
                "p95"
            ],

        "fit_seconds":
            fit_seconds,
    })


# =============================================================================
# SAVE COMBINED RESULTS
# =============================================================================

metrics = (
    pd.DataFrame(
        all_results
    )
    .sort_values(
        "seed"
    )
    .reset_index(
        drop=True
    )
)


metrics.to_csv(
    OUT
    / "HUC4_fixed_35seed_metrics.csv",
    index=False,
)


pd.concat(
    all_predictions,
    ignore_index=True,
).to_csv(
    OUT
    / "HUC4_fixed_35seed_all_predictions.csv",
    index=False,
)


pd.concat(
    all_manifests,
    ignore_index=True,
).to_csv(
    OUT
    / "HUC4_fixed_35seed_all_station_manifests.csv",
    index=False,
)


# =============================================================================
# FINAL 35-SPLIT SUMMARY
# =============================================================================

banner(
    "FINAL 35-SPLIT HUC4 FIXED-PARAMETER VALIDATION"
)


summary_metrics = [

    "test_r2",
    "test_rmse",
    "test_mae",
    "test_bias",
    "test_slope",

]


summary_rows = []


for metric in summary_metrics:

    x = metrics[
        metric
    ].astype(float)


    summary_rows.append({

        "metric":
            metric,

        "mean":
            x.mean(),

        "sd":
            x.std(
                ddof=1
            ),

        "median":
            x.median(),

        "min":
            x.min(),

        "max":
            x.max(),

        "p05":
            x.quantile(
                0.05
            ),

        "p95":
            x.quantile(
                0.95
            ),
    })


summary_df = pd.DataFrame(
    summary_rows
)


summary_df.to_csv(
    OUT
    / "HUC4_fixed_35seed_summary.csv",
    index=False,
)


print(
    summary_df.to_string(
        index=False,
        float_format=
            lambda x:
                f"{x:.4f}",
    )
)


# =============================================================================
# 95% CI FOR MEAN R²
# =============================================================================

r2 = metrics[
    "test_r2"
].astype(float)


r2_mean = (
    r2.mean()
)


r2_sd = (
    r2.std(
        ddof=1
    )
)


r2_sem = (
    r2_sd
    /
    np.sqrt(
        len(r2)
    )
)


tcrit = t.ppf(
    0.975,
    df=
        len(r2)
        - 1,
)


r2_ci_low = (
    r2_mean
    -
    tcrit
    * r2_sem
)


r2_ci_high = (
    r2_mean
    +
    tcrit
    * r2_sem
)


# =============================================================================
# TOP SPLITS
# =============================================================================

banner(
    "TOP 10 HUC4 SPLITS"
)


top10 = (
    metrics[
        [
            "seed",
            "test_r2",
            "test_rmse",
            "test_mae",
            "test_bias",
            "test_slope",
            "test_n",
        ]
    ]
    .sort_values(
        "test_r2",
        ascending=False,
    )
    .head(10)
)


print(
    top10.to_string(
        index=False,
        float_format=
            lambda x:
                f"{x:.4f}",
    )
)


# =============================================================================
# THRESHOLD COUNTS
# =============================================================================

banner(
    "R² STABILITY"
)


threshold_lines = []


for threshold in [

    0.50,
    0.55,
    0.60,

]:

    n_above = int(
        (
            metrics[
                "test_r2"
            ]
            >= threshold
        )
        .sum()
    )


    line = (
        f"R² >= {threshold:.2f}: "
        f"{n_above}/35 "
        f"({100*n_above/35:.1f}%)"
    )


    threshold_lines.append(
        line
    )


    print(
        line
    )


# =============================================================================
# MANUSCRIPT SUMMARY
# =============================================================================

rmse_mean = (
    metrics[
        "test_rmse"
    ]
    .mean()
)

rmse_sd = (
    metrics[
        "test_rmse"
    ]
    .std(
        ddof=1
    )
)


mae_mean = (
    metrics[
        "test_mae"
    ]
    .mean()
)

mae_sd = (
    metrics[
        "test_mae"
    ]
    .std(
        ddof=1
    )
)


bias_mean = (
    metrics[
        "test_bias"
    ]
    .mean()
)

bias_sd = (
    metrics[
        "test_bias"
    ]
    .std(
        ddof=1
    )
)


slope_mean = (
    metrics[
        "test_slope"
    ]
    .mean()
)

slope_sd = (
    metrics[
        "test_slope"
    ]
    .std(
        ddof=1
    )
)


summary_text = f"""
FINAL 35-SPLIT HUC4 STATION-LEVEL VALIDATION
============================================================

Outer repetitions = 35
Training stations per split = 3,860
Validation stations per split = 1,287

Fixed XGBoost parameters used for all repetitions.

R²    = {r2_mean:.4f} ± {r2_sd:.4f}
95% CI for mean R² = {r2_ci_low:.4f} to {r2_ci_high:.4f}

RMSE  = {rmse_mean:.3f} ± {rmse_sd:.3f} µg/L
MAE   = {mae_mean:.3f} ± {mae_sd:.3f} µg/L
Bias  = {bias_mean:+.3f} ± {bias_sd:.3f} µg/L
Slope = {slope_mean:.4f} ± {slope_sd:.4f}

R² range = {r2.min():.4f} to {r2.max():.4f}
Median R² = {r2.median():.4f}

{chr(10).join(threshold_lines)}
"""


print(
    summary_text
)


with open(
    OUT
    / "HUC4_fixed_35seed_MANUSCRIPT_SUMMARY.txt",
    "w",
    encoding="utf-8",
) as f:

    f.write(
        summary_text
    )

    f.write(
        "\nTOP 10 SPLITS\n"
    )

    f.write(
        "=" * 60
        + "\n"
    )

    f.write(
        top10.to_string(
            index=False
        )
    )

    f.write(
        "\n"
    )


# =============================================================================
# R² STABILITY FIGURE
# =============================================================================

banner(
    "CREATING R² STABILITY FIGURE"
)


fig, ax = plt.subplots(
    figsize=(
        10.5,
        5.8,
    )
)


x = metrics[
    "seed"
].to_numpy()


y = metrics[
    "test_r2"
].to_numpy()


# ±1 SD band
ax.axhspan(
    r2_mean - r2_sd,
    r2_mean + r2_sd,
    alpha=0.15,
    label="Mean ± 1 SD",
)


# Mean line
ax.axhline(
    r2_mean,
    linewidth=1.8,
    linestyle="-",
    label=(
        f"Mean R² = "
        f"{r2_mean:.3f}"
    ),
)


# 0.60 reference
ax.axhline(
    0.60,
    linewidth=1.2,
    linestyle="--",
    label="R² = 0.60",
)


# Per-seed points
ax.plot(
    x,
    y,
    marker="o",
    markersize=5,
    linewidth=1.0,
)


ax.set_xlabel(
    "HUC4 station-holdout repetition"
)


ax.set_ylabel(
    "Validation R²"
)


ax.set_title(
    "Repeated HUC4-Stratified Station-Level Validation"
)


ax.set_xticks(
    np.arange(
        1,
        36,
        2,
    )
)


ax.grid(
    True,
    alpha=0.18,
)


ax.legend(
    frameon=True,
)


metric_box = (

    f"n = 35 repetitions\n"
    f"R² = {r2_mean:.3f} ± {r2_sd:.3f}\n"
    f"Range = {r2.min():.3f}–{r2.max():.3f}\n"
    f"95% CI = {r2_ci_low:.3f}–{r2_ci_high:.3f}"

)


ax.text(

    0.985,
    0.035,

    metric_box,

    transform=
        ax.transAxes,

    horizontalalignment=
        "right",

    verticalalignment=
        "bottom",

    fontsize=
        10,

    bbox=dict(
        boxstyle=
            "round,pad=0.4",

        facecolor=
            "white",

        edgecolor=
            "0.65",

        alpha=
            0.9,
    ),
)


fig.tight_layout()


fig.savefig(
    OUT
    / "plots"
    / "HUC4_fixed_35seed_R2_stability.png",
    dpi=400,
    bbox_inches="tight",
)


fig.savefig(
    OUT
    / "plots"
    / "HUC4_fixed_35seed_R2_stability.pdf",
    bbox_inches="tight",
)


plt.close(
    fig
)


# =============================================================================
# TOTAL RUNTIME
# =============================================================================

elapsed = (
    time.perf_counter()
    -
    overall_start
)


banner(
    "DONE"
)


print(
    f"Total runtime: "
    f"{elapsed / 60:.2f} minutes"
)


print(
    "\nResults:"
)

print(
    OUT
    / "HUC4_fixed_35seed_metrics.csv"
)

print(
    OUT
    / "HUC4_fixed_35seed_summary.csv"
)

print(
    OUT
    / "HUC4_fixed_35seed_MANUSCRIPT_SUMMARY.txt"
)

print(
    OUT
    / "HUC4_fixed_35seed_all_predictions.csv"
)

print(
    OUT
    / "HUC4_fixed_35seed_all_station_manifests.csv"
)

print(
    OUT
    / "plots"
    / "HUC4_fixed_35seed_R2_stability.png"
)

