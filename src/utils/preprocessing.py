import os

import numpy as np

ID_AND_WEIGHT_COLUMNS = [
    "FMONTH",
    "IDATE",
    "IMONTH",
    "IDAY",
    "IYEAR",
    "DISPCODE",
    "SEQNO",
    "_PSU",
    "QSTVER",
    "QSTLANG",
    "_STSTR",
    "_STRWT",
    "_RAWRAKE",
    "_WT2RAKE",
    "_CLLCPWT",
    "_LLCPWT",
    "_DUALCOR",
]

RAW_DUPLICATE_COLUMNS = [
    "WEIGHT2",
    "HEIGHT3",
    "FRUITJU1",
    "FRUIT1",
    "FVBEANS",
    "FVGREEN",
    "FVORANG",
    "VEGETAB1",
    "ALCDAY5",
    "EXEROFT1",
    "EXEROFT2",
    "EXERHMM1",
    "EXERHMM2",
    "STRENGTH",
    "FLSHTMY2",
    "HIVTSTD3",
]

DEFAULT_DROP_COLUMNS = ID_AND_WEIGHT_COLUMNS + RAW_DUPLICATE_COLUMNS

SEVEN_IS_VALID = {"EMPLOY1", "_RACE"}

NONE_88_COLUMNS = {
    "PHYSHLTH",
    "MENTHLTH",
    "POORHLTH",
    "CHILDREN",
    "DRNK3GE5",
    "DOCTDIAB",
    "CHKHEMO3",
    "FEETCHK",
    "ASERVIST",
    "ASDRVIST",
    "ASRCHKUP",
    "ADPLEASR",
    "ADDOWN",
    "ADSLEEP",
    "ADENERGY",
    "ADEAT1",
    "ADFAIL",
    "ADTHINK",
    "ADMOVE",
    "EXRACT21",
}

EXTRA_MISSING_CODES = {
    "_AGEG5YR": [14],
    "_AGE65YR": [3],
    "DROCDY3_": [900],
    "_DRNKWEK": [99900],
    "MAXVO2_": [999],
    "FC60_": [999],
    "DIABAGE2": [98],
    "SCNTLWK1": [97, 98],
    "SCNTWRK1": [97, 98],
}

FORCE_CATEGORICAL = {"_STATE"}

# Key cardiovascular risk factors whose pairwise products are added when
# interactions=True. Value None = numeric (ordinal) column, an int v = the
# indicator (column == v) for non-ordinal codes.
INTERACTION_VARIABLES = {
    "_AGE80": None,     # age
    "SEX": None,        # 1 male, 2 female
    "GENHLTH": None,    # 1 excellent ... 5 poor
    "_RFHYPE5": None,   # 2 = high blood pressure
    "_RFCHOL": None,    # 2 = high cholesterol
    "DIABETE3": 1,      # diabetes
    "CVDSTRK3": 1,      # stroke
    "CHCKIDNY": 1,      # kidney disease
    "_SMOKER3": None,   # 1 daily smoker ... 4 never smoked
    "DIFFWALK": 1,      # difficulty walking
    "_BMI5": None,      # body mass index
}


def load_feature_names(data_path):
    """Feature names of x_train.csv (without the Id column), read-only."""
    with open(os.path.join(data_path, "x_train.csv")) as f:
        header = f.readline().strip().split(",")
    return header[1:]


def special_code_rules(x_train, names):
    """Find, for each column, which BRFSS special codes mean "missing"
    ("don't know", "refused", "not sure") and which mean "none" (-> 0),
    based on the coding width of the column in the training set.

    Args:
        x_train: ndarray of shape (N, D), raw training features
        names:   list of the D feature names

    Returns:
        dict name -> (missing_codes, zero_codes)
    """
    rules = {}
    for j, name in enumerate(names):
        col = x_train[:, j]
        valid = col[~np.isnan(col)]
        missing, zero = list(EXTRA_MISSING_CODES.get(name, [])), []
        # only integer-valued raw answers use the generic codes
        if valid.size > 0 and np.all(valid == np.round(valid)):
            vmax = valid.max()
            if vmax == 9:
                # 1-digit answers: 7 = don't know, 9 = refused / missing
                missing += [9] if name in SEVEN_IS_VALID else [7, 9]
            elif vmax in (77, 98, 99):
                # 2-digit answers: 77 = don't know, 99 = refused, 88 = none
                missing += [77, 99]
                if name in NONE_88_COLUMNS:
                    zero.append(88)
            elif vmax == 999 and valid.min() >= 100:
                # 3-digit frequency answers: 777 / 999 = missing, 888 / 555 = none
                missing += [777, 999]
                zero += [888, 555]
            elif vmax == 9999:
                missing += [7777, 9999]
        if missing or zero:
            rules[name] = (missing, zero)
    return rules


def clean_special_codes(x, names, rules):
    """Apply the rules of special_code_rules() to x.

    Args:
        x:     ndarray of shape (N, D), raw features
        names: list of the D feature names
        rules: dict returned by special_code_rules()

    Returns:
        ndarray of shape (N, D), cleaned copy of x.
    """
    x = x.copy()
    for j, name in enumerate(names):
        if name not in rules:
            continue
        missing, zero = rules[name]
        col = x[:, j]
        col[np.isin(col, missing)] = np.nan
        col[np.isin(col, zero)] = 0
    return x


def fit_preprocessing(
    x_train,
    names,
    drop_columns=DEFAULT_DROP_COLUMNS,
    max_nan_ratio=0.8,
    max_categories=15,
    min_freq=0.01,
    clip_quantile=0.01,
    degree=2,
    interactions=False,
):
    """Learn every preprocessing statistic on the training set.

    Args:
        x_train:        ndarray of shape (N, D), raw training features
        names:          list of the D feature names
        drop_columns:   names of the columns to remove
        max_nan_ratio:  columns with more missing values (after cleaning)
                        are removed
        max_categories: columns with at most this many distinct values are
                        one-hot encoded, the others are continuous
        min_freq:       categories (and missing indicators) seen in less than
                        this fraction of the samples are not encoded
        clip_quantile:  continuous values are clipped to the
                        [q, 1 - q] quantiles
        degree:         1 = linear continuous features, 2 = also add squares
        interactions:   add the pairwise products of INTERACTION_VARIABLES

    Returns:
        dict of parameters used by transform().
    """
    rules = special_code_rules(x_train, names)
    x = clean_special_codes(x_train, names, rules)
    n = x.shape[0]
    drop = set(drop_columns)

    categorical, continuous = [], []
    for j, name in enumerate(names):
        if name in drop:
            continue
        col = x[:, j]
        is_nan = np.isnan(col)
        nan_ratio = np.mean(is_nan)
        if nan_ratio > max_nan_ratio:
            continue
        valid = col[~is_nan]
        values, counts = np.unique(valid, return_counts=True)
        if len(values) <= 1:
            continue
        add_nan = nan_ratio >= min_freq

        if name in FORCE_CATEGORICAL or len(values) <= max_categories:
            frequent = values[counts / n >= min_freq]
            # most frequent category is the reference (absorbed by the bias)
            reference = values[np.argmax(counts)]
            levels = frequent[frequent != reference]
            if len(levels) == 0 and not add_nan:
                continue
            categorical.append((j, name, levels, add_nan))
        else:
            low, high = np.quantile(valid, [clip_quantile, 1 - clip_quantile])
            clipped = np.clip(valid, low, high)
            median = np.median(clipped)
            filled = np.where(is_nan, median, np.clip(col, low, high))
            mean, std = filled.mean(), filled.std()
            if std == 0:
                continue
            continuous.append((j, name, low, high, median, mean, std, add_nan))

    # standardized risk factors (median-imputed) for the pairwise products
    risk_factors = []
    if interactions:
        for name, value in INTERACTION_VARIABLES.items():
            j = names.index(name)
            col = x[:, j]
            if value is not None:
                col = np.where(np.isnan(col), np.nan, (col == value).astype(float))
            median = np.nanmedian(col)
            filled = np.where(np.isnan(col), median, col)
            mean, std = filled.mean(), filled.std()
            if std > 0:
                risk_factors.append((j, name, value, median, mean, std))

    params = {
        "rules": rules,
        "categorical": categorical,
        "continuous": continuous,
        "degree": degree,
        "risk_factors": risk_factors,
    }
    params["feature_names"] = _feature_names(params)

    # standardization statistics of the final (non-bias) features,
    # computed without extra (N, F) temporaries
    tx = _build_features(x, params)
    mean = tx[:, 1:].mean(axis=0)
    var = np.einsum("ij,ij->j", tx[:, 1:], tx[:, 1:]) / n - mean**2
    std = np.sqrt(np.maximum(var, 0))
    std[std < 1e-12] = 1
    params["mean"], params["std"] = mean, std
    return params


def transform(x, names, params):
    """Apply the fitted preprocessing to raw features.

    Args:
        x:      ndarray of shape (N, D), raw features (train or test)
        names:  list of the D feature names
        params: dict returned by fit_preprocessing()

    Returns:
        tx: ndarray of shape (N, 1 + F), with a leading bias column.
    """
    x = clean_special_codes(x, names, params["rules"])
    tx = _build_features(x, params)
    # in-place standardization (the arrays can be > 1 GB)
    tx[:, 1:] -= params["mean"]
    tx[:, 1:] /= params["std"]
    return tx


def _build_features(x, params):
    """Bias column + encoded (not yet standardized) features, written into
    a single preallocated array."""
    tx = np.empty((x.shape[0], 1 + len(params["feature_names"])))
    tx[:, 0] = 1
    k = 1
    for j, _, levels, add_nan in params["categorical"]:
        col = x[:, j]
        for v in levels:
            tx[:, k] = col == v
            k += 1
        if add_nan:
            tx[:, k] = np.isnan(col)
            k += 1

    for j, _, low, high, median, mean, std, add_nan in params["continuous"]:
        col = x[:, j]
        is_nan = np.isnan(col)
        z = (np.where(is_nan, median, np.clip(col, low, high)) - mean) / std
        tx[:, k] = z
        k += 1
        if params["degree"] >= 2:
            tx[:, k] = z**2
            k += 1
        if add_nan:
            tx[:, k] = is_nan
            k += 1

    z = []
    for j, _, value, median, mean, std in params["risk_factors"]:
        col = x[:, j]
        if value is not None:
            col = np.where(np.isnan(col), np.nan, (col == value).astype(float))
        z.append((np.where(np.isnan(col), median, col) - mean) / std)
    for a in range(len(z)):
        for b in range(a + 1, len(z)):
            tx[:, k] = z[a] * z[b]
            k += 1

    return tx


def _feature_names(params):
    """Names of the columns produced by _build_features(), for inspection."""
    out = []
    for _, name, levels, add_nan in params["categorical"]:
        out += [f"{name}={v:g}" for v in levels]
        if add_nan:
            out.append(f"{name}=nan")
    for _, name, *_rest, add_nan in params["continuous"]:
        out.append(name)
        if params["degree"] >= 2:
            out.append(f"{name}^2")
        if add_nan:
            out.append(f"{name}=nan")
    risk = [name for _, name, *_rest in params["risk_factors"]]
    for a in range(len(risk)):
        for b in range(a + 1, len(risk)):
            out.append(f"{risk[a]}*{risk[b]}")
    return out
