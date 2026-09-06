"""Feature construction for the Week 11 regressor.

ALLOWLIST, NOT DENYLIST.  The Week 10 dataset carries the label itself
(`d_km`, `log_d_km`), the quantity it was drawn from (`d_drawn_km`), and the
surface coordinates the state was constructed from (`gap`, `alpha`, `theta`,
`amplitude_km`) and recovered at (`alpha_found`, `theta_found`).  Any of those
reaching the model produces a spectacular and meaningless result.  A denylist
that misses one column fails silently; an allowlist that misses one column
fails loudly at `assert_no_leakage`.  Hence: nothing enters the feature matrix
unless it is named here.

THE FEATURES, and why each is in.

  x y z vx vy vz   The state.  This is the actual input a warm-start would
                   have; everything else is context.

  C                Jacobi constant.  CONTINUOUS, not a one-hot over energy_id.
                   A one-hot cannot extrapolate to an unseen energy, and
                   Week 15's transfer study requires exactly that.  See the
                   audit note below on which column this should come from.

  family one-hot   L1 vs L2 is a genuine categorical: the two families are
                   different objects, not two values of an ordered variable.
                   Integer-coding them would assert an ordering that does not
                   exist and let the model interpolate "half of L1".

AUDIT NOTE ON C -- AND A CORRECTION.  `sampling.label_and_pack` writes
`C=np.full(n, C)`: the SHEET's energy broadcast to every row.  An earlier
version of this module treated that as a defect and told the notebook to
recompute each state's own Jacobi constant instead.  That was backwards, and it
manufactured a shortcut.

The offsets were never reprojected onto the energy manifold, so an off-surface
state's own Jacobi constant differs from its sheet's by an amount proportional
to the offset -- which is the regression target.  Measured on the Week 10
dataset: the stored sheet C alone scores R2 = -0.0001, while the RECOMPUTED
per-state Jacobi alone scores R2 = +0.7976.  The clean column was the one being
discarded.

It looks fair to use a state's own energy, since a query state's C is
computable at inference.  It is not, and the reason is coverage: with only
THREE energy sheets, the distance from a state's energy to the nearest sheet is
a near-deterministic function of the offset.  Real QPIT families fill a
continuous energy range, so any state is close in energy to some torus and the
proxy evaporates.  The signal is an artefact of sampling three slices of a
continuum, not a property of the dynamics.

So: C ENTERS AS THE STORED SHEET ENERGY.  `jacobi_constant` below is retained
for the audit -- to detect this condition and to cross-check `cr3bp_3d`'s
convention -- and must not be fed to a model while the dataset's offsets remain
unreprojected.

CONVENTION.  C = 2U - v^2 with U = (x^2+y^2)/2 + (1-mu)/r1 + mu/r2.  The
constant mu(1-mu)/2 is OMITTED, matching this project's convention -- verified
against L2 halo #29, whose stored C = 3.060334 is reproduced as 3.060334478
without the term and 3.072337427 with it.  `tests/test_features.py` pins this.
"""

import numpy as np

# mu and LU_KM come from constants.py, which is the single definition for
# the whole project.  This module keeps its own `jacobi_constant` independent
# of cr3bp_3d on purpose (see the module docstring) -- that independence is
# about the CONVENTION, not about holding a different number.  It used to
# hold a truncated copy (0.012150585609624, 23 ULP low).
from constants import MU_EARTH_MOON, LU_KM   # re-exported: see constants.py

STATE_NAMES = ('x', 'y', 'z', 'vx', 'vy', 'vz')

#: Semantic allowlist.  `build_features` emits exactly these, in this order,
#: with `family` expanded to one column per level -- unless the one-hot is
#: dropped, in which case it stops after 'C'.
FEATURE_ALLOWLIST = ('x', 'y', 'z', 'vx', 'vy', 'vz', 'C', 'family_onehot')

#: The two feature sets this project fits.  Named because Week 15 must not
#: compare across them: changing the feature set changes the model, so no
#: Week 11 or Week 12 R2 is admissible as a Week 15 comparator.
FEATURE_SET_W11 = ('x', 'y', 'z', 'vx', 'vy', 'vz', 'C', 'family_0', 'family_1')
FEATURE_SET_W15 = ('x', 'y', 'z', 'vx', 'vy', 'vz', 'C')

#: Columns that must never appear in a feature matrix.  Checked by PREFIX so
#: that `alpha_found`, `d_km`, `d_drawn_km` are all caught by the stems below.
LEAKY_STEMS = (
    'd_km', 'log_d', 'd_drawn', 'at_floor',      # the label and its origin
    'alpha', 'theta', 'gap', 'amplitude',        # surface coordinates
    'group_id', 'source', 'random_dir',          # generation bookkeeping
    'energy_id',                                 # use continuous C instead
)


def jacobi_constant(states, mu=MU_EARTH_MOON):
    """Jacobi constant of each row of `states` (n, 6), rotating frame, nondim.

    Independent of `cr3bp_3d` on purpose -- see module docstring.
    """
    s = np.atleast_2d(np.asarray(states, dtype=float))
    x, y, z, vx, vy, vz = (s[:, i] for i in range(6))
    r1 = np.sqrt((x + mu) ** 2 + y ** 2 + z ** 2)
    r2 = np.sqrt((x - 1.0 + mu) ** 2 + y ** 2 + z ** 2)
    U = 0.5 * (x * x + y * y) + (1.0 - mu) / r1 + mu / r2
    return 2.0 * U - (vx * vx + vy * vy + vz * vz)


def assert_feature_set(names, expected):
    """Raise unless `names` is exactly `expected`, in order.

    WHY THIS EXISTS.  `include_family` defaults to True, so a Week 15 fit that
    forgets to pass False silently gets nine features where its siblings got
    seven.  Nothing downstream would notice: the fit succeeds, the R2 is
    plausible, and the retention ratio -- which divides one fold's score by
    another's -- quietly compares two different models.  Every Week 15 fit
    calls this immediately after `build_features`, so the mistake fails at the
    call site instead of surviving into a number.
    """
    got, exp = tuple(names), tuple(expected)
    if got != exp:
        missing, extra = set(exp) - set(got), set(got) - set(exp)
        raise ValueError(
            f"feature set mismatch\n  expected: {exp}\n  got:      {got}"
            + (f"\n  missing:  {sorted(missing)}" if missing else "")
            + (f"\n  extra:    {sorted(extra)}" if extra else ""))
    return True


def assert_sheet_energy(jacobi, energy_id):
    """Raise unless the energy column is constant within `energy_id`.

    THE CONDITION, NOT A PROXY FOR IT.  The module docstring's audit turns on
    whether the energy column is the SHEET energy or each state's own Jacobi
    constant.  The per-state column scores R2 = +0.7976 alone, against
    -0.0001 for the sheet column, because the offsets were never reprojected --
    so a state's own energy differs from its sheet's by an amount proportional
    to the regression target.  Feeding it manufactures the shortcut.

    Counting distinct values would be a proxy and would false-fire the moment a
    family fills a continuous energy range.  Constancy within `energy_id` is
    the property itself: the sheet energy is by definition one value per sheet.
    """
    jac = np.asarray(jacobi, dtype=float).reshape(-1)
    eid = np.asarray(energy_id).reshape(-1)
    if len(jac) != len(eid):
        raise ValueError("jacobi and energy_id differ in length")
    bad = [int(e) for e in np.unique(eid) if len(np.unique(jac[eid == e])) != 1]
    if bad:
        raise ValueError(
            f"energy column is not constant within energy_id {bad}; this looks "
            f"like the per-state Jacobi constant, which encodes the target "
            f"through the unreprojected offset. Pass the stored sheet C. "
            f"See the audit note in src/features.py.")
    return True


def assert_no_leakage(names):
    """Raise if any feature name matches a leaky stem.  Call before training."""
    bad = [n for n in names
           if any(n == s or n.startswith(s) for s in LEAKY_STEMS)]
    if bad:
        raise ValueError(
            f"leaked columns in feature matrix: {bad}\n"
            f"these encode the label or the surface coordinates the state was "
            f"constructed from; see src/features.py")
    return True


def build_features(states, jacobi, family_id, family_levels,
                   include_family=True):
    """Assemble the feature matrix.

    Parameters
    ----------
    states : (n, 6) rotating-frame state.
    jacobi : (n,) energy per state.  Pass the column the AUDIT endorsed --
        `C_state` if the dataset's `C` turned out to be a family tag.
    family_id : (n,) integer family index.
    family_levels : sequence of the integer levels present, in a FIXED order.
        Passed explicitly rather than inferred from `family_id` so that the
        one-hot layout does not change when a subset is fed in (e.g. a single
        family during a learning-curve fold), which would silently permute
        columns between fits.  Still required, and still validated, when
        `include_family` is False: the call site should declare the family
        structure it believes it has either way.
    include_family : bool, default True.  False drops the one-hot entirely.

        WHY WEEK 15 DROPS IT.  Hold out L2 and its column is constant-zero
        across every training row, so its first-layer weights receive exactly
        zero gradient and stay at initialisation, while the surviving column is
        constant-one and trains as an extra bias.  Both flip at test time.  The
        prediction shift is a seed-dependent constant with no relation to the
        dynamics, so a family-holdout run with the one-hot in measures whether
        an MLP copes with a dead input column.

        The energy folds drop it too, even though `C` is continuous and has no
        such defect, because a family fold and an energy fold must be the same
        model for their retention ratios to be comparable.

    Returns
    -------
    X : (n, 6 + 1 + len(family_levels)) float array
    names : list[str]
    scale_mask : (n_features,) bool -- True where standardisation applies.
        One-hot columns are excluded: scaling them destroys the 0/1 structure
        and makes an absent level indistinguishable from a present one.
    """
    s = np.asarray(states, dtype=float)
    if s.ndim != 2 or s.shape[1] != 6:
        raise ValueError(f"states must be (n, 6), got {s.shape}")
    jac = np.asarray(jacobi, dtype=float).reshape(-1)
    fid = np.asarray(family_id).reshape(-1)
    if not (len(jac) == len(fid) == s.shape[0]):
        raise ValueError("states, jacobi, family_id must have equal length")

    levels = list(family_levels)
    unseen = set(np.unique(fid)) - set(levels)
    if unseen:
        raise ValueError(f"family_id values absent from family_levels: {unseen}")

    if include_family:
        onehot = np.zeros((s.shape[0], len(levels)), dtype=float)
        for j, lv in enumerate(levels):
            onehot[:, j] = (fid == lv)
        X = np.column_stack([s, jac, onehot])
        names = list(STATE_NAMES) + ['C'] + [f'family_{lv}' for lv in levels]
        scale_mask = np.array([True] * 7 + [False] * len(levels))
    else:
        X = np.column_stack([s, jac])
        names = list(STATE_NAMES) + ['C']
        # No one-hot, so every column is continuous and every column scales.
        scale_mask = np.array([True] * 7)
    assert_no_leakage(names)
    return X, names, scale_mask


class Standardizer:
    """Zero-mean unit-variance scaling, FIT ON TRAIN ROWS ONLY.

    Fitting on the full array before splitting leaks test-set moments into the
    training pipeline.  It is a small leak and it is still a leak; more to the
    point it makes the held-out number not mean what it says.
    """

    def __init__(self, mask=None):
        self.mask = mask
        self.mean_ = None
        self.std_ = None

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        m = np.ones(X.shape[1], dtype=bool) if self.mask is None \
            else np.asarray(self.mask, dtype=bool)
        self.mask = m
        self.mean_ = np.where(m, X.mean(axis=0), 0.0)
        std = X.std(axis=0)
        # a constant column would divide by zero; leave it alone instead
        self.std_ = np.where(m & (std > 0), std, 1.0)
        return self

    def transform(self, X):
        if self.mean_ is None:
            raise RuntimeError("Standardizer.transform before .fit")
        return (np.asarray(X, dtype=float) - self.mean_) / self.std_

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def to_dict(self):
        return dict(mean=self.mean_, std=self.std_, mask=self.mask)
