"""Uncertainty for measurements."""

import warnings

import numpy as np

from .common import _get_significant_digit_one, _round_arr_or_scalar


class Uncertainty:
    """An uncertainty that gives correct string printout.

    Supports addition with other uncertainties with a given correlation
    coefficient and the full floating point precision is kept until we
    convert this object to a string.

    If the content is an array, this type will internally represent the
    data as a NumPy array.

    Examples
    --------
    `str` keeps only one significant digit:

    >>> u = Uncertainty(9123)
    >>> str(u)
    '9000'

    But if the leading digit is 1, `str` keeps two siginificant digits:

    >>> u = Uncertainty(1.1243)
    >>> str(u)
    '1.1'
    >>> u = Uncertainty(0.104)
    >>> str(u)
    '0.10'

    Edge case behaviour:
    >>> u = Uncertainty(0.198)
    >>> str(u)
    '0.2'
    >>> u = Uncertainty(1.96)
    >>> str(u)
    '2'

    Adding `Uncertainty` is done in quadrature by default:

    >>> Uncertainty(1.14923) + Uncertainty(0.84213)
    Uncertainty(1.4, full=1.4247499885243025)

    Specify a custom correlation coefficient with `add_uncert`:

    >>> Uncertainty(1.14923).add_uncert(Uncertainty(0.84213), r=1)
    Uncertainty(2, full=1.99136)

    `Uncertainty` can be multiplied or divided with/by a scalar:

    >>> 2 * Uncertainty(13)
    Uncertainty(30, full=26)
    >>> Uncertainty(36) / 7
    Uncertainty(5, full=5.142857142857143)

    One can convert between array-type `Uncertainty` and a list of
    `Uncertainty`:

    >>> Uncertainty([1, 2, 15, 23]).as_simple_list()
    [Uncertainty(1.0, full=1), Uncertainty(2, full=2), Uncertainty(15, full=15), Uncertainty(20, full=23)]
    >>> Uncertainty.from_simple_list([Uncertainty(1), Uncertainty(2), Uncertainty(15), Uncertainty(23)])
    Uncertainty([1.0, 2, 15, 20], full=[1, 2, 15, 23])

    Array-type `Uncertainty` supports NumPy-like arithmetic directly:

    >>> 3 * Uncertainty([10, 10]) + Uncertainty([10, 10])
    Uncertainty([30, 30], full=[31.622776601683793, 31.622776601683793])

    Arithmetic between scalar and array `Uncertainty` threads like NumPy operations:

    >>> Uncertainty([10, 10]) + Uncertainty(5)
    Uncertainty([11, 11], full=[11.180339887498949, 11.180339887498949])
    """

    def __init__(self, uncert, full=None):
        # These to allow useful `repr` while still upholding the contract
        # of outputting a representation that can be used to recreate the object
        if full is not None:
            uncert = full
        # Fix negative inputs
        self.u = abs(np.asarray(uncert))

    def get_significant_digit(self):
        """Get the negative index of MSD for rounding uncertainties.

        Find $n$ such that $10^{-n}$ is the decimal weight of the most
        significant digit of `self`, unless when that digit is one, in which
        case the resulting $n$ shall correspond to the next digit.

        Returns
        -------
        n : int
            The index as described above, useful for passing into `round`.
        """
        return _get_significant_digit_one(self.u)

    def get_value(self):
        """Get the underlying uncertainty value."""
        return self.u

    def get_rounded_value(self):
        """Get the underlying uncertainty value after rounding."""
        npow = self.get_significant_digit()
        return _round_arr_or_scalar(self.u, npow)

    def is_array_type(self):
        """Check if this `Uncertainty` is an array or a scalar."""
        return len(self.u.shape) != 0

    def as_simple_list(self):
        """Convert an array `Uncertainty` to a scalar `Uncertainty` list."""
        if not self.is_array_type():
            return self
        return list(iter(self))

    @classmethod
    def from_simple_list(cls, items):
        """Create an array `Uncertainty` from a scalar `Uncertainty` list."""
        return cls([x.u for x in items])

    def __iter__(self):
        return map(Uncertainty, self.u)

    def __getitem__(self, idx):
        return Uncertainty(self.u[idx])

    def __setitem__(self, idx, value):
        self.u[idx] = value

    def __delitem__(self, idx):
        self.u = np.delete(self.u, idx)

    def __str__(self):
        def str_one(u, npow):
            uncert = round(u, npow)
            if npow >= 0:
                return f"{uncert:.{npow}f}"
            # npow negative => keep only int part
            return str(int(uncert))
        npow = self.get_significant_digit()
        if self.is_array_type():
            return "[" + ", ".join(str_one(u, n) for u, n in zip(self.u, npow)) + "]"
        return str_one(self.u, npow)

    def __repr__(self):
        if self.is_array_type():
            # I don't like NumPy's default repr
            return f"Uncertainty({self}, full={list(self.u)})"
        return f"Uncertainty({self}, full={self.u})"

    def add_uncert(self, other, r=0.0):
        """Add two uncertainties assuming a given correlation coefficient.

        Parameters
        ----------
        other : int or float or ndarray
            The other uncertainty to add.
        r : int or float or ndarray, optional
            The correlation coefficient between the two measurements. The
            default is 0 (no correlation).

        Returns
        -------
        Uncertainty
            Resulting uncertainty.
        """
        if not isinstance(other, Uncertainty):
            raise TypeError("Can only add two instances of `Uncertainty`")
        m = self.u**2 + other.u**2 + 2 * self.u * other.u * r
        return Uncertainty(m ** 0.5)

    def __add__(self, other):
        # Assume independence
        return self.add_uncert(other)

    def __radd__(self, other):
        # Assume independence
        return self.add_uncert(other)

    def __iadd__(self, other):
        # Assume independence
        self.u = self.add_uncert(other).u
        return self

    def __mul__(self, other):
        return Uncertainty(self.u * other)

    def __rmul__(self, other):
        return Uncertainty(other * self.u)

    def __imul__(self, other):
        self.u *= other
        return self

    def __truediv__(self, other):
        return Uncertainty(self.u / other)
        # no r*div

    def __itruediv__(self, other):
        self.u /= other
        return self

    def __floordiv__(self, other):
        warnings.warn("Are you sure you want to floordiv an uncertainty?")
        return Uncertainty(self.u // other)
        # no r*div

    def __ifloordiv__(self, other):
        warnings.warn("Are you sure you want to floordiv an uncertainty?")
        self.u //= other
        return self

    def __int__(self):
        return int(self.u)

    def __float__(self):
        return float(self.u)

    def __len__(self):
        return len(self.u)

    def _comparison_method(self, other, operation):
        """Shared code for `__lt__`, `__le__`, etc."""
        method_name = f"__{operation}__"
        if isinstance(other, Uncertainty):
            return getattr(self.u, method_name)(other.u)
        return getattr(self.u, method_name)(other)

    def __lt__(self, other):
        return self._comparison_method(other, "lt")

    def __le__(self, other):
        return self._comparison_method(other, "le")

    def __eq__(self, other):
        return self._comparison_method(other, "eq")

    def __ne__(self, other):
        return self._comparison_method(other, "ne")

    def __gt__(self, other):
        return self._comparison_method(other, "gt")

    def __ge__(self, other):
        return self._comparison_method(other, "ge")
