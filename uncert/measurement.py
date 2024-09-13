import warnings

import numpy as np

from .common import _round_arr_or_scalar
from .uncertainty import Uncertainty


class Measurement:
    """Represents a quantity with uncertainty.

    Arithmetics on `Measurement` should propagate uncertainties correctly
    assuming independence. No rounding on intermediate results until the
    final string conversion.

    Examples
    --------
    Uncertainties are propagated assuming independence:

    >>> Measurement(10.12, 1.999) + Measurement(20, 3.1)
    Measurement(30, 4)

    Also useful for formatting a single value with uncertainty:

    >>> import numpy as np
    >>> arr = np.array([1.623, 2.123, 2.623])
    >>> str(Measurement(np.mean(arr), np.std(arr)))
    '2.1 ± 0.4'

    Basic arithmetic operations work (also assuming independence):

    >>> Measurement(10.12, 1.999) * Measurement(20, 1.1)
    Measurement(200, 40)
    >>> 10 * Measurement(20, 1.1)
    Measurement(200, 11)
    >>> Measurement(10.12, 1.999) / Measurement(20, 1.1)
    Measurement(0.51, 0.10)
    >>> 1 / Measurement(10, 1)
    Measurement(0.100, 0.010)

    There is also array-type `Measurement`:
    >>> mar = Measurement(np.arange(5), np.arange(0.1, 0.3, 0.04))
    >>> str(mar)
    '[0.00 ± 0.10, 1.00 ± 0.14, 2.00 ± 0.18, 3.0 ± 0.2, 4.0 ± 0.3]'

    Array-type `Measurement` supports NumPy-like arithmetic directly:

    >>> 3 * mar
    [Measurement(0.0, 0.3), Measurement(3.0, 0.4), Measurement(6.0, 0.5), Measurement(9.0, 0.7), Measurement(12.0, 0.8)]
    """

    def __init__(self, center, uncert):
        self.center = np.asarray(center)
        if isinstance(uncert, Uncertainty):
            # No conversion needed
            self.uncert = uncert
        elif hasattr(uncert, "__len__"):
            # Make array-type `Uncertainty`
            self.uncert = Uncertainty.from_simple_list(uncert)
        else:
            self.uncert = Uncertainty(uncert)
        if self.uncert.is_array_type() and len(self.center) != len(self.uncert):
            raise ValueError("The lengths of `center` and `uncert` must match")
        # array-type uncert implies array center,
        # but array center does not imply array-type uncert

    def is_array_type(self):
        """Check if this `Uncertainty` is an array or a scalar."""
        return len(self.center.shape) != 0

    def get_center(self):
        """Get the center value of self."""
        return self.center

    def get_uncert(self):
        """Get the uncertainty of self."""
        return self.uncert

    def get_rounded_center(self):
        """Get the rounded center value of self."""
        npow = self.uncert.get_significant_digit()
        return _round_arr_or_scalar(self.center, npow)

    def get_rounded_uncert(self):
        """Get the rounded uncertainty of self."""
        return self.uncert.get_rounded_value()

    @staticmethod
    def _shared_stringify(center, uncert):
        # We do not use `get_rounded_x` here to save one round of computation
        npow = uncert.get_significant_digit()
        center = np.round(center, npow)
        uncertstr = str(uncert)
        if npow >= 0:
            centerstr = f"{center:.{npow}f}"
        else:
            # npow negative => keep only int part
            centerstr = str(int(center))
        return centerstr, uncertstr

    def _shared_stringify_high(self, individual_formatter):
        if self.is_array_type():
            if self.uncert.is_array_type():
                return "[" + ", ".join(
                    individual_formatter.format(*self._shared_stringify(c, u)) for c, u in zip(self.center, self.uncert)
                ) + "]"
            return "[" + ", ".join(
                individual_formatter.format(*self._shared_stringify(c, self.uncert)) for c in self.center
            ) + "]"
        return individual_formatter.format(*self._shared_stringify(self.center, self.uncert))

    def __str__(self):
        return self._shared_stringify_high("{0} ± {1}")

    def __repr__(self):
        return self._shared_stringify_high("Measurement({0}, {1})")

    @staticmethod
    def _check_other_is_us(other):
        if not isinstance(other, Measurement):
            raise TypeError("Use normal Python operators instead")

    def add_with_correlation(self, other, r=0.0):
        """Add two `Measurement`s with the given correlation coefficient."""
        self._check_other_is_us(other)
        new_uncert = self.uncert.add_uncert(other.uncert, r=r)
        return Measurement(self.center + other.center, new_uncert)

    def __add__(self, other):
        if isinstance(other, Measurement):
            return self.add_with_correlation(other)
        # Assume `other` is a pure number
        return Measurement(self.center + other, self.uncert)

    def __radd__(self, other):
        # if isinstance(other, Measurement):
        #     unreachable: Python should call other's __add__
        # Assume `other` is a pure number
        return Measurement(other + self.center, self.uncert)

    def sub_with_correlation(self, other, r=0.0):
        """Subtract two `Measurement`s with the given correlation coefficient."""
        self._check_other_is_us(other)
        new_uncert = self.uncert.add_uncert(other.uncert, r=r)
        return Measurement(self.center - other.center, new_uncert)

    def __sub__(self, other):
        if isinstance(other, Measurement):
            return self.sub_with_correlation(other)
        # Assume `other` is a pure number
        return Measurement(self.center - other, self.uncert)

    def __rsub__(self, other):
        # if isinstance(other, Measurement):
        #     unreachable: Python should call other's __sub__
        # Assume `other` is a pure number
        return Measurement(other - self.center, self.uncert)

    def mul_with_correlation(self, other, r=0.0):
        """Multiply two `Measurement`s with the given correlation coefficient."""
        self._check_other_is_us(other)
        # u(f)**2 = (partial(f,a)u(a))**2+(partial(f,b)u(b))**2+corrterm
        # u(f)**2 = (u(a)b)**2 + (u(b)a)**2+corrterm
        # (u(f)/f)**2 = (u(a)/a)**2 + (u(b)/b)**2+corrterm/(ab)**2
        new_reluncert = (
            self.uncert/self.center).add_uncert(other.uncert/other.center, r=r)
        new_center = self.center * other.center
        return Measurement(new_center, new_reluncert * new_center)

    def __mul__(self, other):
        if isinstance(other, Measurement):
            return self.mul_with_correlation(other)
        # Assume `other` is a pure number
        return Measurement(self.center * other, self.uncert * other)

    def __rmul__(self, other):
        # if isinstance(other, Measurement):
        #     unreachable: Python should call other's __mul__
        # Assume `other` is a pure number
        return Measurement(other * self.center, other * self.uncert)

    def truediv_with_correlation(self, other, r=0.0):
        """Multiply two `Measurement`s with the given correlation coefficient."""
        self._check_other_is_us(other)
        new_center = self.center / other.center
        new_reluncert = (self.uncert/self.center).add_uncert(
            other.uncert / other.center, r=r)
        return Measurement(new_center, new_reluncert * new_center)

    def __truediv__(self, other):
        if isinstance(other, Measurement):
            return self.truediv_with_correlation(other)
        # Assume `other` is a pure number
        return Measurement(self.center / other, self.uncert / other)

    def __rtruediv__(self, other):
        # if isinstance(other, Measurement):
        #     unreachable: Python should call other's __mul__
        # Assume `other` is a pure number
        new_center = other / self.center
        reluncert = self.uncert / self.center
        return Measurement(new_center, reluncert * new_center)

    def __floordiv__(self, other):
        # Does not really make much sense to produce an uncertainty for this
        return self.center // other

    def __rfloordiv__(self, other):
        # Does not really make much sense to produce an uncertainty for this
        return other // self.center

    # I'll leave it for Python to implement the default in-place methods

    def __abs__(self):
        return Measurement(abs(self.center), self.uncert)

    def tscore(self, other, r=0.0):
        """Compute the t-score between two `Measurement`s.

        Examples
        --------
        >>> a = Measurement(10, 1)
        >>> b = Measurement(11, 1)
        >>> a.tscore(b)
        0.7071067811865475

        The other parameter does not have to be a `Measurement` (in which
        case it is assumed to be exact):
        >>> a.tscore(11)
        1.0

        `tscore` also accepts a given correlation coefficient:
        >>> a.tscore(b, r=1)
        0.5
        """
        if isinstance(other, Measurement):
            diff = self.sub_with_correlation(other, r=r)
        else:
            # This might raise TypeError if `other` is not to be added
            diff = self - other
        return abs(diff.center) / float(diff.uncert)

    def _comparison_method(self, other, operation):
        """Shared code for `__lt__`, `__le__`, etc."""
        method_name = f"__{operation}__"
        if isinstance(other, Measurement):
            warnings.warn("Comparison of measurements compares the center value only."
                          " For statistical comparison, use `Measurement.tscore`")
            return getattr(self.center, method_name)(other.center)
        return getattr(self.center, method_name)(other)

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
