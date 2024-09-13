"""Simple arithmetic and displaying of measurements and uncertainties.

Written for the Physics 2CL course at UC San Diego."""

import math
import warnings

class Uncertainty:
    """An uncertainty that gives correct string printout.

    Supports addition with other uncertainties with a given correlation
    coefficient and the full floating point precision is kept until we
    convert this object to a string.

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
    Uncertainty(1.4)

    Specify a custom correlation coefficient with `add_uncert`:

    >>> Uncertainty(1.14923).add_uncert(Uncertainty(0.84213), r=1)
    Uncertainty(2)

    `Uncertainty` can be multiplied or divided with/by a scalar:

    >>> 2 * Uncertainty(13)
    Uncertainty(30)
    >>> Uncertainty(36) / 7
    Uncertainty(5)
    """

    def __init__(self, uncert):
        self._v = uncert

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
        if self._v == 0:
            return self._v, 0
        absv = abs(self._v)
        # Rounding away this power of 10 (MSE exponent - 1)
        npow = int(math.floor(math.log10(absv)))
        # Find the most significant digit
        msd = int(absv/10**npow)
        if msd == 1:
            # Keep the next (lower) power of ten
            npow -= 1
            # Check for edge case:
            # If the next two digits will round up, too bad. Erase them.
            # XXX: is there a better way?
            tryround = abs(round(self._v, -npow))
            trymsd, trynmsd = divmod(int(tryround/10**npow), 10)
            if trymsd == 2 and trynmsd == 0:
                npow += 1
        return -npow

    def __str__(self):
        npow = self.get_significant_digit()
        uncert = round(self._v, npow)
        if npow >= 0:
            return f"{uncert:.{npow}f}"
        # npow negative => keep only int part
        return str(int(uncert))

    def __repr__(self):
        return f"Uncertainty({self})"

    def add_uncert(self, other, r=0.0):
        """Add two uncertainties assuming a given correlation coefficient.

        Parameters
        ----------
        r : int or float, optional
            The correlation coefficient between the two measurements. The
            default is 0 (no correlation).

        Returns
        -------
        Uncertainty
            Resulting uncertainty.
        """
        if not isinstance(other, Uncertainty):
            raise TypeError("Can only add two instances of `Uncertainty`")
        m = self._v**2 + other._v**2 + 2 * self._v * other._v * r
        return Uncertainty(m ** 0.5)

    def __add__(self, other):
        # Assume independence
        return self.add_uncert(other)

    def __radd__(self, other):
        # Assume independence
        return self.add_uncert(other)

    def __iadd__(self, other):
        # Assume independence
        self._v = self.add_uncert(other)._v
        return self

    def __mul__(self, other):
        return Uncertainty(self._v * other)

    def __rmul__(self, other):
        return Uncertainty(other * self._v)

    def __imul__(self, other):
        self._v *= other
        return self

    def __truediv__(self, other):
        return Uncertainty(self._v / other)
        # no r*div

    def __itruediv__(self, other):
        self._v /= other
        return self

    def __floordiv__(self, other):
        warnings.warn("Are you sure you want to floordiv an uncertainty?")
        return Uncertainty(self._v // other)
        # no r*div

    def __ifloordiv__(self, other):
        warnings.warn("Are you sure you want to floordiv an uncertainty?")
        self._v //= other
        return self

    def __int__(self):
        return int(self._v)

    def __float__(self):
        return float(self._v)

    def _comparison_method(self, other, operation):
        """Shared code for `__lt__`, `__le__`, etc."""
        method_name = f"__{operation}__"
        if isinstance(other, Uncertainty):
            return getattr(self._v, method_name)(other._v)
        return getattr(self._v, method_name)(other)

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



class Measurement:
    """Represents a quantity with uncertainty.

    Arithmetics on `Measurement` should propagate uncertainties correctly
    assuming independence. No rounding on intermediate results until the
    final string conversion.

    Examples
    --------
    The uncertainties are propagated assuming independence:

    >>> Measurement(10.12, 1.999) + Measurement(20, 3.1)
    Measurement(30, 4)

    Also useful for formatting a single value with uncertainty:

    >>> import numpy as np
    >>> arr = np.array([1.623, 2.123, 2.623])
    >>> str(Measurement(np.mean(arr), np.std(arr)))
    '2.1 ± 0.4'
    """
    def __init__(self, center, uncert):
        self._center = center
        if isinstance(uncert, Uncertainty):
            # No conversion needed
            self._uncert = uncert
        else:
            self._uncert = Uncertainty(uncert)

    def _shared_stringify(self):
        npow = self._uncert.get_significant_digit()
        center = round(self._center, npow)
        uncertstr = str(self._uncert)
        if npow >= 0:
            centerstr = f"{center:.{npow}f}"
        else:
            centerstr = str(center)
        return centerstr, uncertstr

    def __str__(self):
        return " ± ".join(self._shared_stringify())

    def __repr__(self):
        return "Measurement({0}, {1})".format(*self._shared_stringify())

    def add_with_correlation(self, other, r=0.0):
        """Add two `Measurement`s with the given correlation coefficient."""
        if not isinstance(other, Measurement):
            raise TypeError("Use the addition operator instead")
        new_uncert = self._uncert.add_uncert(other._uncert, r=r)
        return Measurement(self._center + other._center, new_uncert)

    def __add__(self, other):
        if isinstance(other, Measurement):
            return self.add_with_correlation(other)
        # Assume `other` is a pure number
        return Measurement(self._center + other, self._uncert)

    def sub_with_correlation(self, other, r=0.0):
        """Subtract two `Measurement`s with the given correlation coefficient."""
        if not isinstance(other, Measurement):
            raise TypeError("Use the addition operator instead")
        new_uncert = self._uncert.add_uncert(other._uncert, r=r)
        return Measurement(self._center - other._center, new_uncert)

    def __sub__(self, other):
        if isinstance(other, Measurement):
            return self.sub_with_correlation(other)
        # Assume `other` is a pure number
        return Measurement(self._center - other, self._uncert)

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
        return abs(diff._center) / float(diff._uncert)

    def _comparison_method(self, other, operation):
        """Shared code for `__lt__`, `__le__`, etc."""
        method_name = f"__{operation}__"
        if isinstance(other, Measurement):
            warnings.warn("Comparison of measurements compares the center value only."
                          " For statistical comparison, use Measurement.tscore")
            return getattr(self._center, method_name)(other._center)
        return getattr(self._center, method_name)(other)

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


# Run tests when invoked directly
if __name__ == "__main__":
    import doctest
    doctest.testmod()
