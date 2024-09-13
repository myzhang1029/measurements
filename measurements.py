import math

import numpy as np


class Uncertainty:
    """An uncertainty with math ability that gives correct string printout.
    
    Examples
    --------
    >>> u = Uncertainty(1.1243)
    >>> str(u)
    '1.1'
    >>> u = Uncertainty(0.104)
    >>> str(u)
    '0.10'
    >>> u = Uncertainty(9123)
    >>> str(u)
    '9000'
    """

    def __init__(self, uncert):
        self._v = uncert

    def get_significant_digit(self):
        """Get the negative index of MSD for rounding uncertainties.

        Find $n$ such that $10^{-n}$ is the decimal weight of the most
        significant digit of `self`, unless when that digit is one, in which
        case the resulting $n$ shall correspond to the next digit.
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
        return -npow

    def __str__(self):
        npow = self.get_significant_digit()
        uncert = round(self._v, npow)
        if npow >= 0:
            return f"{uncert:.{npow}f}"
        return str(uncert)

    def __repr__(self):
        return f"Uncertainty({self})"

    def add_uncert(self, other, r=0):
        if not isinstance(other, Uncertainty):
            raise TypeError("Can only add two instances of `Uncertainty`")
        m = self._v**2 + other._v**2 + 2 * self._v * other._v * r
        return m ** 0.5

    def __add__(self, other):
        # Assume independence
        return self.add_uncert(other)


class Measurement:
    """Represents a quantity with uncertainty.
    
    Arithmetics on `Measurement` should propagate uncertainties correctly
    assuming independence.

    Examples
    --------
    >>> Measurement(10.12, 1.999) + Measurement(20, 3.1)
    Measurement(30, 4)
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
        return " ± ".join(*self._shared_stringify())

    def __repr__(self):
        return "Measurement({0}, {1})".format(*self._shared_stringify())

    def __add__(self, other):
        if isinstance(other, Measurement):
            # Overloaded Uncertainty addition
            new_uncert = self._uncert + other._uncert
            return Measurement(self._center + other._center, new_uncert)
        # Assume `other` is a pure number
        return Measurement(self._center + other, self._uncert)

    def __sub__(self, other):
        if isinstance(other, Measurement):
            # Overloaded Uncertainty addition
            new_uncert = self._uncert + other._uncert
            return Measurement(self._center - other._center, new_uncert)
        # Assume `other` is a pure number
        return Measurement(self._center - other, self._uncert)


# Run tests when invoked directly
if __name__ == "__main__":
    import doctest
    doctest.testmod()
