# functions for single operations

def round_uncert(r):
    """Rounding uncertainties.

    Round a number to the most significant digit, except for when that digit
    is one, keep one more digit.

    Returns (result, corresponding second parameter to `round`).
    """
    if r == 0:
        return r, 0
    absr = abs(r)
    # Rounding away this power of 10 (MSE exponent - 1)
    npow = int(math.floor(math.log10(absr)))
    # Find the most significant digit
    msd = int(absr/10**npow)
    if msd == 1:
        # Keep the next (lower) power of ten
        npow -= 1
    return round(r, -npow), -npow


def round_uncert_np(uncerts, avgs):
    r_uncerts = np.zeros_like(uncerts)
    r_avgs = np.zeros_like(avgs)
    for idx, uncert in enumerate(uncerts):
        r_uncerts[idx], npow = round_uncert(uncert)
        r_avgs[idx] = round(avgs[idx], npow)
    return r_uncerts, r_avgs


def round_uncert_str(r):
    r, npow = round_uncert(r)
    if npow > 0:
        return f"{r:.{npow}f}"
    return str(r)


def round_uncert_arr_str(uncerts, avgs):
    r_uncerts = [0] * len(uncerts)
    r_avgs = [0] * len(avgs)
    for idx, uncert in enumerate(uncerts):
        r, npow = round_uncert(uncert)
        if npow > 0:
            r_uncerts[idx] = f"{r:.{npow}f}"
            r_avgs[idx] = f"{avgs[idx]:.{npow}f}"
        else:
            r_uncerts[idx] = str(r)
            r_avgs[idx] = str(avgs[idx])
    return r_uncerts, r_avgs
