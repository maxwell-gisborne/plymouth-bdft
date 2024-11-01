import matplotlib.pyplot as plt


def plot_points(ax, points, aspect=1, **kwargs):
    if ax is None:
        _, ax = plt.subplots()
    import numpy as np
    points = np.array(points)
    ax.scatter(*points.transpose(), **kwargs)
    if aspect is not None:
        ax.set_aspect(aspect)
    return ax


def vector_field(ax, Points, Vectors, **args):
    args = dict(angles='xy', scale_units='xy', scale=1) | args
    X, Y = Points.transpose()
    U, V = Vectors.transpose()
    ax.quiver(X, Y, U, V, **args)
    ax.set_aspect(1)
    return ax


def vec_grid(ax, a1, a2, N, Ovec, M=0):
    for i in range(M, N+1):
        ax.axline(Ovec + i*a2, i*a2 + a1, color='#d3d3d3')
        ax.axline(Ovec + i*a1, i*a1 + a2, color='#d3d3d3')
    for i in range(M, N):
        for j in range(M, N):
            ax.text(*(i*a1 + j*a2 + Ovec + 0.5*(a1+a2)), f"$({i},{j})$")
    return ax


def sci_form(number: float, sf=3, ld=1, one=False, zeros=False, sign=False):
    '''format number in scientific format, with sf significant figures
    and ld leading digits before the point.
    if one=True, then 10^0 is printed for digits between 0 and 9.
    if zeros=True, then trailing zeros are added so that there are sf digits.
    if sign=True, then a leading + will be printed for positive numbers.'''
    from math import log10
    s = int(number/abs(number)) == -1
    lo = int(log10(abs(number)) // 1) + 1
    exp = lo - ld
    digits = str(round(number / 10**lo, sf))[2+s:]
    digits = digits.ljust(max(ld, sf), '0')
    mant = digits[:ld], digits[ld:sf]
    if not zeros:
        mant = mant[0], ('!' + mant[1]).strip('0')[1:]
        
    if all(d == '0' for d in digits):
        return '0'
        
    return ( [('+' if sign else ''),'-'][s]
            + mant[0]
            + ('' if len(mant[1]) == 0
                       else '.' + mant[1])
            + ('' if (exp == 0 and not one) else 
               ('\\times 10^{' +str(exp) +'}')))
