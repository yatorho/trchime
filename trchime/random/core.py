"""
Here the initlize randomly for tensor.
"""
from ..core import Tensor
from ..core.gather import Tensorable

# noinspection PyProtectedMember
from ..core.gather import _ensure_tensor
import numpy as np


def random(size=None, requires_grad: bool = False) -> 'Tensor':
    """
    random(size=None, requires_grad=False)

    Return random floats in the half-open interval [0.0, 1.0).

    """
    data = np.random.random(size)
    requires_grad = requires_grad
    depends_on = []

    return Tensor(data,
                  requires_grad,
                  depends_on)


def randn(*dn, **kwargs) -> 'Tensor':
    """
    rand(d0, d1, ..., dn, requires_grad=False)

    Random values in a given shape.

    .. note::
    This is a convenience function for users porting code from Matlab,
    and wraps `random_sample`. That function takes a
    tuple to specify the size of the output, which is consistent with
    other trichime functions like `trchime.zeros` and `tichime.ones`.

    Create an array of the given shape and populate it with
    random samples from a standard normal distribution
    over ``[0, 1)``.

    Parameters
    ----------
    d0, d1, ..., dn : int, optional
    The dimensions of the returned array, must be non-negative.
    If no argument is given a single Python float is returned.

    kwargs: bool, optional
    If inputs: `requires_grad =`:
    If True, returned tensor with need-requires_grad.
    The default value of requires_grad is False.

    Returns
    -------
    out : tensor, shape ``(d0, d1, ..., dn)``
    Random values.

    See Also
    --------
    random

    Examples
    --------
    >>> import trchime as tce
    >>> tce.random.randn(3,2)
    tensor([[ 0.14022471,  0.96360618],  #random
            [ 0.37601032,  0.25528411],  #random
            [ 0.49313049,  0.94909878]]) #random
    """
    data = np.random.randn(*dn)
    requires_grad = kwargs.get('requires_grad', False)
    depends_on = []

    return Tensor(data,
                  requires_grad,
                  depends_on)


def rand(*dn, **kwargs) -> 'Tensor':
    """
    uniform(d0, d1, ..., dn, requires_grad=False)

    Random values in a given shape.

    .. note::
    This is a convenience function for users porting code from Matlab,
    and wraps `random_sample`. That function takes a
    tuple to specify the size of the output, which is consistent with
    other Trchime functions like `trchime.zeros` and `trchime.ones`.

    Create a tensor of the given shape and populate it with
    random samples from a uniform distribution
    over ``[0, 1)``.

    Parameters
    ----------
    d0, d1, ..., dn : int, optional
    The dimensions of the returned array, must be non-negative.
    If no argument is given a single Python float is returned.

    kwargs :
    If inputs is `requires_grad = ` :
    If True, returned tensor with need-requires_grad.
    The default value of requires_grad is False.

    Returns
    -------
    out : tensor, shape ``(d0, d1, ..., dn)``
    Random values.

    See Also
    --------
    random

    Examples
    --------
    >>> import trchime as tce
    >>> tce.random.rand(3,2)
    array([[ 0.14022471,  0.96360618],  #random
           [ 0.37601032,  0.25528411],  #random
           [ 0.49313049,  0.94909878]]) #random
    """
    data = np.random.rand(*dn)
    require_grad = kwargs.get('require_grad', False)
    depends_on = []

    return Tensor(data,
                  require_grad,
                  depends_on)


def uniform(low: Tensorable = 0., high: Tensorable = 1., size=None, requires_grad: bool = False) -> 'Tensor':
    """
    uniform(low=0.0, high=1.0, size=None，reuires_grad=False)

    Draw samples from a uniform distribution.

    Samples are uniformly distributed over the half-open interval
    ``[low, high)`` (includes low, but excludes high).  In other words,
    any value within the given interval is equally likely to be drawn
    by `uniform`.

    .. note::
    New code should use the ``uniform`` method of a ``default_rng()``
    instance instead; please see the :ref:`random-quick-start`.

    Parameters
    ----------
    low : float or tensor_like of floats, optional
    Lower boundary of the output interval.  All values generated will be
    greater than or equal to low.  The default value is 0.
    high : float or array_like of floats
    Upper boundary of the output interval.  All values generated will be
    less than or equal to high.  The default value is 1.0.
    size : int or tuple of ints, optional
    Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
    ``m * n * k`` samples are drawn.  If size is ``None`` (default),
    a single value is returned if ``low`` and ``high`` are both scalars.
    Otherwise, ``np.broadcast(low, high).size`` samples are drawn.
    requires_grad : bool, optional
    If True, returned tensor would be assigned with need-grad.
    Default value is False.

    Returns
    -------
    out : tensor
    Drawn samples from the parameterized uniform distribution.

    Notes
    -----
    The probability density function of the uniform distribution is

    .. math:: p(x) = \frac{1}{b - a}

    anywhere within the interval ``[a, b)``, and zero elsewhere.

    When ``high`` == ``low``, values of ``low`` will be returned.
    If ``high`` < ``low``, the results are officially undefined
    and may eventually raise an error, i.e. do not rely on this
    function to behave when passed arguments satisfying that
    inequality condition. The ``high`` limit may be included in the
    returned array of floats due to floating-point rounding in the
    equation ``low + (high-low) * random_sample()``.

    """
    low = _ensure_tensor(low)
    high = _ensure_tensor(high)

    data = np.random.uniform(low = low.data,
                             high = high.data,
                             size = size)

    requires_grad = requires_grad
    depends_on = []

    return Tensor(data,
                  requires_grad,
                  depends_on)


def normal(loc: Tensorable = 0.0, scale: Tensorable = 1.0, size=None, requires_grad: bool = False) -> 'Tensor':
    """
    normal(loc=0.0, scale=1.0, size=None, requires_grad=False)

    Draw random samples from a normal (Gaussian) distribution.

    The probability density function of the normal distribution, first
    derived by De Moivre and 200 years later by both Gauss and Laplace
    independently [2]_, is often called the bell curve because of
    its characteristic shape (see the example below).

    The normal distributions occurs often in nature.  For example, it
    describes the commonly occurring distribution of samples influenced
    by a large number of tiny, random disturbances, each with its own
    unique distribution [2]_.

    .. note::
    New code should use the ``normal`` method of a ``default_rng()``
    instance instead; please see the :ref:`random-quick-start`.

    Parameters
    ----------
    loc : float or tensor_like of floats
    Mean ("centre") of the distribution.
    scale : float or tensor_like of floats
    Standard deviation (spread or "width") of the distribution. Must be
    non-negative.
    size : int or tuple of ints, optional
    requires_grad : bool, optional
    Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
    ``m * n * k`` samples are drawn.  If size is ``None`` (default),
    a single value is returned if ``loc`` and ``scale`` are both scalars.
    Otherwise, ``np.broadcast(loc, scale).size`` samples are drawn.

    Returns
    -------
    out : tensor
    Drawn samples from the parameterized normal distribution.

    Notes
    -----
    The probability density for the Gaussian distribution is

    .. math:: p(x) = \frac{1}{\sqrt{ 2 \pi \sigma^2 }}
    e^{ - \frac{ (x - \mu)^2 } {2 \sigma^2} },

    where :math:`\mu` is the mean and :math:`\sigma` the standard
    deviation. The square of the standard deviation, :math:`\sigma^2`,
    is called the variance.

    The function has its peak at the mean, and its "spread" increases with
    the standard deviation (the function reaches 0.607 times its maximum at
    :math:`x + \sigma` and :math:`x - \sigma` [2]_).  This implies that
    normal is more likely to return samples lying close to the mean, rather
    than those far away.

    Examples
    --------
    Draw samples from the distribution:

    >>> import trchime as tce
    >>> mu, sigma = 0, 0.1 # mean and standard deviation
    >>> s = tce.random.normal(mu, sigma, 1000)

    Verify the mean and the variance:

    >>> abs(mu - np.mean(s))
    0.0  # may vary

    >>> abs(sigma - np.std(s, ddof=1))
    0.1  # may vary

    Two-by-four array of samples from N(3, 6.25):

    >>> np.random.normal(3, 2.5, size=(2, 4))
    array([[-4.49401501,  4.00950034, -1.81814867,  7.29718677],   # random
    [ 0.39924804,  4.68456316,  4.99394529,  4.84057254]])  # random
    """
    loc = _ensure_tensor(loc)
    scale = _ensure_tensor(scale)

    data = np.random.normal(loc = loc.data,
                            scale = scale.data,
                            size = size)

    requires_grad = requires_grad
    depends_on = []

    return Tensor(data,
                  requires_grad,
                  depends_on)


def standard_normal(size=None, requires_grad: bool = False) -> 'Tensor':
    """
    standard_normal(size=None, requires_grad=False)

    Draw samples from a standard Normal distribution (mean=0, stdev=1).

    .. note::
    New code should use the ``standard_normal`` method of a ``default_rng()``
    instance instead; please see the :ref:`random-quick-start`.

    Parameters
    ----------
    size : int or tuple of ints, optional
    requires_grad : bool, optional
    Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
    ``m * n * k`` samples are drawn.  Default is None, in which case a
    single value is returned.

    Returns
    -------
    out : tensor of float or floats
    A floating-point array of shape ``size`` of drawn samples, or a
    single sample if ``size`` was not specified.

    See Also
    --------
    normal :
    Equivalent function with additional ``loc`` and ``scale`` arguments
    for setting the mean and standard deviation.

    Notes
    -----
    For random samples from :math:`N(\mu, \sigma^2)`, use one of::

    mu + sigma * np.random.standard_normal(size=...)
    np.random.normal(mu, sigma, size=...)

    Examples
    --------
    >>>import trchime as tce
    >>> tce.random.standard_normal()
    2.1923875335537315 #random

    >>> s = np.random.standard_normal(8000)
    >>> s
    array([ 0.6888893 ,  0.78096262, -0.89086505, ...,  0.49876311,  # random
    -0.38672696, -0.4685006 ])                                # random
    >>> s.shape
    (8000,)
    >>> s = np.random.standard_normal(size=(3, 4, 2))
    >>> s.shape
    (3, 4, 2)

    Two-by-four array of samples from :math:`N(3, 6.25)`:

    >>> 3 + 2.5 * np.random.standard_normal(size=(2, 4))
    array([[-4.49401501,  4.00950034, -1.81814867,  7.29718677],   # random
    [ 0.39924804,  4.68456316,  4.99394529,  4.84057254]])  # random

    """
    return normal(size = size, requires_grad = requires_grad)


def random_integers(low: int, high: int = None, size=None, requies_grad: bool = False) -> 'Tensor':
    """
    random_integers(low, high=None, size=None)

    Random integers of type `tce.int` between `low` and `high`, inclusive.

    Return random integers of type `int` from the "discrete uniform"
    distribution in the closed interval [`low`, `high`].  If `high` is
    None (the default), then results are from [1, `low`]. The `tce.int`
    type translates to the C long integer type and its precision
    is platform dependent.

    This function has been deprecated. Use randint instead.

    Parameters
    ----------
    low : int
    Lowest (signed) integer to be drawn from the distribution (unless
    ``high=None``, in which case this parameter is the *highest* such
    integer).
    high : int, optional
    If provided, the largest (signed) integer to be drawn from the
    distribution (see above for behavior if ``high=None``).
    size : int or tuple of ints, optional
    requies_grad : bool, optional
    Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
    ``m * n * k`` samples are drawn.  Default is None, in which case a
    single value is returned.

    Returns
    -------
    out : tensor of int or ints
    `size`-shaped array of random integers from the appropriate
    distribution, or a single such random int if `size` not provided.

    See Also
    --------
    randint : Similar to `random_integers`, only for the half-open
    interval [`low`, `high`), and 0 is the lowest value if `high` is
    omitted.

    Notes
    -----
    To sample from N evenly spaced floating-point numbers between a and b,
    use::

    a + (b - a) * (tce.random.random_integers(N) - 1) / (N - 1.)

    Examples
    --------
    >>> import trchime as tce
    >>> tce.random.random_integers(5)
    4 # random
    >>> type(np.random.random_integers(5))
    <class 'tce.Tensor>
    >>> tce.random.random_integers(5, size=(3,2))
    tensor([[5, 4], # random
           [3, 3],
           [4, 5]])

    Choose five random numbers from the set of five evenly-spaced
    numbers between 0 and 2.5, inclusive (*i.e.*, from the set
    :math:`{0, 5/8, 10/8, 15/8, 20/8}`):

    >>> 2.5 * (tce.random.random_integers(5, size=(5,)) - 1) / 4.
    array([ 0.625,  1.25 ,  0.625,  0.625,  2.5  ]) # random

    Roll two six sided dice 1000 times and sum the results:

    >>> d1 = tce.random.random_integers(1, 6, 1000)
    >>> d2 = tce.random.random_integers(1, 6, 1000)
    >>> dsums = d1 + d2

    Display results as a histogram:

    >>> import matplotlib.pyplot as plt
    >>> count, bins, ignored = plt.hist(dsums, 11, density=True)
    >>> plt.show()
    """
    data = np.random.random_integers(low, high, size)
    requies_grad = requies_grad
    depeds_on = []

    return Tensor(data,
                  requies_grad,
                  depeds_on)


def randint(low: Tensorable,
            high: Tensorable = None,
            size=None,
            dtype=int,
            requires_grad: bool = False) -> 'Tensor':
    """
    randint(low, high=None, size=None, dtype=int)

    Return random integers from `low` (inclusive) to `high` (exclusive).

    Return random integers from the "discrete uniform" distribution of
    the specified dtype in the "half-open" interval [`low`, `high`). If
    `high` is None (the default), then results are from [0, `low`).


    Parameters
    ----------
    low : int or tensor-like of ints
    Lowest (signed) integers to be drawn from the distribution (unless
    ``high=None``, in which case this parameter is one above the
    *highest* such integer).
    high : int or tensor-like of ints, optional
    If provided, one above the largest (signed) integer to be drawn
    from the distribution (see above for behavior if ``high=None``).
    If tensor-like, must contain integer values
    size : int or tuple of ints, optional
    Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
    ``m * n * k`` samples are drawn.  Default is None, in which case a
    single value is returned.
    dtype : dtype, optional
    Desired dtype of the result. Byteorder must be native.
    The default value is int.
    requires_grad : bool, optional
    If Ture, returned tensor would be assigned with need-grad propertery.
    Default vale is False.

    Returns
    -------
    out : tensor of int or ints
    `size`-shaped array of random integers from the appropriate
    distribution, or a single such random int if `size` not provided.

    See Also
    --------
    random_integers : similar to `randint`, only for the closed
    interval [`low`, `high`], and 1 is the lowest value if `high` is
    omitted.


    Examples
    --------
    >>> import trchime as tce
    >>> tce.random.randint(2, size=10)
    tensor([1, 0, 0, 0, 1, 1, 0, 0, 1, 0]) # random
    >>> tce.random.randint(1, size=10)
    tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    Generate a 2 x 4 array of ints between 0 and 4, inclusive:

    >>> tce.random.randint(5, size=(2, 4))
    tensor([[4, 0, 2, 1], # random
           [3, 2, 2, 0]])

    Generate a 1 x 3 array with 3 different upper bounds

    >>> tce.random.randint(1, [3, 5, 10])
    tensor([2, 2, 9]) # random

    Generate a 1 by 3 array with 3 different lower bounds

    >>> tce.random.randint(tce.Tensor([5, 1, 3]), 10)
    tensor([9, 8, 7]) # random

    Generate a 2 by 4 array using broadcasting with dtype of uint8

    >>> tce.random.randint([1, 3, 5, 7], [[10], [20]], dtype=tce.uint8)
    tensor([[ 8,  6,  9,  7], # random
            [ 1, 16,  9, 12]], dtype=uint8)
    """

    low = _ensure_tensor(low)
    if high is not None:
        high = _ensure_tensor(high)

        data = np.random.randint(low = low.data, high = high.data, size = size, dtype = dtype)
    else:
        data = np.random.randint(low = low.data, high = high, size = size, dtype = dtype)
    requires_grad = requires_grad
    depends_on = []

    return Tensor(data,
                  requires_grad,
                  depends_on)
