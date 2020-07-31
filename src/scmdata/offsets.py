"""
A simplified version of :class:`pandas.DateOffset` which use datetime-like
objects instead of :class:`pandas.Timestamp`.

This differentiation allows for times which exceed the range of :class:`pandas.Timestamp`
(see `here <https://stackoverflow.com/a/37226672>`__) which is particularly important
for longer running models.

..  TODO: use np.timedelta64 instead?

"""
import datetime
from typing import Iterable

import cftime
from xarray.coding import cftime_offsets


def to_offset(rule: str) -> cftime_offsets.BaseCFTimeOffset:
    """
    Return a wrapped :class:`DateOffset` instance for a given rule.

    The :class:`DateOffset` class is manipulated to return datetimes instead of
    :class:`pd.Timestamp`, allowing it to handle times outside panda's limited time
    range of ``1677-09-22 00:12:43.145225`` to ``2262-04-11 23:47:16.854775807``, see `this
    discussion <https://stackoverflow.com/a/37226672>`_.

    Parameters
    ----------
    rule
        The rule to use to generate the offset. For options see `pandas offset aliases
        <http://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#timeseries-offset-aliases>`_.

        Business-related offsets, such as 'B', 'C' or 'RE', are not supported.

    Returns
    -------
    :obj:`DateOffset`
        An object of a custom class which provides support for datetimes outside of Timesteps

    Raises
    ------
    ValueError
        If unsupported offset rule is requested, e.g. all business-related offsets
    """
    return cftime_offsets.to_offset(rule)


def generate_range(
    start: datetime.datetime, end: datetime.datetime, offset: cftime_offsets.BaseCFTimeOffset
) -> Iterable[datetime.datetime]:
    """
    Generate a range of datetime objects between start and end, using offset to
    determine the steps.

    The range will extend both ends of the span to the next valid timestep, see
    examples.

    Parameters
    ----------
    start: datetime.datetime
        Starting datetime from which to generate the range (noting roll backward
        mentioned above and illustrated in the examples).

    end: datetime.datetime
        Last datetime from which to generate the range (noting roll forward mentioned
        above and illustrated in the examples).

    offset: :obj:`pandas.tseries.DateOffset`
        Offset object for determining the timesteps. An offsetter obtained from
        :func:`to_offset` *must* be used.

    Yields
    ------
    :obj:`datetime.datetime`
        Next datetime in the range

    Raises
    ------
    ValueError
        Offset does not result in increasing :class`datetime.datetime`s

    Examples
    --------
    The range is extended at either end to the nearest timestep. In the example below,
    the first timestep is rolled back to 1st Jan 2001 whilst the last is extended to 1st
    Jan 2006.

    >>> import datetime as dt
    >>> from pprint import pprint
    >>> from scmdata.offsets import to_offset, generate_range
    >>> g = generate_range(
    ...     dt.datetime(2001, 4, 1),
    ...     dt.datetime(2005, 6, 3),
    ...     to_offset("AS"),
    ... )

    >>> pprint([d for d in g])
    [datetime.datetime(2001, 1, 1, 0, 0),
     datetime.datetime(2002, 1, 1, 0, 0),
     datetime.datetime(2003, 1, 1, 0, 0),
     datetime.datetime(2004, 1, 1, 0, 0),
     datetime.datetime(2005, 1, 1, 0, 0),
     datetime.datetime(2006, 1, 1, 0, 0)]

    In this example the first timestep is rolled back to 31st Dec 2000 whilst the last
    is extended to 31st Dec 2005.

    >>> g = generate_range(
    ...     dt.datetime(2001, 4, 1),
    ...     dt.datetime(2005, 6, 3),
    ...     to_offset("A"),
    ... )
    >>> pprint([d for d in g])
    [datetime.datetime(2000, 12, 31, 0, 0),
     datetime.datetime(2001, 12, 31, 0, 0),
     datetime.datetime(2002, 12, 31, 0, 0),
     datetime.datetime(2003, 12, 31, 0, 0),
     datetime.datetime(2004, 12, 31, 0, 0),
     datetime.datetime(2005, 12, 31, 0, 0)]

    In this example the first timestep is already on the offset so stays there, the last
    timestep is to 1st Sep 2005.

    >>> g = generate_range(
    ...     dt.datetime(2001, 4, 1),
    ...     dt.datetime(2005, 6, 3),
    ...     to_offset("QS"),
    ... )
    >>> pprint([d for d in g])
    [datetime.datetime(2001, 4, 1, 0, 0),
     datetime.datetime(2001, 7, 1, 0, 0),
     datetime.datetime(2001, 10, 1, 0, 0),
     datetime.datetime(2002, 1, 1, 0, 0),
     datetime.datetime(2002, 4, 1, 0, 0),
     datetime.datetime(2002, 7, 1, 0, 0),
     datetime.datetime(2002, 10, 1, 0, 0),
     datetime.datetime(2003, 1, 1, 0, 0),
     datetime.datetime(2003, 4, 1, 0, 0),
     datetime.datetime(2003, 7, 1, 0, 0),
     datetime.datetime(2003, 10, 1, 0, 0),
     datetime.datetime(2004, 1, 1, 0, 0),
     datetime.datetime(2004, 4, 1, 0, 0),
     datetime.datetime(2004, 7, 1, 0, 0),
     datetime.datetime(2004, 10, 1, 0, 0),
     datetime.datetime(2005, 1, 1, 0, 0),
     datetime.datetime(2005, 4, 1, 0, 0),
     datetime.datetime(2005, 7, 1, 0, 0)]
    """
    start_cf = cftime.datetime(*start.timetuple()[:6])
    end_cf = cftime.datetime(*end.timetuple()[:6])

    return cftime_offsets.cftime_range(start_cf, end_cf, freq=offset)
