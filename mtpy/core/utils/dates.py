from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from enum import Enum
import numpy as np
import pandas as pd
from pandas.core.generic import NDFrame
import re
from scipy.signal import find_peaks
import statsmodels.tsa.api as smt

from typing import Any, Literal, Optional
from typing_extensions import Self

from .helpers import (
    array_adjust,
    array_shape,
    is_array,
    is_empty,
    is_frame,
    is_index,
    is_number,
    is_numeric,
    is_pandas,
    is_string,
    to_pandas,
    to_series
)

_freq_alias = {
    'year': ('y', 'a', 'year', 'years'),
    'quarter': ('q', 'qtr', 'quarter', 'quarters'),
    'month': ('m', 'month', 'months'),
    'week': ('w', 'week', 'weeks'),
    'day': ('d', 'day', 'days'),
    'hour': ('h', 'hr', 'hour', 'hours'),
    'minute': ('i', 'min', 'minute', 'minutes'),
    'second': ('s', 'sec', 'second', 'seconds'),
    'millisecond': ('ms', 'msec', 'millisecond', 'milliseconds'),
    'microsecond': ('us', 'usec', 'microsecond', 'microseconds')
}
_freq_alias_map = {
    alias: freq
    for freq, aliases in _freq_alias.items()
    for alias in aliases
}

_pd_freq = {
    'year': 'Y',
    'quarter': 'Q',
    'month': 'M',
    'week': 'W',
    'day': 'D',
    'hour': 'h',
    'minute': 'min',
    'second': 's',
    'millisecond': 'ms',
    'microsecond': 'us'
}
_pd_period = {
    'year': 'YS-JAN',
    'quarter': 'QS',
    'month': 'MS',
    'week': 'W-MON',
    'day': 'D',
    'hour': 'h',
    'minute': 'min',
    'second': 's',
    'millisecond': 'ms',
    'microsecond': 'us'
}

_str_label = {
    'year': '%Y',
    'quarter': '%b-%y',
    'month': '%b-%y',
    'week': '%d-%b',
    'day': '%d-%b',
    'hour': '%d-%b-%y',
    'minute': '%d-%b-%y',
    'second': '%d-%b-%y',
    'millisecond': '%d-%b-%y',
    'microsecond': '%d-%b-%y'
}

_regex_delta = re.compile(
    r"^(?P<sign>[+-])?"
    r"P(?!\b)"
    r"(?P<years>[0-9]+([,.][0-9]+)?Y)?"
    r"(?P<months>[0-9]+([,.][0-9]+)?M)?"
    r"(?P<weeks>[0-9]+([,.][0-9]+)?W)?"
    r"(?P<days>[0-9]+([,.][0-9]+)?D)?"
    r"((?P<separator>T)(?P<hours>[0-9]+([,.][0-9]+)?H)?"
    r"(?P<minutes>[0-9]+([,.][0-9]+)?M)?"
    r"(?P<seconds>[0-9]+([,.][0-9]+)?S)?)?$"
)


class Freq(str, Enum):
    """
    Frequency units for time series data.
    """

    YEAR = 'year'
    QUARTER = 'quarter'
    MONTH = 'month'
    WEEK = 'week'
    DAY = 'day'
    HOUR = 'hour'
    MINUTE = 'minute'
    SECOND = 'second'
    MILLISECOND = 'millisecond'
    MICROSECOND = 'microsecond'

    def __str__(self) -> str:
        """
        Always treat the value as a string.
        """
        return self.value

    @classmethod
    def _missing_(cls, value):
        """
        If the freq string is not found, try to parse it.
        """
        return Freq.parse(value)
    
    @staticmethod
    def parse(freq: str, default: Optional[str] = None) -> str:
        """
        Parse a frequency string into a valid `Freq` value.

        Parameters
        ----------
        freq : str
            The frequency string to parse.
        default : str, optional
            The default frequency to return if the parsing fails.
        
        Returns
        -------
        Freq
            The resulting frequency value.
        """
        if freq is None or not isinstance(freq, str):
            if default in list(Freq):
                return Freq(default)
        
        if freq in list(Freq):
            return Freq(freq)

        # Try to infer the frequency from the Pandas frequency strings
        s = re.search(r'^[+-]?\d*({})[A-Za-z-]*$'.format('|'.join(list(_pd_freq.values()))), freq)
        if s is not None:
            freq = s.group(1)

        # Match the frequency string with the available aliases
        if freq.lower() in _freq_alias_map:
            return Freq(_freq_alias_map[freq.lower()])

        if default in list(Freq):
            return Freq(default)

        raise ValueError('Invalid <freq> [{}]'.format(freq))
    
    @staticmethod
    def infer(x: Any, default: Optional[str] = None) -> str:
        """
        Infer the frequency of a time series data.

        Parameters
        ----------
        x : Any
            The time series data to infer the frequency from.
        default : str, optional
            The default frequency to return if the inference fails.
        
        Returns
        -------
        Freq
            The inferred frequency value.
        """
        # A freq definition (D, M, Y, ...)
        if is_string(x):
            return Freq.parse(x, default=default)

        # Already a datetime index -> infer freq
        if is_timeindex(x):
            if len(x) > 2:  # We need at least 3 dates in order to infer freq
                return Freq.parse(x.inferred_freq, default=default)
            else:
                return Freq.parse(default)

        # An array-like object
        if is_array(x):
            # A Pandas object (DataFrame/Series)
            if is_pandas(x):
                # Index is a datetime index -> infer freq
                if is_timeindex(x.index):
                    if len(x.index) > 2:  # We need at least 3 dates in order to infer freq
                        return Freq.parse(x.index.inferred_freq, default=default)
                    else:
                        return Freq.parse(default)
                else:
                    seq = np.asarray(x.index)
            else:
                seq = np.asarray(x)

            # Try to convert array elements to datetime and drop all invalid types -> infer freq
            # Only if we have a datetime sequence
            dt_seq = pd.to_datetime(seq, errors='coerce').dropna()
            if len(dt_seq) > 2:  # We need at least 3 dates in order to infer freq
                return Freq.parse(pd.DatetimeIndex(dt_seq).inferred_freq, default=default)

        return Freq.parse(default)

    @property
    def pd_freq(self) -> str:
        """
        Get the equivalent Pandas frequency string.
        """
        return _pd_freq[self.value]

    @property
    def pd_period(self) -> str:
        """
        Get the equivalent Pandas period string.
        """
        return _pd_period[self.value]
    
    @property
    def str_label(self) -> str:
        """
        Get the equivalent strftime format string.
        """
        return _str_label[self.value]


class Format(str, Enum):
    """
    Default datetime format types.
    """

    SQL = '%Y-%m-%d %H:%M:%S'
    SQLD = '%Y-%m-%d'
    ZULU = '%Y-%m-%dT%H:%M:%SZ'
    ISO = '%c'

    def __str__(self) -> str:
        """
        Always treat the value as a string.
        """
        return self.value


class Delta(pd.DateOffset):
    """
    Delta object for time series data, representing a time interval.
    """

    @staticmethod
    def parse(
        delta: int | str,
        freq: Optional[str] = None,
        **kwargs
    ) -> Self:
        """
        Parse a delta string or integer into a `Delta` object.

        Parameters
        ----------
        delta : int or str
            The delta string or integer to parse.
        freq : str, optional
            The frequency of the delta.
        
        Returns
        -------
        Delta
            The resulting `Delta` object.
        """
        if is_string(delta):
            m = re.match(_regex_delta, delta)

            if m is None:
                raise ValueError('Invalid <delta> [{}]'.format(delta))

            groups = m.groupdict()

            n = -1 if groups['sign'] == '-' else 1
            kwds = {
                key: int(groups[key][:-1]) if groups.get(key) is not None else 0
                for key in ['years', 'months', 'weeks', 'days', 'hours', 'minutes', 'seconds']
            }

            return Delta(n, **kwds)
        elif is_number(delta):
            freq = Freq.parse(freq)

            if freq == Freq.WEEK:
                delta *= 7
                freq = Freq.DAY
            elif freq == Freq.QUARTER:
                delta *= 3
                freq = Freq.MONTH
            elif freq is None:
                raise ValueError('Invalid <freq> [{}]'.format(freq))
            
            n = -1 if delta < 0 else 1
            kwds = {
                key: abs(delta) if _freq_alias_map.get(key) == freq else 0
                for key in ['years', 'months', 'weeks', 'days', 'hours', 'minutes', 'seconds']
            }

            return Delta(n, **kwds)

        return Delta(**kwargs)

    @property
    def pd_offset(self) -> str:
        """
        Get the equivalent Pandas `DateOffset` object.
        """
        return pd.DateOffset(n=self.n, normalize=self.normalize, **self.kwds)

    @property
    def py_timedelta(self) -> str:
        """
        Get the equivalent Python `datetime.timedelta` object.
        """
        return self._pd_timedelta.to_pytimedelta()
    
    @property
    def kwds(self) -> dict:
        return {
            key: getattr(self, key)
            for key in ['years', 'months', 'weeks', 'days', 'hours', 'minutes', 'seconds']
        }

    @property
    def kwds_abs(self) -> dict:
        return {
            'years': int(self.years),
            'months': int(self.years * 12 + self.months),
            'weeks': int(self.weeks + self.days // 7),
            'days': int(self.days),
            'hours': int(self.days * 24 + self.hours),
            'minutes': int((self.days * 24 + self.hours) * 60 + self.minutes),
            'seconds': int(((self.days * 24 + self.hours) * 60 + self.minutes) * 60 + self.seconds)
        }


def is_datetime(
    x: Any
) -> bool:
    """
    Check if a value is a valid datetime object.

    Parameters
    ----------
    x : Any
        The value to check.
    
    Returns
    -------
    bool
        Whether the value is a valid datetime object.
    """
    if is_empty(x):
        return False

    try:
        pd.Timestamp(str(x))
    except (TypeError, ValueError):
        return False

    return True


def is_timeseries(
    x: Any
) -> bool:
    if not is_array(x) or is_empty(x):
        return False

    if hasattr(x, 'dtype'):
        return isinstance(x.dtype, np.dtype) and np.issubdtype(x.dtype, np.datetime64)

    if hasattr(x, 'dtypes'):
        return all([True if isinstance(dtype, np.dtype) and np.issubdtype(dtype, np.datetime64) else False for dtype in x.dtypes])

    try:
        pd.to_datetime(x, errors='raise')
    except (TypeError, ValueError):
        return False

    return True


def is_timeindex(
    x: Any
) -> bool:
    return is_index(x) and is_timeseries(x)


def date_trunc(
    dt: datetime | str,
    freq: str
) -> datetime:
    """
    Truncate a datetime object to the specified unit.

    Parameters
    ----------
    dt : datetime or str
        The datetime object to truncate.
    freq : str
        The unit to truncate the datetime object to.
        See `Freq` for the available values.
        If `Freq.WEEK` is specified, Monday will be used as first day of the week.
    """
    dt = pd.Timestamp(dt)
    freq = Freq.parse(freq)
    
    if freq == Freq.SECOND:
        return dt.replace(microsecond=0)
    elif freq == Freq.MINUTE:
        return dt.replace(second=0, microsecond=0)
    elif freq == Freq.HOUR:
        return dt.replace(minute=0, second=0, microsecond=0)
    elif freq == Freq.DAY:
        return dt.replace(hour=0, minute=0, second=0, microsecond=0)
    elif freq == Freq.WEEK:
        return (dt - relativedelta(days=dt.weekday())).replace(hour=0, minute=0, second=0, microsecond=0)
    elif freq == Freq.MONTH:
        return dt.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    elif freq == Freq.QUARTER:
        return dt.replace(month=((dt.month - 1) // 3) * 3 + 1, day=1, hour=0, minute=0, second=0, microsecond=0)
    elif freq == Freq.YEAR:
        return dt.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
    else:
        raise ValueError('Invalid <freq> [{}]'.format(freq))


def get_delta(
    dt_from: datetime | str,
    dt_to: Optional[datetime | str] = None,
    freq: str = Freq.DAY
) -> int:
    """
    Calculate the difference between two dates in the specified unit.

    Parameters
    ----------
    dt_from : datetime or str
        The start date.
    dt_to : datetime or str, optional
        The end date.
        If not provided, the current date will be used.
    freq : str, default Freq.DAY
        The unit to calculate the difference in.
        See `Freq` for the available values.
    
    Returns
    -------
    int
        The difference between the two dates in the specified unit.
    """
    dt_from = pd.Timestamp(dt_from)
    dt_to = pd.Timestamp(dt_to or pd.Timestamp.now())
    freq = Freq.parse(freq)

    if pd.isna(dt_from) or pd.isna(dt_to):
        return np.NaN
    
    td = dt_to.to_period(freq.pd_freq) - dt_from.to_period(freq.pd_freq)

    if not hasattr(td, 'n'):
        return np.NaN

    return td.n


def add_delta(
    dt: datetime | str,
    delta: int | str,
    freq: Optional[str] = None
) -> datetime:
    """
    Add a delta interval to a datetime object.

    Parameters
    ----------
    dt : datetime or str
        The datetime object to add the delta to.
    delta : int | str
        The delta to add.
        If an integer is provided, it will be interpreted as the number of units to add.
        If a string is provided, it will be interpreted as the unit of the delta.
    freq : str, optional
        The unit of the delta.
        See `Freq` for the available values.
    
    Returns
    -------
    datetime
        The resulting datetime object.
    """
    if not is_datetime(dt):
        raise ValueError('Invalid <dt> [{}]'.format(dt))
    
    dt = pd.Timestamp(dt)
    delta = Delta.parse(delta, freq)

    return dt + delta


def strfdate(
    dt: datetime | str,
    format: str = Format.SQL,
    freq: Optional[str] = None
) -> str:
    """
    Format a datetime object as a string.

    Parameters
    ----------
    dt : datetime or str
        The datetime object.
    format : str, default Format.SQL
        The output format of the datetime object.
        If a string is provided, it will be interpreted as the format.
        If a `Format` value is provided, it will be used as the format.
        See `Format` for the available values.
    freq : str, optional
        If provided, the datetime object will be truncated to the specified unit.
        See `Freq` for the available values.
    
    Returns
    -------
    str
        The formatted string.
    """
    dt = pd.Timestamp(dt)

    if format in list(Format):
        format = Format(format)

    if freq is not None:
        dt = date_trunc(dt, freq)

    return dt.strftime(format)


def strfdelta(
    dt: datetime | timedelta | str | int,
    dt_to: Optional[datetime | str] = None
) -> str:
    """
    Format the difference between two dates as a string.

    Parameters
    ----------
    dt : datetime | timedelta | str | int
        If a datetime object is provided, the difference between the two dates will be calculated.
        If a timedelta object is provided, it will be used as the difference.
        If a string is provided, it will be converted to detatime and the difference will be calculated.
        If an integer is provided, it will be interpreted as a difference in seconds.
    dt_to : datetime | str, optional
        The end date to calculate the difference from.
        If not provided, the current date will be used.
        If `dt` is timedelta object or an integer difference, this parameter will be ignored.
    
    Returns
    -------
    str
        The formatted string.
    """
    if isinstance(dt, timedelta):
        seconds = int(dt.total_seconds())
    elif isinstance(dt, (int, float)):
        seconds = int(dt)
    else:
        dt_from = pd.Timestamp(dt_from)
        dt_to = pd.Timestamp(dt_to or pd.Timestamp.now())
        seconds = (dt_to - dt_from).seconds

    # Get delta components and left trim in order to remove irrelevant parts
    delta = np.trim_zeros([
        int(seconds // 86400),  # days
        int((seconds % 86400) // 3600),  # hours
        int((seconds % 3600) // 60),  # minutes
        int(seconds % 60 // 1)  # seconds
    ], trim='f')

    if len(delta) == 0:
        return '0s'

    parts = ['{:d}d', '{:d}h', '{:d}m', '{:d}s'][-len(delta):]

    return ' '.join(parts).format(*delta)


def ts_range(
    d: NDFrame,
    dt_min: Optional[datetime | str] = None,
    dt_max: Optional[datetime | str] = None
) -> NDFrame:
    """
    Get a time series range between two dates.

    Parameters
    ----------
    d : NDFrame
        Time series data.
    dt_min : datetime or str, optional
        Minimum date to filter.
    dt_max : datetime or str, optional
        Maximum date to filter.
    
    Returns
    -------
    NDFrame
        Filtered time series data.
    """
    ds = to_pandas(d)

    if dt_min is not None:
        if not isinstance(dt_min, datetime):
            dt_min = pd.to_datetime(dt_min)

        ds = ds[ds.index >= dt_min]

    if dt_max is not None:
        if not isinstance(dt_max, datetime):
            dt_max = pd.to_datetime(dt_max)

        ds = ds[ds.index <= dt_max]

    return ds


def ts_resample(
    d: NDFrame,
    agg: str = 'sum',
    freq: str = 'day'
) -> NDFrame:
    """
    Resample a time series data so that it has a fixed frequency.

    Parameters
    ----------
    d : NDFrame
        Time series data.
    agg : str, default 'sum'
        Aggregation function to apply.
    freq : str, default 'day'
        Frequency to resample the data.
    
    Returns
    -------
    NDFrame
        Resampled time series data.
    """
    ds = to_pandas(d)
    freq = Freq.parse(freq)

    if not isinstance(ds.index, pd.DatetimeIndex):
        ds.index = pd.DatetimeIndex(ds.index, freq=freq.pd_period)

    ds = ds.resample('D').asfreq().rename_axis('date').reset_index()

    ds['date'] = ds['date'].dt.to_period(freq.pd_freq).dt.to_timestamp()

    ds = ds.groupby('date').agg(agg).rename_axis(None).resample(freq.pd_period).asfreq()

    return ds


def ts_diff(
    x: np.ndarray,
    periods: Optional[int] = 1,
    axis: Optional[int] = -1,
    freq: Optional[str] = None
) -> np.ndarray:
    """
    Calculate the n-th discrete difference along the given axis.

    It uses the `numpy.diff` function to calculate the difference,
    but also fills the first `n` periods with `np.NaN` values, in order to keep array shape.

    See `numpy.diff` for more details.

    Parameters
    ----------
    x : array-like
        Input array.
    periods : int, optional
        The number of times values are differenced. If zero, the input is returned as-is.
    axis : int, optional
        The axis along which the difference is taken, default is the last axis.
    freq : str, optional
        Frequency unit in which the difference is calculated.
        If not provided, it will be inferred from the input array.
        If it is not posible to infer a frequency, it will default to `Freq.DAY`.
    
    Returns
    -------
    array-like
        The n-th differences. The shape of the output is preserved by filling
        the first `n` periods with `np.NaN` values.
        The type of the output is the same as the input.
    """
    if not is_timeseries(x):
        raise ValueError(f'Input of type [{type(x)}] is not a valid time series.')

    # Adjust the input array to a 2-D ndarray
    ndim = x.ndim
    _, kvar = array_shape(x)
    ts = array_adjust(x, kvar)

    if freq is not None:
        freq = Freq.parse(freq)
    else:
        # Use the time series to infer its frequency
        freq = Freq.infer(ts[:, 0], default=Freq.DAY)

    # Convert the array to a datetime array of the given frequency
    ts = ts.astype(f'datetime64[{freq.pd_freq}]')

    # Compute the difference along the given axis and set results as a float ndarray
    td = (np.diff(ts, periods, axis=axis) / np.timedelta64(1, freq.pd_freq)).astype(float)

    # Fill the first `n` periods of the given axis with `np.NaN` values
    td = np.concatenate((np.full((periods if axis == 0 else td.shape[0], periods if axis == 1 else td.shape[1]), np.nan), td), axis=axis)

    if ndim == 1:
        # If the input is a 1-D array, return the array as a 1-D array
        td = array_adjust(td, 0)

    if isinstance(x, pd.Series):
        td = pd.Series(td, index=x.index, name=x.name)
    elif isinstance(x, pd.DataFrame):
        td = pd.DataFrame(td, index=x.index, columns=x.columns)

    return td


def ts_to_delta(
    x: np.ndarray,
    dt_from: Optional[datetime | str] = None,
    freq: Optional[str] = None
) -> np.ndarray:
    """
    Calculate the difference between each date and a reference date.

    If no date is provided, it will calculate the difference between elements in the following way:
        - If the input is a 1-D array, difference between each date and the previous one.
        - If the input is a 2-D array, difference between last two time series columns.

    Parameters
    ----------
    x : array-like
        Input array.
    dt_from : datetime or str, optional
        Reference date to calculate the difference from.
        If 'min', the minimum date of the input array will be used.
        If 'max', the maximum date of the input array will be used.
        If a datetime object, it will be used as the reference date.
        If not provided, `ts_diff` will be used to calculate the difference along the last axis.
    freq : str, optional
        Frequency unit in which the difference is calculated.
        If not provided, it will be inferred from the input array.
        If it is not posible to infer a frequency, it will default to `Freq.DAY`.
    
    Returns
    -------
    array-like
        The difference between each date and the reference date.
        The type of the output is the same as the input.
    
    Raises
    ------
    ValueError
        If the input is not a valid time series or the reference date is invalid.
    
    Examples
    --------
    >>> x = np.array(['2021-01-01', '2021-01-02', '2021-01-03', '2021-01-04'], dtype='datetime64[D]')
    >>> ts_delta(x, dt_from='min')
    array([0., 1., 2., 3.])

    >>> x = np.array(['2021-01-01', '2021-01-02', '2021-01-03', '2021-01-04'], dtype='datetime64[D]')
    >>> ts_delta(x)
    array([1., 1., 1., 1.])

    >>> x = np.array([['2021-01-01', '2021-01-02'], ['2021-01-03', '2021-01-04']], dtype='datetime64[D]')
    >>> ts_delta(x, dt_from='max')
    array([[-3., -2.], [-1., 0.]])

    >>> x = np.array([['2021-01-01', '2021-01-02'], ['2021-01-03', '2021-01-04']], dtype='datetime64[D]')
    >>> ts_delta(x)
    array([2., 2.])
    """
    if not is_timeseries(x):
        raise ValueError(f'Input of type [{type(x)}] is not a valid time series.')

    # Adjust the input array to a 2-D ndarray
    ndim = x.ndim
    _, kvar = array_shape(x)
    ts = array_adjust(x, kvar)
    
    if dt_from is None:
        # No reference date provided -> calculate the difference between elements
        axis = 1 if kvar > 1 else 0
        td = ts_diff(ts, freq=freq, axis=axis).astype(float)

        if kvar > 1:
            # If the input is a 2-D array, return the second column as a 1-D array
            td = td[:, 1]
        elif ndim == 1:
            # If the input is a 1-D array, return the array as a 1-D array
            td = td.ravel()
        
        if isinstance(x, pd.Series):
            td = pd.Series(td, index=x.index, name=f'{x.name}__delta')
        elif isinstance(x, pd.DataFrame):
            td = pd.Series(td, index=x.index, name=f'{x.columns[-2]}__{x.columns[-1]}__delta')
        
        return td

    if freq is not None:
        freq = Freq.parse(freq)
    else:
        # Use the time series to infer its frequency
        freq = Freq.infer(ts[:, 0], default=Freq.DAY)

    # Convert the array to a datetime array of the given frequency
    ts = ts.astype(f'datetime64[{freq.pd_freq}]')
    
    if dt_from == 'min':
        dt_from = np.nanmin(ts)
    elif dt_from == 'max':
        dt_from = np.nanmax(ts)
    elif is_datetime(dt_from):
        dt_from = pd.Timestamp(dt_from)
    else:
        raise ValueError(f'Invalid <dt_from> [{dt_from}]')
    
    # Compute the difference between each date and the reference date
    td = ((ts - np.datetime64(dt_from, freq.pd_freq)) / np.timedelta64(1, freq.pd_freq)).astype(float)

    if ndim == 1:
        # If the input is a 1-D array, return the array as a 1-D array
        td = array_adjust(td, 0)

    if isinstance(x, pd.Series):
        td = pd.Series(td, index=x.index, name=f'{x.name}__delta')
    elif isinstance(x, pd.DataFrame):
        td = pd.DataFrame(td, index=x.index, columns=x.columns).add_suffix('__delta')
    
    return td


def ts_from_delta(
    x: np.ndarray,
    dt_from: datetime | str,
    freq: Optional[str] = Freq.DAY
) -> np.ndarray:
    """
    Calculate the resulting dates from a difference between each delta and a reference date.

    Parameters
    ----------
    x : array-like
        Input array.
    dt_from : datetime or str
        Reference date to calculate the difference from.
    freq : str, optional, default Freq.DAY
        Frequency unit in which the difference is calculated.
    
    Returns
    -------
    array-like
        The date from the difference.
    """
    if not is_numeric(x):
        raise ValueError(f'Input of type [{type(x)}] is not a valid numeric array.')

    if not is_datetime(dt_from):
        raise ValueError(f'Invalid <dt_from> [{dt_from}]')

    # Adjust the input array to a 2-D ndarray
    ndim = x.ndim
    _, kvar = array_shape(x)
    td = array_adjust(x, kvar)

    freq = Freq.parse(freq)

    # Convert the array to a timedelta array of the given frequency
    td = td.astype(f'timedelta64[{freq.pd_freq}]')

    # Compute the resulting dates adding deltas to the reference date
    ts = np.datetime64(pd.Timestamp(dt_from), freq.pd_freq) + td

    if ndim == 1:
        # If the input is a 1-D array, return the array as a 1-D array
        ts = array_adjust(ts, 0)

    if isinstance(x, pd.Series):
        ts = pd.Series(ts, index=x.index, name=f'{x.name}__ts')
    elif isinstance(x, pd.DataFrame):
        ts = pd.DataFrame(ts, index=x.index, columns=x.columns).add_suffix('__ts')
    
    return ts


def ts_add_delta(
    x: np.ndarray,
    delta: Optional[int | np.ndarray] = None,
    freq: Optional[str] = None
) -> np.ndarray:
    """
    Add a delta interval to a time series data.

    Parameters
    ----------
    x : array-like
        Input array.
    delta : int or array-like, optional
        Delta interval to add.
    freq : str, optional
        Frequency unit in which the delta is calculated.
        If not provided, it will be inferred from the input array.
        If it is not posible to infer a frequency, it will default to `Freq.DAY`.
    
    Returns
    -------
    array-like
        The time series data with the delta interval added.
        The type of the output is the same as the input.
    """
    if delta is None and is_frame(x):
        delta = x[x.columns[-1]]
        x = x.drop(x.columns[-1], axis=1)
        if x.shape[1] == 1:
            x = to_series(x)

    if not is_timeseries(x):
        raise ValueError(f'Input of type [{type(x)}] is not a valid time series.')

    # Adjust the input array to a 2-D ndarray
    ndim = x.ndim
    _, kvar = array_shape(x)
    ts = array_adjust(x, kvar)

    if freq is not None:
        freq = Freq.parse(freq)
    else:
        # Use the time series to infer its frequency
        freq = Freq.infer(ts[:, 0], default=Freq.DAY)

    ts = ts.astype('datetime64')

    if is_array(delta) and is_numeric(delta):
        # Adjust the delta array to a 2-D Nx1 timedelta ndarray
        td = array_adjust(delta, 1).astype(f'timedelta64')
    elif is_number(delta):
        # Set delta as a timedelta value
        td = np.timedelta64(delta, freq.pd_freq)
    else:
        raise ValueError(f'Invalid delta value of type [{type(delta)}].')
    
    # Adjust special frequencies to their equivalent standards
    if freq == Freq.WEEK:
        # Adjust the delta to weeks
        td = td * 7
        freq = Freq.DAY
    elif freq == Freq.QUARTER:
        # Adjust the delta to months
        td = td * 3
        freq = Freq.MONTH
    
    # To perform the addition, first we need to truncate dates to the given frequency,
    # so we need to calculate the residual difference between each date and its truncated value,
    # this will be added back after the addition in order to keep the original date resolution.
    resid = ts - ts.astype(f'datetime64[{freq.pd_freq}]')
    # Add the delta to the original dates and reset their residual difference
    ts = ts.astype(f'datetime64[{freq.pd_freq}]') + td + resid

    if ndim == 1:
        # If the input is a 1-D array, return the array as a 1-D array
        ts = array_adjust(ts, 0)

    if isinstance(x, pd.Series):
        ts = pd.Series(ts, index=x.index, name=x.name)
    elif isinstance(x, pd.DataFrame):
        ts = pd.DataFrame(ts, index=x.index, columns=x.columns)
    
    return ts


def ts_impute(
    d: NDFrame,
    freq: Optional[str] = None,
    drop_empty: bool = False,
    fill_out: bool = False,
    method: Optional[Literal['ffill', 'bfill', 'spline', 'linear', 'value']] = None,
    fill_value: Optional[float] = None
):
    """
    Impute missing values in a time series data.

    Parameters
    ----------
    d : NDFrame
        Time series data.
    freq : str, optional
        Frequency to resample the data.
    drop_empty : bool, default False
        If True, drop empty columns.
    fill_out : bool, default False
        If True, fill values outside the range.
    method : {'ffill', 'bfill', 'spline', 'linear', 'value'}, optional
        Imputation method to use.

        - ``ffill`` :
            Forward fill.
        - ``bfill`` :
            Backward fill.
        - ``spline`` :
            Spline interpolation.
        - ``linear`` :
            Linear interpolation.
        - ``value`` :
            Fill with a specific value.
    fill_value : float, optional
        Value to fill missing values with.
    
    Returns
    -------
    NDFrame
        Imputed time series data.
    """
    ds = to_pandas(d)

    if freq is not None:
        ds = ds.resample(Freq.parse(freq).pd_period).asfreq()

    if drop_empty:
        ds = ds.where(ds > 0, np.nan)
        if len(ds.shape) > 1:
            ds = ds.dropna(axis=1, how='all')
    limit_area = None if fill_out else 'inside'

    if method == 'ffill':
        ds = ds.fillna(method='ffill')
    elif method == 'bfill':
        ds = ds.fillna(method='bfill')
    elif method == 'spline':
        ds = ds.interpolate(method='spline', order=2, limit_area=limit_area)
    elif method == 'linear':
        ds = ds.interpolate(limit_area=limit_area)
    elif method == 'value' and fill_value is not None:
        ds = ds.fillna(fill_value)

    return ds


def ts_fitreg(
    d: pd.Series,
    ix: Optional[np.ndarray] = None,
    weights: Optional[np.ndarray] = None,
    reg_type: Literal['ls', 'lk'] = 'ls',
    alpha: Optional[float] = None,
    **kwargs
) -> pd.Series:
    """
    Fit a regression model to a time series data.

    Parameters
    ----------
    d : Series
        Time series data.
    ix : array-like, optional
        Points to be predicted.
        If None, it will use the original series index.
    weights : array-like, optional
        Weights to be applied, must have the same length as the series.
    reg_type : {'ls', 'lk'}, default 'ls'
        Regression method to use.

        - ``ls`` :
            Least Squares.
        - ``lk`` :
            Local Kernel.
    alpha : float, optional
        The alpha level for the confidence interval, a value of 0.05 returns a 95% CI.
        If None, it will not return the confidence interval.
    
    Returns
    -------
    Series
        Fitted time series data.
    """
    from .stat import LeastSquaresEstimator, LocalKernelEstimator

    ds = to_pandas(d, pdtype='Series')
    if ix is None:
        ix = ds.index

    if reg_type == 'ls':
        est = LeastSquaresEstimator(
            ds,
            weights=weights,
            **kwargs
        ).fit()
    elif reg_type == 'lk':
        est = LocalKernelEstimator(
            ds,
            weights=weights,
            **kwargs
        ).fit()
    else:
        raise ValueError('Invalid <reg_type> [{}]'.format(reg_type))

    y_est = est.predict(ix, alpha=alpha)

    return y_est


def ts_trend(
    d: pd.Series,
    period: int = 7,
    model: Literal['additive', 'multiplicative'] = 'additive',
    impute_method: Optional[Literal['ffill', 'bfill', 'spline', 'linear', 'value']] = None
) -> pd.Series:
    """
    Decompose time series data in order to remove seasonality and extract its trend component.
    Statsmodels `seasonal_decompose` function is used.

    Parameters
    ----------
    d : Series
        Time series data.
    period : int, default 7
        Period of the series.
    model : {'additive', 'multiplicative'}, default 'additive'
        Type of seasonal component.
    impute_method : {'ffill', 'bfill', 'spline', 'linear', 'value'}, optional
        Imputation method to use. See ``ts_impute`` function for more details.
        If None, it will set missing values as 0.
    
    Returns
    -------
    Series
        Trend component of the time series data.
    """
    ds = to_pandas(d)

    if impute_method is not None:
        ds = ts_impute(ds, method=impute_method)
    else:
        ds = ds.fillna(0)

    # If the series has negative values, add a constant to make it positive
    n_add = ds.min() + 1 if ds.min() <= 0 else 0
    ds = ds + n_add

    try:
        ds = smt.seasonal_decompose(
            ds,
            two_sided=False,
            model=model,
            period=period
        ).trend
    except Exception:
        return pd.Series(index=ds.index)
    
    # Remove the constant previously added
    ds = ds - n_add

    return ds


def ts_momentum(
    d: pd.Series,
    deg: int = 3,
    span: int = 15,
    log: bool = False,
    impute_method: Optional[Literal['ffill', 'bfill', 'spline', 'linear', 'value']] = None
) -> pd.Series:
    """
    Calculate the momentum of a time series data.
    It measures the rate of change of a recurrent Exponential Weighted Moving Average (EWMA).
    If `deg` equals 3, this is called 'Trix' (from 'triple exponential') in technical analysis.

    A rising or falling line is an uptrend or downtrend and Momentum shows the slope of that line,
    so it's positive for a steady uptrend, negative for a downtrend, and a crossing through zero is a trend-change,
    i.e. a peak or trough in the underlying average.

    Parameters
    ----------
    d : Series
        Time series data.
    deg : int, default 3
        Number of times to sequentially apply the EWMA.
    span : int, default 15
        Window span of each EMWA application.
    log : bool, default False
        If True, apply log transformation to the data.
    impute_method : {'ffill', 'bfill', 'spline', 'linear', 'value'}, optional
        Imputation method to use. See ``ts_impute`` function for more details.
        If None, no imputation will be applied.
    
    Returns
    -------
    Series
        Momentum of the time series data.
    """
    ds = to_pandas(d)

    if impute_method is not None:
        ds = ts_impute(ds, method=impute_method)

    # If the series has negative values, add a constant to make it positive
    n_add = ds.min() + 1 if ds.min() <= 0 else 0
    ds = ds + n_add

    if log:
        ds = np.log1p(ds)

    # Exponencial Moving Average (applied <deg> times)
    for _ in np.arange(deg):
        ds = ds.ewm(span=span).mean()

    ds = 100 * ds.pct_change(periods=1)

    return ds


def ts_peaks(
    d: pd.Series,
    dt_min: Optional[datetime | str] = None,
    dt_max: Optional[datetime | str] = None,
    distance: Optional[int] = None
) -> pd.Series:
    """
    Find peaks inside a signal based on peak properties.
    Scipy `find_peaks` function is used.

    Parameters
    ----------
    d : Series
        Time series data.
    dt_min : datetime or str, optional
        Minimum date to filter.
    dt_max : datetime or str, optional
        Maximum date to filter.
    distance : int, optional
        Minimum distance between peaks. See `find_peaks` function for more details.
        If provided, must be greater or equal to 1.
    """
    ds = ts_range(d, dt_min=dt_min, dt_max=dt_max)

    peaks, _ = find_peaks(ds, distance=distance)
    ds = ds.iloc[peaks].sort_values(ascending=False)

    return ds


def ts_scale_to_max(
    d: pd.DataFrame,
    ref_col: Optional[str] = None,
    lag_col: Optional[str] = None,
    dt_min: Optional[datetime | str] = None,
    dt_max: Optional[datetime | str] = None,
    distance: Optional[int] = 30
) -> pd.DataFrame:
    """
    Scale time series data to its maximum value.
    It can also scale the data to the maximum value of a reference column or a lag column.

    Parameters
    ----------
    d : DataFrame
        Time series data.
    ref_col : str, optional
        Reference column to scale the data.
    lag_col : str, optional
        Lag column to align the data.
    dt_min : datetime or str, optional
        Minimum date to filter.
    dt_max : datetime or str, optional
        Maximum date to filter.
    distance : int, optional, default 30
        Minimum distance between peaks. See `ts_peaks` function for more details.
    
    Returns
    -------
    DataFrame
        Scaled time series data.
    """
    df = to_pandas(d, pdtype='DataFrame')

    cols = list(df.columns)
    dt_range = [dt_min, dt_max]
    dt_lag = None

    if ref_col is not None:
        ref_peaks = ts_peaks(df[ref_col], dt_min=dt_range[0], dt_max=dt_range[1], distance=distance)
        if ref_peaks.shape[0] > 0:
            dt_range = [
                ref_peaks.index[0] - pd.DateOffset(distance),
                ref_peaks.index[0] + pd.DateOffset(distance)
            ]

    if lag_col is not None:
        lag_peaks = ts_peaks(df[lag_col], dt_min=dt_range[0], dt_max=dt_range[1], distance=distance)
        if lag_peaks.shape[0] > 0:
            dt_lag = lag_peaks.index[0]

    for col in cols:
        col_peaks = ts_peaks(df[col], dt_min=dt_range[0], dt_max=dt_range[1], distance=distance)
        if col_peaks.shape[0] > 0:
            df[col] = 100 * df[col] / col_peaks.values[0]
            if dt_lag is not None:
                n_lag = (col_peaks.index[0] - dt_lag).days
                if n_lag > 0:
                    df[col] = df[col].shift(-n_lag)

    if len(cols) == 1:
        return df[cols[0]]

    return df


def ts_benchmark(
    d: pd.Series,
    agg: str = 'sum',
    freq: str = 'day'
) -> pd.DataFrame:
    """
    Get a yearly time series benchmark based on the data.
    It takes a 1-D time series data and returns a DataFrame with each year's series as columns.

    Parameters
    ----------
    d : Series
        Time series data.
    agg : str, default 'sum'
        Aggregation function to apply.
    freq : str, default 'day'
        Frequency to resample the data.
    
    Returns
    -------
    DataFrame
        Benchmark time series data.
    """
    freq = Freq.parse(freq)

    if freq == Freq.DAY:
        year_from, year_to = pd.Series(d.index.year).agg(['min', 'max'])
        n_rows = 365
    elif freq == Freq.WEEK:
        year_from, year_to = d.index.isocalendar().year.agg(['min', 'max'])
        n_rows = 52
    elif freq == Freq.MONTH:
        year_from, year_to = pd.Series(d.index.year).agg(['min', 'max'])
        n_rows = 12
    else:
        raise ValueError('Invalid <freq> [{}]'.format(freq))

    columns = np.arange(year_from, year_to + 1).astype(str)
    index = np.arange(n_rows)

    df = pd.DataFrame(columns=columns, index=index)
    for year in columns:
        s = d[d.index.year == int(year)].rename_axis('date').reset_index()

        if freq == Freq.DAY:
            s = s.groupby('date').agg(agg).reset_index()
            s = s[~((s.date.dt.month == 2) & (s.date.dt.day == 29))].groupby('date').agg(agg).reset_index().head(n_rows)
        elif freq == Freq.WEEK:
            s['date'] = s.date - pd.to_timedelta(s.date.dt.dayofweek, unit='d')
            s = s.groupby('date').agg(agg).reset_index().head(n_rows)
        elif freq == Freq.MONTH:
            s['date'] = s.date - pd.to_timedelta(s.date.dt.day - 1, unit='d')
            s = s.groupby('date').agg(agg).reset_index().head(n_rows)

        df[[year]] = s[[s.columns[1]]]

    df = df[df.columns[::-1]]

    if freq == Freq.DAY:
        dt_index = pd.date_range(
            start='{}-01-01'.format(year_to),
            end='{}-12-31'.format(year_to),
            freq='D'
        )
        df.index = dt_index[~((dt_index.month == 2) & (dt_index.day == 29))]
    elif freq == Freq.WEEK:
        df.index = [
            datetime.strptime('{}-W{}-1'.format(year_to, week + 1), '%Y-W%W-%w')
            for week in df.index
        ]
    elif freq == Freq.MONTH:
        df.index = pd.date_range(
            start='{}-01-01'.format(year_to),
            periods=12,
            freq=Freq.MONTH.pd_period
        )

    return df


def apply_mom_func(data, sort, group, column, name='momentum', window=15, log=False):
    g = data.sort_values(sort).groupby(group)
    keys = list(g.groups.keys())

    seq = []

    for key in keys:
        cur_g = g.get_group(key)

        y = cur_g[column].reset_index(drop=True)
        sl = int(len(y))

        momentum = ts_momentum(
            y,
            span=window,
            log=log
        )

        for i in range(sl):
            val = momentum[i] if i in momentum.keys() else 0.
            seq.append([key, i, val])

    df = pd.DataFrame(np.array(seq), columns=[group, sort, name])

    df[sort] = df[sort].astype(float)
    df[name] = df[name].astype(float)

    df = df.set_index([group, sort])

    return df
