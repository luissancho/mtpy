from ast import literal_eval
from datetime import datetime
import json
from itertools import groupby
from operator import itemgetter
import IPython.display as ipd
import math
import numpy as np
import pandas as pd
from pandas.core.generic import NDFrame
import random
import re
import string
from tqdm import tqdm

from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import StandardScaler
from scipy.signal import savgol_filter
from scipy.stats import gaussian_kde
from statsmodels.stats.weightstats import DescrStatsW
import statsmodels.tsa.api as smt

from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer

from typing import Any, Literal, Optional


mtypes_map = {
    'str': ['char', 'string', 'text', 'varchar'],
    'cat': ['category', 'categorical'],
    'obj': ['array', 'json', 'list', 'mixed', 'object'],
    'dtd': ['date'],
    'dts': ['datetime', 'time', 'timestamp'],
    'num': ['decimal', 'double', 'float', 'floating', 'number', 'numeric'],
    'int': ['integer'],
    'bin': ['bool', 'boolean', 'binary']
}
mtypes_base = {
    'str': {'length': 256, 'nullable': True, 'default': None},
    'cat': {'length': 256, 'nullable': True, 'default': None},
    'obj': {'length': 256, 'nullable': True, 'default': None},
    'dtd': {'length': 4, 'nullable': True, 'default': None},
    'dts': {'length': 8, 'nullable': True, 'default': None},
    'num': {'length': [16, 2], 'nullable': True, 'default': None},
    'int': {'length': 4, 'nullable': True, 'default': None},
    'bin': {'length': 1, 'nullable': True, 'default': None}
}


def alarm(d: int = 1) -> None:
    """
    Play an alarm sound.

    Parameters
    ----------
    d : int
        Duration of the alarm in seconds.
    """
    # Set the sampling rate to 4000 Hz.
    s = 4000
    
    # Generate a time vector from 0 to d seconds with a sampling rate of s samples per second
    t = np.linspace(0, d, int(d * s))
    
    # Create a sine wave with a frequency of 440 Hz scaled to half amplitude to reduce volume
    x = 0.5 * np.sin(2 * np.pi * 440 * t)

    # Play the generated sine wave as an audio signal
    ipd.Audio(x, rate=s, autoplay=True)


def get_str_code(size: int = 8) -> str:
    """
    Generates a random string code of a specific size.

    Parameters
    ----------
    size : int
        Size of the generated code.
    
    Returns
    -------
    str
        Random string code.
    """
    chars = string.ascii_lowercase + string.digits

    return ''.join(random.choice(chars) for _ in range(size))


def get_ts_code(unit: Optional[str] = None) -> str:
    """
    Generates a timestamp code.

    Parameters
    ----------
    unit : str, optional
        Unit to use as the base of the timestamp.

        - 'milli' : milliseconds
        - 'micro' : microseconds
        - 'nano' : nanoseconds
    
    Returns
    -------
    str
        Timestamp code.
    """
    units = {
        'milli': 1000,
        'micro': 1000000,
        'nano': 1000000000
    }

    ts = datetime.utcnow().timestamp()

    if unit is not None and unit in units:
        ts *= units[unit]

    return str(round(ts))


def to_roman(num: int) -> str:
    val = [
        1000, 900, 500, 400,
        100, 90, 50, 40,
        10, 9, 5, 4,
        1
    ]
    syb = [
        'M', 'CM', 'D', 'CD',
        'C', 'XC', 'L', 'XL',
        'X', 'IX', 'V', 'IV',
        'I'
    ]

    roman = ''
    i = 0

    while num > 0:
        for _ in range(num // val[i]):
            roman += syb[i]
            num -= val[i]

        i += 1

    return roman


def get_mtype(dtype: str) -> str | None:
    """
    Maps a data type to its corresponding meta type.

    Parameters
    ----------
    dtype : str
        Data type to map.
    
    Returns
    -------
    str
        Standardized meta type name.
    None
        If the data type is not a valid mtype or alias.
    """
    dtype = re.sub(r'[^a-z]', '', dtype.lower())

    if dtype in mtypes_map:
        return dtype

    for type, aliases in mtypes_map.items():
        if dtype in aliases:
            return type


def infer_mtype(x: pd.Series) -> str | None:
    """
    Infers the meta type of a Pandas Series.

    Parameters
    ----------
    x : Series
        Pandas series to infer the meta type from.

    Returns
    -------
    str
        Inferred meta type.
    """
    return get_mtype(
        pd.api.types.infer_dtype(x)
    )


def infer_meta(
    x: NDFrame,
    detailed: bool = False,
    exclude: Optional[list] = None
) -> dict:
    """
    Infers the standardized column descriptions dictionary from data.

    Parameters
    ----------
    x : DataFrame, Series or Index
        Data to infer column types from.
    detailed : bool, optional
        If True, it will return a pair of mtype and max length for each column.
    exclude : list, optional
        Columns to exclude from the inferred meta.

    Returns
    -------
    dict
        Standardized column descriptions.
    """
    exclude = exclude or []
    meta = {}

    for col in x.columns:
        if col in exclude:
            continue
        
        mtype = infer_mtype(x[col])

        if mtype is None:
            mtype = 'str'

        if not detailed and mtype == 'obj':
            mtype = 'str'
        
        if mtype == 'str' and x[col].notnull().any() and x[col].unique().shape[0] < 8:
            mtype = 'cat'
        
        if mtype == 'dts' and x[col].notnull().any() and x[col].dt.time.max() == x[col].dt.time.min():
            mtype = 'dtd'
        
        if detailed and x[col].notnull().any():
            if mtype in ['str', 'obj', 'cat']:
                mlen = int(x[col].str.len().max())
                mtype = [mtype, mlen]
            elif mtype == 'int':
                mlen = int(math.log(x[col].max(), 256)) + 1
                mtype = [mtype, mlen]

        meta[col] = mtype

    return meta


def parse_meta(meta: dict) -> dict:
    """
    Parses a dictionary with column descriptions, in order to standardize its format.

    Parameters
    ----------
    meta : dict or dict-like
        Column descriptions, a dictionary with specific meta information.

        These are the parameters needed to define each of the columns:

        - ``type`` :
            Data type, which consists in any of the folowing:
                - ``str`` : cast value as string (default length 256).
                - ``obj`` : for objects, arrays or any json-type values (default length 256).
                - ``cat`` : categorize value as a 'category' dtype (default length 256).
                - ``dtd`` : cast value as datetime and keep only the date part (default length 4).
                - ``dts`` : cast value as datetime with date and time parts (default length 8).
                - ``num`` : cast value as float (default length [16, 2]).
                - ``int`` : cast value as integer (default length 4).
                - ``bin`` : cast value as binary or boolean-type (default length 1).
        - ``length`` :
            Maximum length of the value (in bytes), needed for database table definition.
        - ``nullable`` :
            If True, this column will accept null values, needed for database table definition (default True).
        - ``default`` :
            Default value of column, in order to fill missing values if needed (default None).

        If ``meta`` is not provided, it will try to infer column types from the data itself.

        Examples:

        {
            'col1': {'type': str', 'length': 256, 'nullable': True, 'default': None},  # As a dict
            'col2': ['str', 256, True, None],  # As a list of parameters
            'col3': ['int', 4, True, 'mean'],  # Use column 'mean' function as default value
            'col4': ['num', [16, 2]]  # Numeric type needs definition of integer and decimal parts
            'col5': 'dtd'  # Just the column type (keeping parameters defaults),
            'col6': 'varchar'  # Alias may be used instead of the original type name
        }
    
    Returns
    -------
    dict
        Standardized column descriptions.
    """
    meta_ = {}
    
    if not meta or not isinstance(meta, dict):
        return meta_
    
    for name, desc in meta.items():
        pkeys = ['type', 'length', 'nullable', 'default']

        if isinstance(desc, (list, tuple)):
            params = {pk: desc[i] if len(desc) > i else None for i, pk in enumerate(iterable=pkeys)}
        elif isinstance(desc, dict):
            params = {pk: desc[pk] if pk in desc else desc[pk[:3]] if pk[:3] in desc else None for pk in pkeys}
        else:
            params = dict(zip(pkeys, [desc] + list(np.repeat(None, len(pkeys) - 1))))
        
        if params.get('type') is not None:
            params['type'] = get_mtype(params['type'])
        else:
            params['type'] = list(mtypes_base)[0]

        for pk in pkeys[1:]:
            if params.get(pk) is None:
                params[pk] = mtypes_base[params['type']][pk]

        meta_[name] = params

    return meta_


def format_data(
    df: pd.DataFrame,
    meta: Optional[dict] = None,
    index: Optional[str] = None,
    sort: Optional[str] = None,
    int_type: Literal['numeric', 'nullable', 'strict'] = 'numeric',
    bin_type: Literal['bool', 'category', 'numeric', 'strict'] = 'bool',
    obj_type: Literal['obj', 'json'] = 'obj',
    prevent_nulls: bool = False
) -> pd.DataFrame:
    """
    Format a DataFrame in order to match column descriptions,
    optimizing each column's type to its lowest memory size.

    Parameters
    ----------
    df : DataFrame
        Data to be formatted.
    meta : dict or dict-like, optional
        Column descriptions, a dictionary with specific meta information.

        See ``parse_meta`` function for more details.
    index : str, optional
        Name of the column to set as data index. If None, don't set data index.
    sort : str, optional
        Name of the column to sort the resulting data (ascending). If None, don't sort data.
    int_type : {'numeric', 'nullable', 'strict'}, default 'numeric'
        How to handle integer columns.

        - ``numeric`` :
            Try to convert to 'int' dtype if there are no null values or 'float' instead.
            This is better if we need to operate on data, allowing optimization of memory and modules communication.
        - ``nullable`` :
            Set dtype to Pandas 'Int', which allows null values in its integer type.
            Needed if we are performing database operations with nullable integer fields, preventing unhelping errors.
        - ``strict`` :
            Fill missing values and force 'int' dtype.
    bin_type : {'bool', 'category', 'numeric', 'strict'}, default 'bool'
        How to handle binary columns.

        - ``bool`` :
            Set dtype to 'bool' (default), setting missing values to False.
        - ``category`` :
            Set dtype to 'category' (True | False), useful for some data operations and visualization.
        - ``numeric`` :
            Treat column as a numeric value (1 or 0), in order to fit into numeric matrices and training tensors.
        - ``nullable`` :
            Set dtype to Pandas 'Int', which allows null values in its integer type.
            Needed if we are performing database operations with nullable integer fields, preventing unhelping errors.
        - ``strict`` :
            Fill missing values and force 'bool' dtype.
    obj_type : {'obj', 'json'}, default 'obj'
        How to handle object columns.

        - ``obj`` :
            Set dtype to Python object (default).
        - ``json`` :
            Set dtype to JSON string.
    prevent_nulls : bool, default False
        If True, fill missing values with their column default value (defined in ``meta`` dictionary for each column).

    Returns
    -------
    df : DataFrame
        Formatted data.

    Examples
    -----
    df = format_data(df, meta={
        'col1': 'cat',
        'col2': 'int',
        'col3': 'num',
        'col4': 'dtd'
    })
    """

    if meta is None:
        meta = infer_meta(df)

    meta = parse_meta(meta)

    if df is None or df.empty:
        df = pd.DataFrame(columns=list(meta))

    missing = [col for col in meta if col not in df.columns]
    for col in missing:
        df[col] = None

    df = df[list(meta)]
    df.columns = meta

    for col, desc in meta.items():
        dtype = desc['type']
        ddef = desc['default']

        try:
            if dtype == 'str':
                df[col] = format_data_str(df[col], ddef, prevent_nulls)
            elif dtype == 'obj':
                df[col] = format_data_obj(df[col], ddef, prevent_nulls, obj_type)
            elif dtype == 'cat':
                df[col] = format_data_cat(df[col], ddef, prevent_nulls)
            elif dtype == 'dts':
                df[col] = format_data_dts(df[col], ddef, prevent_nulls)
            elif dtype == 'dtd':
                df[col] = format_data_dtd(df[col], ddef, prevent_nulls)
            elif dtype == 'num':
                df[col] = format_data_num(df[col], ddef, prevent_nulls)
            elif dtype == 'int':
                df[col] = format_data_int(df[col], ddef, prevent_nulls, int_type)
            elif dtype == 'bin':
                df[col] = format_data_bin(df[col], ddef, prevent_nulls, bin_type)
        except Exception as e:
            raise ValueError('Error formatting column [{}] as [{}]: {}'.format(col, dtype, e)) from e

    if index is not None:
        df = df.set_index(index).sort_index()

    if sort is not None and sort != index:
        df = df.sort_values(sort)
        if index is None:
            df = df.reset_index(drop=True)

    return df


def format_data_str(
    x: pd.DataFrame | pd.Series,
    default: Optional[str] = None,
    prevent_nulls: bool = False
) -> pd.DataFrame | pd.Series:
    """
    Set data elements as string.

    Parameters
    ----------
    x : DataFrame or Series
        Data to be transformed.
    default : str, optional
        Default value to fill missing values.
    prevent_nulls : bool, optional
        If True, fill missing values with an empty string.
    
    Returns
    -------
    DataFrame or Series
        Transformed data.
    """
    d = x.astype(str, errors='ignore').replace({'None': None, 'nan': None, '<NA>': None, 'NaT': None})

    if is_numeric(x):
        d = d.replace(r'\.0$', '', regex=True)
    
    if default is not None:
        d = d.fillna(default)
    
    if prevent_nulls:
        d = d.fillna('')
    
    return d


def format_data_obj(
    x: pd.DataFrame | pd.Series,
    default: Optional[list | dict] = None,
    prevent_nulls: bool = False,
    obj_type: Literal['obj', 'json'] = 'obj'
) -> pd.DataFrame | pd.Series:
    """
    Set data elements as object/dict.

    Parameters
    ----------
    x : DataFrame or Series
        Data to be transformed.
    default : list or dict, optional
        Default value to fill missing values.
    prevent_nulls : bool, optional
        If True, fill missing values with an empty list.
    obj_type : {'obj', 'json'}, default 'obj'
        How to handle object columns.

        - ``obj`` :
            Set dtype to Python object (default).
        - ``json`` :
            Set dtype to JSON string.
    
    Returns
    -------
    DataFrame or Series
        Transformed data.
    """
    d = format_data_str(x)

    def parse_obj(v):
        if isinstance(v, (list, tuple, dict)):
            return v
        elif is_empty(v) or not isinstance(v, str):
            return

        v = str(v)
        
        if is_json(v):
            return json.loads(v)
        
        if is_object(v):
            return literal_eval(v)
        
        return v

    d = d.map(parse_obj)
    
    if default is not None:
        d = d.fillna(default)
    
    if prevent_nulls:
        d = d.fillna([])
    
    if obj_type == 'json':
        d = d.map(lambda v: json.dumps(v) if v is not None else None)

    return d


def format_data_cat(
    x: pd.DataFrame | pd.Series,
    default: Optional[str] = None,
    prevent_nulls: bool = False,
    categories: Optional[list[str]] = None,
    ordered: bool = False
) -> pd.DataFrame | pd.Series:
    """
    Set data elements as categorical.

    Parameters
    ----------
    x : DataFrame or Series
        Data to be transformed.
    default : str, optional
        Default value to fill missing values.
    prevent_nulls : bool, optional
        If True, fill missing values with an empty string.
    categories : list, optional
        Categories to set.
    ordered : bool, optional
        If True, set categories as ordered.
    
    Returns
    -------
    DataFrame or Series
        Transformed data.
    """
    d = format_data_str(x)
    
    if default is not None:
        d = d.fillna(default)
    
    if prevent_nulls:
        d = d.fillna('')
    
    d = set_categorical(d, categories=categories, ordered=ordered)

    return d


def format_data_dts(
    x: pd.DataFrame | pd.Series,
    default: Optional[datetime | str] = None,
    prevent_nulls: bool = False,
    dt_format: Optional[str] = None
) -> pd.DataFrame | pd.Series:
    """
    Set a data elements as datetime.

    Parameters
    ----------
    x : DataFrame or Series
        Data to be transformed.
    default : str, optional
        Default value to fill missing values.
    prevent_nulls : bool, optional
        If True, fill missing values with Pandas NaT value.
    dt_format : str, optional
        Date format to parse the series.
    
    Returns
    -------
    DataFrame or Series
        Transformed data.
    """
    d = pd.to_datetime(x, format=dt_format, errors='coerce')
    
    if isinstance(default, datetime):
        d = d.fillna(default)
    elif default == 'now':
        d = d.fillna(datetime.now())
    elif default == 'utcnow':
        d = d.fillna(datetime.utcnow())
    elif default is not None:
        d = d.fillna(pd.to_datetime(default, format=dt_format, errors='coerce'))
    
    if prevent_nulls:
        d = d.fillna(pd.NaT)
    
    d = d.dt.floor('s', ambiguous=False)

    return d


def format_data_dtd(
    x: pd.DataFrame | pd.Series,
    default: Optional[datetime | str] = None,
    prevent_nulls: bool = False,
    dt_format: Optional[str] = None
) -> pd.DataFrame | pd.Series:
    """
    Set a data elements as date.

    Parameters
    ----------
    x : DataFrame or Series
        Data to be transformed.
    default : str, optional
        Default value to fill missing values.
    prevent_nulls : bool, optional
        If True, fill missing values with Pandas NaT value.
    dt_format : str, optional
        Date format to parse the series.
    
    Returns
    -------
    DataFrame or Series
        Transformed data.
    """
    d = format_data_dts(
        x,
        default=default,
        prevent_nulls=prevent_nulls,
        dt_format=dt_format
    ).dt.floor('d', ambiguous=False)

    return d


def format_data_num(
    x: pd.DataFrame | pd.Series,
    default: Optional[float | str] = None,
    prevent_nulls: bool = False
) -> pd.DataFrame | pd.Series:
    """
    Set a data elements as float.

    Parameters
    ----------
    x : DataFrame or Series
        Data to be transformed.
    default : str, optional
        Default value to fill missing values.
    prevent_nulls : bool, optional
        If True, fill missing values with 0 value.
    
    Returns
    -------
    DataFrame or Series
        Transformed data.
    """
    d = pd.to_numeric(x, errors='coerce')

    if default == 'mean':
        d = d.fillna(d.mean())
    elif default == 'median':
        d = d.fillna(d.median())
    elif default == 'seq':
        d = d.fillna(np.arange(d.shape[0]))
    elif default is not None:
        d = d.fillna(default)
    
    if prevent_nulls:
        d = d.fillna(0.0)
    
    d = downcast_numeric(d, downcast='float')

    return d


def format_data_int(
    x: pd.DataFrame | pd.Series,
    default: Optional[float | str] = None,
    prevent_nulls: bool = False,
    int_type: Literal['numeric', 'nullable', 'strict'] = 'numeric'
) -> pd.DataFrame | pd.Series:
    """
    Set a data elements as integer.

    Parameters
    ----------
    x : DataFrame or Series
        Data to be transformed.
    default : str, optional
        Default value to fill missing values.
    prevent_nulls : bool, optional
        If True, fill missing values with 0 value.
    int_type : {'numeric', 'nullable', 'strict'}, default 'numeric'
        How to handle integer columns.

        - ``numeric`` :
            Try to convert to 'int' dtype if there are no null values or 'float' instead.
            This is better if we need to operate on data, allowing optimization of memory and modules communication.
        - ``nullable`` :
            Set dtype to Pandas 'Int', which allows null values in its integer type.
            Needed if we are performing database operations with nullable integer fields, preventing unhelping errors.
        - ``strict`` :
            Fill missing values and force 'int' dtype.
    
    Returns
    -------
    DataFrame or Series
        Transformed data.
    """
    d = format_data_num(x, default, prevent_nulls)

    if int_type == 'nullable':
        d = pd.to_numeric(d, errors='coerce').round().astype('Int64', errors='ignore')
        d = downcast_numeric(d, downcast='integer')
    elif int_type == 'strict':
        d = pd.to_numeric(d, errors='coerce').fillna(0).round().astype(int, errors='ignore')
        d = downcast_numeric(d, downcast='integer')
    elif d.isnull().any():
        d = pd.to_numeric(d, errors='coerce').round().astype(float, errors='ignore')
        d = downcast_numeric(d, downcast='float')
    else:
        d = pd.to_numeric(d, errors='coerce').round().astype(int, errors='ignore')
        d = downcast_numeric(d, downcast='integer')

    return d


def format_data_bin(
    x: pd.DataFrame | pd.Series,
    default: Optional[float | str] = None,
    prevent_nulls: bool = False,
    bin_type: Literal['bool', 'category', 'numeric', 'strict'] = 'bool'
) -> pd.DataFrame | pd.Series:
    """
    Set a data elements as boolean.

    Parameters
    ----------
    x : DataFrame or Series
        Data to be transformed.
    default : str, optional
        Default value to fill missing values.
    prevent_nulls : bool, optional
        If True, fill missing values with 0 value.
    bin_type : {'bool', 'category', 'numeric', 'strict'}, default 'bool'
        How to handle binary columns.

        - ``bool`` :
            Set dtype to 'bool' (default), setting missing values to False.
        - ``category`` :
            Set dtype to 'category' (True | False), useful for some data operations and visualization.
        - ``numeric`` :
            Treat column as a numeric value (1 or 0), in order to fit into numeric matrices and training tensors.
        - ``nullable`` :
            Set dtype to Pandas 'Int', which allows null values in its integer type.
            Needed if we are performing database operations with nullable integer fields, preventing unhelping errors.
        - ``strict`` :
            Fill missing values and force 'bool' dtype.
    
    Returns
    -------
    DataFrame or Series
        Transformed data.
    """
    bin_map = {
        'true': True, 't': True, 'yes': True, 'y': True, 'on': True, '1': True,
        'false': False, 'f': False, 'no': False, 'n': False, 'off': False, '0': False
    }

    d = format_data_str(x).str.lower().map(bin_map).astype(float, errors='ignore').apply(
        lambda v: True if v == 1 else False if v == 0 else np.NaN
    )

    if default is not None:
        d = d.fillna(default)

    if prevent_nulls:
        d = d.fillna(False)

    if bin_type == 'category':
        d = d.astype('category', errors='ignore')
    elif bin_type == 'numeric':
        if d.isnull().any():
            d = pd.to_numeric(d, errors='coerce').astype(float, errors='ignore')
            d = downcast_numeric(d, downcast='float')
        else:
            d = pd.to_numeric(d, errors='coerce').astype(int, errors='ignore')
            d = downcast_numeric(d, downcast='integer')
    elif bin_type == 'nullable':
        d = d.astype('boolean', errors='ignore')
    elif bin_type == 'strict':
        d = d.fillna(False)
    else:
        d = d.fillna(False).astype(bool)

    return d


def set_categorical(
    x: pd.DataFrame | pd.Series,
    categories: Optional[list[str]] = None,
    ordered: bool = False
) -> pd.DataFrame | pd.Series:
    d = to_pandas(x)

    if len(d.shape) == 1:
        d = pd.Categorical(unset_categorical(d), categories=categories, ordered=ordered)
    else:
        for col in d.columns:
            d[col] = set_categorical(d[col], categories=categories, ordered=ordered)

    return d


def unset_categorical(
    x: pd.DataFrame | pd.Series
) -> pd.DataFrame | pd.Series:
    d = to_pandas(x)

    if len(d.shape) == 1:
        if d.dtype.name == 'category':
            d = d.astype(str, errors='ignore').replace({'None': None, 'nan': None})
    else:
        for col in d.columns:
            d[col] = unset_categorical(d[col])

    return d


def downcast_numeric(
    x: pd.DataFrame | pd.Series,
    downcast: str = 'float'
) -> pd.DataFrame | pd.Series:
    """
    Downcast numeric values to a smaller dtype in order to optimize memory usage (columnwise).

    Parameters
    ----------
    x : DataFrame or Series
        Numeric series to downcast.
    downcast : str, default 'float'
        Type to downcast to.
    
    Returns
    -------
    DataFrame or Series
        Data with the minimum dtype needed to store the values.
    """
    d = to_pandas(x)

    if len(d.shape) == 1:
        if d.shape[0] > 0:
            dn = pd.to_numeric(d, errors='coerce', downcast=downcast)
            ndtype = dn.dtype if np.equal(dn, d)[d.notnull()].all() else d.dtype
            d = pd.to_numeric(d, errors='coerce').astype(ndtype)
        else:
            d = d.astype(d.dtype)
    else:
        for col in d.columns:
            d[col] = downcast_numeric(d[col], downcast=downcast)

    return d


def get_columns(
    x: pd.DataFrame,
    mtypes: Optional[list] = None,
    exclude: Optional[list] = None
):
    """
    Get columns from a DataFrame that match specific meta types.

    Parameters
    ----------
    x : DataFrame
        DataFrame to get columns from.
    mtypes : list, optional
        Meta types to match.
        If not provided, it will match all meta types.
    exclude : list, optional
        Columns to exclude from the returned columns.
        If not provided, it will not exclude any column.

    Returns
    -------
    list
        Columns that match the specified meta types.
    """
    mtypes = mtypes or list(mtypes_base)
    exclude = exclude or []
    meta = infer_meta(x, exclude=exclude)
    columns = []

    for col in x.columns:
        if col in exclude:
            continue

        if meta[col] in mtypes:
            columns.append(col)

    return columns


def get_histogram(
        d,
        metric: Literal['count', 'prob'] = 'count',
        bins=None,
        weights=None,
        discrete=False,
        outliers: Literal['keep', 'group', 'remove'] = 'keep',
        ol_std=3
):
    ds = to_pandas(d, pdtype='Series').dropna()

    if weights is not None:
        weights = np.array(weights)
        dw = DescrStatsW(ds, weights=weights)
        mean, std = dw.mean, dw.std
    else:
        mean, std = ds.agg(['mean', 'std'])

    if outliers == 'group':
        ds_in = np.abs(ds - mean) / std < ol_std
        ds = ds.where(ds_in, ds[ds_in].max() + 1)
    elif outliers == 'remove':
        ds_in = np.abs(ds - mean) / std < ol_std
        ds = ds.loc[ds_in]
        if weights is not None:
            weights = weights[ds_in]

    normalize = True if metric in ['prob', 'probability', 'density'] else False
    hist = ds.value_counts(normalize=normalize).sort_index()

    if discrete:
        if bins is not None:
            bin_edges = np.arange(ds.min(), ds.max() + bins, bins)
        else:
            bin_edges = np.asarray([])
    else:
        if bins is not None:
            bin_edges = np.histogram_bin_edges(ds, bins=bins)
        else:
            bin_edges = np.histogram_bin_edges(ds, bins='auto')
            if bin_edges.size > hist.size:
                bin_edges = int(np.min([hist.size, 10]))

    if (is_array(bin_edges) and len(bin_edges) > 0) or (is_number(bin_edges) and bin_edges > 0):
        hist, bin_edges = np.histogram(ds, bins=bin_edges, weights=weights)

        idx = [(bin_edges[i] + bin_edges[i + 1]) / 2 for i in range(len(bin_edges) - 1)]
        hist = pd.Series(hist, index=idx)
        if normalize:
            hist /= np.sum(hist)
    else:
        idx = np.arange(hist.index.min(), hist.index.max() + 1).astype(int)
        hist = hist.reindex(idx).fillna(0)

    return hist, bin_edges


def get_kde(x, p=100, weights=None, bw=None, outliers='keep', ol_std=3):
    # bw -> N, scott (default), silverman
    # outliers -> keep (default), group or remove

    x = np.array(x)
    nobs, ndim = array_shape(x)

    y = None
    y_mean, y_std = None, None

    if ndim > 1:
        y = x[:, 1]
        x = x[:, 0]

    if weights is not None:
        weights = np.array(weights)

        xw = DescrStatsW(x, weights=weights)
        x_mean, x_std = xw.mean, xw.std

        if ndim > 1:
            yw = DescrStatsW(y, weights=weights)
            y_mean, y_std = yw.mean, yw.std
    else:
        x_mean, x_std = x.mean(), x.std()

        if ndim > 1:
            y_mean, y_std = y.mean(), y.std()

    if outliers == 'group':
        x_in = np.abs(x - x_mean) / x_std < ol_std
        x = np.where(x_in, x, x[x_in].max() + 1)
        if ndim > 1:
            y_in = np.abs(y - y_mean) / y_std < ol_std
            y = np.where(y_in, y, y[y_in].max() + 1)
    elif outliers == 'remove':
        if ndim > 1:
            x_in = np.abs(x - x_mean) / x_std < ol_std
            y_in = np.abs(y - y_mean) / y_std < ol_std
            xy_in = np.all(np.vstack([x_in, y_in]), axis=0)
            x = x[xy_in]
            y = y[xy_in]
            weights = weights[xy_in]
        else:
            x_in = np.abs(x - x_mean) / x_std < ol_std
            x = x[x_in]
            weights = weights[x_in]

    if ndim > 1:
        if is_number(p):
            xp = np.linspace(x.min(), x.max(), p)
            yp = np.linspace(y.min(), y.max(), p)
        else:
            p = np.array(p)
            pdim = array_shape(p)[1]
            if pdim > 1:
                xp = p[:, 0]
                yp = p[:, 1]
            else:
                xp = yp = p

        xx, yy = np.meshgrid(xp, yp, indexing='ij')
        p = np.vstack([xx.ravel(), yy.ravel()])
        x = np.vstack([x, y])
        k = xp.shape[0]
    else:
        if is_number(p):
            p = np.linspace(x.min(), x.max(), p)
        else:
            p = np.array(p)
        k = len(p)

    kde = gaussian_kde(
        x,
        bw_method=bw,
        weights=weights
    ).evaluate(p).reshape(np.repeat(k, ndim)).T

    return kde


def filter_agg(d, agg='mean', window=7, min_periods=1, center=False, drop_na=True):
    ds = d.rolling(
        window=window,
        min_periods=min_periods,
        center=center
    ).agg(agg)

    if drop_na:
        ds = ds.dropna()

    return ds


def filter_smooth(d, k=21, method='hp', sg_order=2):
    ds = to_pandas(d)

    if len(ds.shape) == 1:
        ds = ds.dropna()
        idx = ds.index

        if ds.shape[0] > 0:
            if method == 'hp':  # Hodrick-Prescott
                ds = smt.filters.hpfilter(ds, k)[1]
            elif method == 'sg':  # Savitzky-Golay
                ds = savgol_filter(ds, k, sg_order)

            ds = pd.Series(ds, index=idx)
    else:
        for col in d.columns:
            ds[col] = filter_smooth(ds[col], k=k, method=method, sg_order=sg_order)

    return ds


def pivot_data(d, index=None, columns=None, values=None, aggfunc='last', **kwargs):
    df = to_pandas(d, pdtype='DataFrame')

    if values is None:
        values = df.columns[0]
    if index is None:
        index = df.index.names[0]
        df = df.reset_index(level=0)
    if columns is None:
        columns = df.index.names[0]
        df = df.reset_index(level=0)

    df = df.pivot_table(
        index=index,
        columns=columns,
        values=values,
        aggfunc=aggfunc,
        **kwargs
    ).rename_axis(index=None).rename_axis(index=None, axis=1).fillna(0)

    return df


def is_number(x):
    if x is None:
        return False

    try:
        n = float(x)
    except (TypeError, ValueError):
        return False

    if np.isnan(n):
        return False

    return True


def is_string(x):
    if x is None:
        return False

    return isinstance(x, str)


def is_object(x):
    if x is None:
        return False

    try:
        literal_eval(str(x))
    except Exception:
        return False

    return True


def is_json(x):
    if x is None:
        return False

    try:
        json.loads(str(x))
    except (TypeError, ValueError):
        return False

    return True


def is_array(x):
    return isinstance(x, (list, tuple)) or (hasattr(x, '__array__') and hasattr(x, '__iter__'))


def is_pandas(x):
    return isinstance(x, (pd.DataFrame, pd.Series, pd.Index))


def is_frame(x):
    return isinstance(x, pd.DataFrame)


def is_series(x):
    return isinstance(x, pd.Series)

def is_index(x):
    return isinstance(x, pd.Index)


def is_empty(x):
    if x is None:
        return True
    elif is_array(x):
        return len(x) == 0
    elif pd.isnull(x) or not x:
        return True

    return False


def is_numeric(x):
    if is_empty(x):
        return False

    if not is_array(x):
        return is_number(x)

    if hasattr(x, 'dtype'):
        return pd.api.types.is_numeric_dtype(x.dtype)

    if hasattr(x, 'dtypes'):
        return all([pd.api.types.is_numeric_dtype(dtype) for dtype in x.dtypes])

    try:
        pd.to_numeric(x, errors='raise')
    except (TypeError, ValueError):
        return False

    return True


def array_shape(x):
    if not is_array(x):
        return

    if isinstance(x, dict):
        x = list(x)

    x = np.asarray(x)

    if is_empty(x):
        return 0, 0

    if x.ndim > 1:
        return x.shape
    elif x.ndim > 0:
        return x.shape[0], 1
    else:
        return 1, 1


def array_adjust(x, k=None, dtype=None):
    if not is_array(x):
        x = [x]

    x = np.asarray(x)

    if x.ndim < 2:
        nobs, kvar = array_shape(x)
        x = x.reshape((nobs, kvar))
    else:
        nobs, kvar = x.shape

    if k is not None and k > kvar:
        if kvar == 1:
            x = np.repeat(x, k - kvar + 1, axis=1)
        else:
            x = np.column_stack([x, np.ones((nobs, k - kvar))])
    elif k == 0:
        x = x.ravel()

    if dtype is not None:
        x = x.astype(dtype)

    return x


def array_get_row(x, i=-1, default=None):
    if isinstance(x, pd.DataFrame):
        try:
            return x.iloc[i]
        except (KeyError, ValueError, IndexError):
            return default or pd.Series(index=x.columns)

    try:
        return x[i]
    except (KeyError, ValueError, IndexError):
        return default or []


def array_pop(x, default=None):
    return array_get_row(x, -1, default=default)


def array_shift(x, default=None):
    return array_get_row(x, 0, default=default)


def coalesce(*values) -> Any:
    """
    Return the first non-null value or None if all values are null
    """
    return next((v for v in values if v is not None and pd.notnull(v)), None)


def format_number(n, decimals=2, dec_force=False, dec_sep=',', thou_sep='.'):
    if pd.isnull(n):
        return ''
    elif not is_number(n):
        return str(n)

    try:
        x = float(n)
        if decimals is not None:
            x = round(x, decimals)

        parts = str(x).split('.')
        x_int = abs(int(parts[0])) if len(parts) > 0 else 0
        s_dec = parts[1] if len(parts) > 1 else '0'
    except (TypeError, ValueError):
        return str(n)

    s = '-' if x < 0 else ''

    s_int = '{:,}'.format(x_int).replace(',', thou_sep)
    s = '{}{}'.format(s, s_int)

    s_dec = s_dec.rstrip('0')
    if decimals is None:
        decimals = len(s_dec)
    if dec_force and len(s_dec) < decimals:
        s_dec = s_dec.ljust(decimals, '0')

    if s_dec != '':
        s = '{}{}{}'.format(s, dec_sep, s_dec)

    return s


def parse_size_bytes(size):
    units = 'BKMG'

    m = re.match(r'^([0-9]+)(.+)$', str(size).upper())
    if m is None:
        return None

    unit = m.group(2)[0] if len(m.group(2)) > 0 else 'B'
    power = units.index(unit) if unit in units else 0
    nbytes = int(m.group(1)) * (1024 ** power)

    return nbytes


def infer_pdtype(x: NDFrame) -> str | None:
    """
    Infers the type of Pandas object.

    Parameters
    ----------
    x : NDFrame
        Pandas object to infer its type from.
    
    Returns
    -------
    str
        Inferred Pandas object type.
        None if is not a Pandas object or its type is not Series, DataFrame or Index.
    """
    if isinstance(x, pd.Series):
        return 'Series'
    elif isinstance(x, pd.DataFrame):
        return 'DataFrame'
    elif isinstance(x, pd.Index):
        return 'Index'


def to_pandas(
    d,
    orient: Literal['columns', 'index', 'tight'] = 'columns',
    pdtype: Optional[Literal['DataFrame', 'Series']] = None
) -> pd.DataFrame | pd.Series:
    if hasattr(d, '__array__'):  # An array-like object
        if hasattr(d, 'index'):  # A Pandas object (DataFrame/Series)
            ds = d.copy()
        else:
            if len(d.shape) > 1:
                ds = pd.DataFrame(d)
            else:
                ds = pd.Series(d)
    elif isinstance(d, dict):  # A dictionary
        ds = pd.DataFrame.from_dict(d, orient=orient)
    elif isinstance(d, (list, tuple)):  # A list or tuple
        if len(np.asarray(d).shape) > 1:
            ds = pd.DataFrame.from_dict(
                dict(zip(np.arange(len(d)), d)),
                orient=orient
            )
        else:
            ds = pd.Series(d)
    else:
        raise ValueError('Invalid type [{}]'.format(type(d)))

    if pdtype == 'DataFrame' and isinstance(ds, pd.Series):
        ds = pd.DataFrame(ds)
        if orient == 'index':
            ds = ds.T
    elif pdtype == 'Series' and isinstance(ds, pd.DataFrame):
        ds = ds[ds.columns[0]]

    return ds


def to_frame(
    d,
    orient: Literal['columns', 'index', 'tight'] = 'columns'
) -> pd.DataFrame:
    return to_pandas(d, orient=orient, pdtype='DataFrame')


def to_series(
    d,
    orient: Literal['columns', 'index', 'tight'] = 'columns'
) -> pd.Series:
    return to_pandas(d, orient=orient, pdtype='Series')


def to_tensor(
    d,
    columns=None,
    mtypes=None,
    exclude=None,
    encode=False,
    scale=False,
    impute=False,
    prevent_nulls=False,
    **kwargs
):
    encoder = kwargs.get('encoder')
    scaler = kwargs.get('scaler')
    imputer = kwargs.get('imputer')

    pdtype = infer_pdtype(d)
    ds = to_frame(d)

    if is_empty(columns):
        columns = get_columns(ds, mtypes=mtypes, exclude=exclude)
        if is_empty(columns):
            return to_pandas(ds, pdtype=pdtype), dict(encoder=encoder, scaler=scaler, imputer=imputer)

    na_value = 0 if prevent_nulls else None
    ds = numerize_data(ds, columns=columns, na_value=na_value)

    if encode:
        na_value = 'none' if prevent_nulls else None
        ds, encoder = encode_data(ds, columns=columns, na_value=na_value, encoder=encoder)
        columns = list(ds.columns)

    if scale:
        na_value = 0 if prevent_nulls else None
        ds, scaler = scale_data(ds, columns=columns, na_value=na_value, scaler=scaler)

    if impute:
        ds, imputer = impute_data(ds, columns=columns, imputer=imputer)

    return to_pandas(ds, pdtype=pdtype), dict(encoder=encoder, scaler=scaler, imputer=imputer)


def numerize_data(d, columns=None, na_value=None):
    columns = columns or []

    pdtype = infer_pdtype(d)
    ds = to_pandas(d, pdtype='DataFrame')

    if len(columns) == 0:
        columns = ds.columns

    columns = get_columns(ds[columns], mtypes=['num', 'int', 'bin'])

    for col in columns:
        if na_value is not None:
            ds[col] = ds[col].fillna(na_value)

        ds[col] = ds[col].astype(float, errors='ignore')
        ds[col] = downcast_numeric(ds[col], downcast='float')

    return to_pandas(ds, pdtype=pdtype)


def encode_data(d, columns=None, na_value=None, encoder=None):
    columns = columns or []

    pdtype = infer_pdtype(d)
    ds = to_pandas(d, pdtype='DataFrame')

    if len(columns) == 0:
        columns = ds.columns

    columns = get_columns(ds[columns], mtypes=['str', 'cat'])

    if len(columns) == 0:
        return to_pandas(ds, pdtype=pdtype), encoder

    x = ds[columns].astype(str)

    if na_value is not None:
        x = x.fillna(na_value)

    x = x.to_dict(orient='records')

    if not encoder:
        encoder = DictVectorizer(separator='__', sparse=False, sort=False)
        x = encoder.fit_transform(x)
    else:
        x = encoder.transform(x)

    x = pd.DataFrame(x, columns=encoder.get_feature_names_out(), index=ds.index)
    ds[x.columns] = x
    ds = ds.drop(columns=columns)

    return to_pandas(ds, pdtype=pdtype), encoder


def scale_data(d, columns=None, log=False, na_value=0, scaler=None):
    columns = columns or []

    pdtype = infer_pdtype(d)
    ds = to_pandas(d, pdtype='DataFrame')

    if len(columns) == 0:
        columns = ds.columns

    columns = get_columns(ds[columns], mtypes=['num', 'int', 'bin'])

    if len(columns) == 0:
        return to_pandas(ds, pdtype=pdtype), scaler

    if log is True:
        log_cols = columns
    elif is_array(log):
        log_cols = [c for c in log if c in columns]
    else:
        log_cols = []

    x = ds[columns].astype(float)

    if na_value is not None:
        x = x.fillna(na_value)

    if len(log_cols) > 0 and np.nanmin(x[log_cols].values) >= 0:
        x[log_cols] = np.log1p(x[log_cols].values)

    x = x.values

    if not scaler:
        scaler = StandardScaler()
        x = scaler.fit_transform(x)
    else:
        x = scaler.transform(x)

    x = pd.DataFrame(x, columns=columns, index=ds.index)
    ds[x.columns] = x

    return to_pandas(ds, pdtype=pdtype), scaler


def inverse_scale_data(d, columns=None, log=None, scaler=None):
    columns = columns or []

    pdtype = infer_pdtype(d)
    ds = to_pandas(d, pdtype='DataFrame')

    if len(columns) == 0:
        columns = ds.columns

    columns = get_columns(ds[columns], mtypes=['num', 'int', 'bin'])

    if len(columns) == 0:
        return to_pandas(ds, pdtype=pdtype), scaler

    if log is True:
        log_cols = columns
    elif is_array(log):
        log_cols = [c for c in log if c in columns]
    else:
        log_cols = []

    x = ds[columns].astype(float).values

    x = scaler.inverse_transform(x)

    x = pd.DataFrame(x, columns=columns, index=ds.index)
    ds[x.columns] = x

    if len(log_cols) > 0:
        ds[log_cols] = np.expm1(ds[log_cols].values)

    return to_pandas(ds, pdtype=pdtype)


def impute_data(d, columns=None, imputer=None, estimator=None, verbose=0, seed=None):
    columns = columns or []

    pdtype = infer_pdtype(d)
    ds = to_pandas(d, pdtype='DataFrame')

    if len(columns) == 0:
        columns = ds.columns

    columns = get_columns(ds[columns], mtypes=['num', 'int', 'bin'])

    if len(columns) == 0:
        return to_pandas(ds, pdtype=pdtype), imputer

    x = ds[columns].astype(float).values

    if not imputer:
        imputer = IterativeImputer(
            estimator=estimator,
            verbose=verbose,
            random_state=seed
        )
        x = imputer.fit_transform(x)
    else:
        x = imputer.transform(x)

    x = pd.DataFrame(x, columns=columns, index=ds.index)
    ds[x.columns] = x

    return to_pandas(ds, pdtype=pdtype), imputer


def agg_func(x, sort, group, funcs):
    g = x.sort_values(sort).groupby(group)

    agg = []

    for col, func, name in funcs:
        if func == 'sum':
            agg.append(g[col].sum().rename(name))
        elif func == 'avg':
            agg.append(g[col].mean().rename(name))
        elif func == 'first':
            agg.append(g[col].first().rename(name))
        elif func == 'last':
            agg.append(g[col].last().rename(name))

    agg = pd.concat(agg, axis=1).fillna(0)

    return agg


# Numpy and Itertools version of Pandas groupby/apply function (huge performance improvement)
def apply_agg_func(
        data,
        by,
        func,
        sort=None,
        columns=None,
        window=None,
        min_periods=None,
        center=False,
        verbose=0,
        **kwargs
):
    df = to_pandas(data, pdtype='DataFrame')

    if isinstance(by, str):
        by = [by]
    elif not isinstance(by, (list, tuple)):
        raise ValueError('Invalid <by> [{}]'.format(by))

    if isinstance(sort, str):
        sort = [sort]
    elif not isinstance(sort, (list, tuple)):
        sort = []

    is_single = False
    if isinstance(func, dict):
        columns = list(func)
    elif isinstance(columns, str):
        is_single = True
        columns = [columns]
    elif not isinstance(columns, (list, tuple)):
        columns = list(df.columns)

    if isinstance(func, dict):
        for col, flist in func.items():
            func[col] = flist if isinstance(flist, (list, tuple)) else [flist]
            for i, fn in enumerate(func[col]):
                if isinstance(fn, str) and hasattr(np, fn):
                    fn = getattr(np, fn)
                elif not callable(fn):
                    raise ValueError('Invalid <func> [{}]'.format(fn))
                func[col][i] = fn
    elif isinstance(func, (list, tuple)):
        for i, fn in enumerate(func):
            if isinstance(fn, str) and hasattr(np, fn):
                fn = getattr(np, fn)
            elif not callable(fn):
                raise ValueError('Invalid <func> [{}]'.format(fn))
            func[i] = fn
        func = {col: func for col in columns}
    elif isinstance(func, str) and hasattr(np, func):
        func = {col: [getattr(np, func)] for col in columns}
    elif callable(func):
        func = {col: [func] for col in columns}
    else:
        raise ValueError('Invalid <func> [{}]'.format(func))

    missing = list(set(by + columns) - set(df.columns))
    if len(missing) > 0:
        raise ValueError('Invalid <columns> {}'.format(missing))

    df = df.sort_values(by + sort)[by + sort + columns]

    by_keys = np.arange(len(by)).tolist()
    sequences = [list(g) for k, g in groupby(df.values, key=itemgetter(*by_keys))]
    seq = {'{}__{}'.format(col, fn.__name__): [] for col, flist in func.items() for fn in flist}

    sequences_ = tqdm(sequences) if verbose > 0 else sequences
    for x in sequences_:
        for col, flist in func.items():
            col_seq = np.array(x)[:, list(df.columns).index(col)].tolist()
            for fn in flist:
                if window is not None:
                    min_periods = min_periods or 0
                    if center:
                        cur_seq = [
                            fn(col_seq[np.max([i - window + 1, 0]):np.min([i + window, len(col_seq)])], **kwargs)
                            for i in np.arange(len(col_seq))
                        ]
                    else:
                        cur_seq = [
                            fn(col_seq[:(i + 1)][-window:], **kwargs)
                            if window == 0 or (window <= i + 1 or (0 < min_periods <= i + 1))
                            else np.nan
                            for i in np.arange(len(col_seq))
                        ]
                else:
                    cur_seq = fn(col_seq, **kwargs)

                seq['{}__{}'.format(col, fn.__name__)].append(cur_seq)

    for key, x in seq.items():
        df[key] = [j for i in x for j in i]

    df = df.set_index(by + sort)[list(seq)]

    if is_single:
        return df[df.columns[0]]

    return df
