from adjustText import adjust_text
import calendar
from datetime import date, datetime
from dateutil.relativedelta import relativedelta
import husl
from IPython.display import display, HTML
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import pandas as pd
from pandas.io.formats.style import Styler
import seaborn as sns
from scipy.cluster import hierarchy
from statsmodels.stats.weightstats import DescrStatsW
import tempfile

from typing import (
    Optional
)

from ..app import App

from .dates import (
    Freq,
    is_timeseries,
    ts_range
)
from .learning import (
    clusters_silhouette,
    cv_model_train,
    feature_collinearity,
    feature_correlations,
    feature_regression,
    feature_stat_test,
    feature_vif,
    shap_values
)
from .stat import (
    LeastSquaresEstimator,
    LocalKernelEstimator,
    KernelDensityEstimator,
    Stat
)
from .helpers import (
    array_shape,
    filter_agg,
    filter_smooth,
    format_number,
    get_columns,
    get_histogram,
    is_array,
    is_empty,
    is_number,
    is_numeric,
    to_pandas,
    to_tensor
)

plt.rc('font', size=12)
plt.rc('figure', facecolor='white')


# Colors


colors = {
    'blue': '#1f77b4',
    'green': '#2ca02c',
    'orange': '#ff7f0e',
    'red': '#d62728',
    'purple': '#9467bd',
    'yellow': '#bcbd22',
    'brown': '#8c564b',
    'pink': '#e377c2',
    'cyan': '#17becf',
    'grey': '#7f7f7f'
}
palette = list(colors)
table_styles = [
    {
        'selector': 'caption',
        'props': [
            ('font-weight', 'bold'),
            ('margin-bottom', '8px')
        ]
    },
    {
        'selector': 'thead th.col_heading',
        'props': [
            ('width', '100px'),
            ('text-align', 'center')
        ]
    },
    {
        'selector': 'th',
        'props': [
            ('border', '1px solid #e0e0e0'),
            ('background-color', '#f7f7f7')
        ]
    },
    {
        'selector': 'td',
        'props': [
            ('border', '1px solid #e3eaef')
        ]
    }
]


# Helpers


def get_color(color, alpha=None):
    if '-' in color:
        color, alpha = color.split('-')

    if alpha == 'alpha':
        alpha = 0.1
    elif alpha == 'light':
        alpha = 0.6
    elif alpha == 'bold':
        alpha = 1.4
    elif alpha == 'bolder':
        alpha = 1.9
    elif is_number(alpha):
        alpha = np.clip(float(alpha), 0, 2)
    else:
        alpha = None

    color = colors[color] if color in colors else color

    if alpha is not None:
        color = mcolors.to_hex(
            build_color_seq_map(color)(alpha / 2)[:3]
        )

    return color


def build_color_cat_map(colors, n=None, gamma=1.0):
    colors = [tuple(mcolors.to_rgb(get_color(c))) for c in colors]
    N = n or len(colors)

    return mcolors.LinearSegmentedColormap.from_list('categorical', colors, N=N, gamma=gamma)


def build_color_seq_map(color, n=50, beta=0.05, gamma=1.0, reversed=False):
    color = tuple(mcolors.to_rgb(get_color(color)))
    hue, sat, lum = husl.rgb_to_husl(*color)

    lum_light = 100 - (100 - lum) * beta
    lum_dark = lum * lum / lum_light
    sat_light = 100 - (lum / lum_light) * (100 - sat)
    sat_dark = sat * sat / sat_light

    color_light = tuple(husl.husl_to_rgb(hue, sat_light, lum_light))
    color_dark = tuple(husl.husl_to_rgb(hue, sat_dark, lum_dark))

    colors = [color_light, color, color_dark]
    if reversed:
        colors = colors[::-1]
    N = 2 * n + 1

    return mcolors.LinearSegmentedColormap.from_list('sequential', colors, N=N, gamma=gamma)


def build_color_div_map(colors, n=50, beta=0.05, gamma=1.0):
    if len(colors) > 2:
        c_ext = [colors[0], colors[2]]
        c_mid = colors[1]
    elif len(colors) == 2:
        c_ext = colors
        c_mid = 'white'
    else:
        return

    c_ext = [tuple(mcolors.to_rgb(get_color(c))) for c in c_ext]
    c_mid = tuple(mcolors.to_rgb(get_color(c_mid)))

    c_ext_dark = []

    for color in c_ext:
        hue, sat, lum = husl.rgb_to_husl(*color)

        lum_light = 100 - (100 - lum) * beta
        lum_dark = lum * lum / lum_light
        sat_light = 100 - (lum / lum_light) * (100 - sat)
        sat_dark = sat * sat / sat_light

        color_dark = tuple(husl.husl_to_rgb(hue, sat_dark, lum_dark))

        c_ext_dark.append(color_dark)

    colors = [
        c_ext_dark[0],
        c_ext[0],
        c_mid,
        c_ext_dark[1],
        c_ext[1]
    ]
    N = 4 * n + 2

    return mcolors.LinearSegmentedColormap.from_list('divergent', colors, N=N, gamma=gamma)


def get_color_map(cm=None, keys=None, rtype=None, **kwargs):
    """
    Parse custom colors and transform into Matplotlib / RGB standards.

    Parameters
    ----------
    cm : list-like of colors, dict-like of (key, color) or str, optional
        Set of colors to be parsed and transformed.
        If (key, color) pairs are given, colors will be assigned to it's corresponding matching key,
        ignoring the ``keys`` parameter.
        If a single color string is given, it will be transformed and assigned to each of the keys.
        If None, fetch colors from the main palette in order until all keys are assigned.
    keys : list-like or int, optional
        Each color must be assigned to it's corresponding keys,
        so the resulting object must have as many colors as keys provided.
        If int, this amount of keys will be assumed and matched with the same number of colors.
        If None, will assume as many keys as colors we have in the main palette.
    rtype : {'list', 'dict', 'pcmap'}, default 'list'
        What kind of object we need to be returned.

        - ``list`` :
            A list of colors, as many as keys we have.
        - ``dict`` :
            A dictionary of (key, color) pairs.
        - ``pcmap`` :
            A Matplotlib ``LinearSegmentedColormap`` built with the resulting colors list.
    **kwargs
        These parameters will be passed to ``build_color_cat_map``.
    """

    if rtype is None:
        # Return a dictionary only if we have keys in either params
        if isinstance(keys, (list, tuple)) or isinstance(cm, dict):
            rtype = 'dict'
        else:
            rtype = 'list'

    if isinstance(cm, dict):  # Keys are specified in cm param
        keys = list(cm.keys())
        n = len(keys)
    elif isinstance(keys, (list, tuple)):  # Keys is a list of names to return along with colors
        n = len(keys)
    elif is_number(keys):  # Keys is the number of colors to return
        n = int(keys)
    elif isinstance(cm, (list, tuple)):  # If no valid keys, listed colors
        n = len(cm)
    else:  # If no valid keys and colors, return the main palette colors
        n = len(palette)

    if isinstance(cm, dict):  # Dictionary values containing colors
        cmap = list(cm.values())[:n]
    elif isinstance(cm, (list, tuple)):  # Combine keys and colors lists
        cmap = list(cm)[:n]
    elif isinstance(cm, str):  # Same color repeated
        cmap = list(np.repeat(cm, n))
    else:  # First N colors in palette
        cmap = palette[:n]

    # Translate and return
    cmap = [get_color(c) for c in cmap]
    if rtype == 'dict':
        cmap = dict(zip(keys, cmap))
    elif rtype == 'pcmap':
        cmap = build_color_cat_map(cmap, **kwargs)

    return cmap


def get_color_luminance(color):
    # Calculate the relative luminance of a color according to W3C standards (Seaborn)
    rgb = mcolors.colorConverter.to_rgba_array(color)[:, :3]
    rgb = np.where(rgb <= .03928, rgb / 12.92, ((rgb + .055) / 1.055) ** 2.4)
    lum = rgb.dot([.2126, .7152, .0722])

    try:
        return lum.item()
    except ValueError:
        return lum


def get_text_color(bg_color='white', cm=None):
    cm = cm or ['#000000', '#ffffff']

    lum = get_color_luminance(bg_color)
    color = cm[0] if lum > .408 else cm[1]

    return color


def get_scale_range(vmin, vmax, n=1):
    dv = abs(vmax - vmin)
    step = np.ceil(dv / n)

    vmin_ = np.ceil(vmin)
    vmax_ = vmin_ + (n * step)

    return np.arange(vmin_, vmax_ + step, step)


def get_params(params_dict, split_keys=None, arg_name=None, **kwargs):
    split_keys = split_keys or []
    is_split = len(split_keys) > 0

    params = {}
    if is_split:
        params = {sk: {} for sk in split_keys}
    if arg_name is not None:
        arg_params = kwargs.get(arg_name, params)
        params = arg_params.copy()
    else:
        arg_params = None

    if is_split:
        for i, sk in enumerate(split_keys):
            params[sk] = {
                k: params[sk].get(k, v) if sk in params else v
                for k, v in params_dict.items()
            }
            for k, v in params[sk].items():
                if k in kwargs:
                    if isinstance(kwargs[k], dict):
                        params[sk][k] = kwargs[k][sk]
                    elif isinstance(kwargs[k], list):
                        params[sk][k] = kwargs[k][i]
                    else:
                        params[sk][k] = kwargs[k]

        params = {k: params[k] for k in split_keys}
    else:
        params = {
            k: params.get(k, v)
            for k, v in params_dict.items()
        }
        for k, v in params.items():
            if k in kwargs:
                params[k] = kwargs[k]

        if arg_params is not None:
            params |= arg_params

    return params


def parse_col_params(col_params, data=None, disp=False, share=False, margin=0.05):
    cols = list(col_params)

    cmap = [p.get('cm') for p in col_params.values()]
    if any(v is None for v in cmap):
        cmap = None
    cmap = get_color_map(cmap, cols)

    for c in cols:
        cp = col_params[c]

        if cp.get('yaxis') == 'split':
            cp['yaxis'] = 'right' if len(cols) > 1 and c == cols[-1] else 'left'

        if cp.get('agg') is not None:
            if not isinstance(cp['agg'], (list, tuple)):
                cp['agg'] = [cp['agg']]

        if cp.get('smooth') is not None:
            if isinstance(cp['smooth'], bool):
                cp['smooth'] = [int(cp['smooth'])]
            elif not isinstance(cp['smooth'], (list, tuple)):
                cp['smooth'] = [cp['smooth']]

        if cp.get('ylim') is not None:
            cp['ymin'], cp['ymax'] = cp['ylim']

        if cp.get('stacked') is not None:
            cp['disp'] = 'stacked'

        cp['cm'] = cmap.get(c)

        if cp.get('label') is True:
            cp['label'] = c
        elif cp.get('label') is False:
            cp['label'] = None

        col_params[c] = cp

    yaxis_arr = np.unique([p.get('yaxis', 'left') for p in col_params.values()])
    axes_map = {yaxis_arr[0]: cols}
    if len(yaxis_arr) > 1:
        axes_map[yaxis_arr[0]] = [c for c in cols if col_params[c].get('yaxis') == yaxis_arr[0]]
        axes_map[yaxis_arr[1]] = [c for c in cols if col_params[c].get('yaxis') != yaxis_arr[0]]

    for axis, cs in axes_map.items():
        ymin = [col_params[c]['ymin'] for c in cs if col_params[c].get('ymin') is not None]
        ymin = np.min(ymin) if len(ymin) > 0 else None

        ymax = [col_params[c]['ymax'] for c in cs if col_params[c].get('ymax') is not None]
        ymax = np.min(ymax) if len(ymax) > 0 else None

        if ymin is not None and ymax is not None:
            continue

        df = None
        if isinstance(data, dict):
            if share:
                df = pd.concat([
                    d[[col for col in d.columns if col in cs]].add_suffix('__' + k) for k, d in data.items()
                ], axis=1).dropna(axis=1, how='all')
        elif isinstance(data, pd.DataFrame):
            df = data[[col for col in data.columns if col in cs]].dropna(axis=1, how='all')

        if df is None or df.shape == (0, 0):
            continue

        for c in cs:
            subcols = [col for col in df.columns if col.startswith(c)]
            if col_params[c].get('agg') is not None:
                df[subcols] = filter_agg(df[subcols], *col_params[c]['agg'])
            if col_params[c].get('smooth') is not None:
                df[subcols] = filter_smooth(df[subcols], *col_params[c]['smooth'])

        if disp == 'stacked':
            if isinstance(data, dict):
                for k in list(data):
                    subcols = [col for col in df.columns if col.endswith('__' + k)]
                    df[k] = pd.DataFrame(df[subcols].sum(axis=1))
                df = df[list(data)]
            else:
                df = pd.DataFrame(df.sum(axis=1))

        if ymin is None:
            ymin = df.min().min()
            if ymin > 0:
                ymin *= (1 - margin)
            else:
                ymin *= (1 + margin)

        if ymax is None:
            ymax = df.max().max()
            if ymax > 0:
                ymax *= (1 + margin)
            else:
                ymax *= (1 - margin)

        for c in cs:
            col_params[c]['ymin'] = ymin
            col_params[c]['ymax'] = ymax

    return col_params


def create_figure(ax=None, n=None, **kwargs):
    if ax is not None:
        return None, ax

    if n is None:
        return plt.subplots(
            figsize=kwargs.get('figsize'),
            subplot_kw={'projection': kwargs.get('projection')}
        )

    n_cols = kwargs.get('n_cols', n)
    n_rows = int(np.ceil(n / n_cols))

    figsize = kwargs.get('figsize')
    if figsize is not None:
        figsize = (figsize[0], figsize[1] * n_rows)

    plt.clf()
    plt.close('all')

    fig, axs = plt.subplots(
        nrows=n_rows,
        ncols=n_cols,
        sharex=kwargs.get('share', False),
        sharey=kwargs.get('share', False),
        figsize=figsize,
        constrained_layout=kwargs.get('constrained_layout', True),
        subplot_kw={'projection': kwargs.get('projection')}
    )

    if not is_array(axs):
        axs = np.asarray([axs])
    axs = axs.flatten()

    return fig, axs


def adjust_figure(fig=None, n=None, **kwargs):
    if fig is None:
        fig = plt.gcf()

    n = n or 1
    n_cols = kwargs.get('n_cols', n)
    n_rows = int(np.ceil(n / n_cols))
    n_axs = n_rows * n_cols

    fig_axs = fig.axes
    for ax in fig_axs[n:n_axs]:
        ax.set_axis_off()

    share = kwargs.get('share', False)
    if share:
        gr_axs = [
            fig_axs[:n]
        ] + [
            fig_axs[n_axs:][i * n:(i + 1) * n]
            for i in np.arange((len(fig_axs[n_axs:]) + n - 1) // n)
        ]

        for g in gr_axs:
            gr_rows = [
                g[i * n_cols:(i + 1) * n_cols]
                for i in np.arange((len(g) + n_cols - 1) // n_cols)
            ]

            for r in gr_rows:
                position = r[0].yaxis.get_ticks_position()

                for i, ax in enumerate(r):
                    if position == 'left' and i > 0:
                        ax.yaxis.set_tick_params(labelleft=False)
                    if position == 'right' and i < n_cols - 1:
                        ax.yaxis.set_tick_params(labelright=False)

    fig.tight_layout(**{
        k: v for k, v in kwargs.items() if k in ['pad', 'h_pad', 'w_pad', 'rect']
    })


def plot_figure(fig=None, show=True, path=None):
    if fig is not None:
        if show:
            plt.show()

        if path:
            save_figure(fig, path)

        fig.clear()

        plt.clf()
        plt.close('all')


def save_figure(fig=None, path=None):
    if fig is not None and path is not None:
        with tempfile.TemporaryFile() as tf:
            fig.savefig(tf, bbox_inches='tight')

            tf.seek(0)
            App.get_().fs.write_bytes(tf.read(), path)


def print_styler(dfs=None, margin=0.05, show=True, path=None):
    if dfs is not None:
        if show:
            try:
                display(HTML(dfs.to_html()))
            except Exception as e:
                print('Failed to show [dataframe_image]: {}'.format(e))
                return
        if path:
            save_styler(dfs, margin=margin, path=path)


def save_styler(dfs=None, margin=0.05, path=None):
    if dfs is not None and path is not None:
        try:
            import dataframe_image as dfi
            from PIL import Image

            with tempfile.TemporaryFile() as tf:
                ext = path.split('.')[-1]
                if ext == 'xlsx':
                    dfs.to_excel(tf)
                else:
                    dfi.export(dfs, tf, max_rows=-1)
                tf.seek(0)

                if margin and margin > 0:
                    img = Image.open(tf)

                    if margin < 1:
                        margin = int(min(img.width, img.height) * margin)

                    new_img = Image.new('RGB', (img.width + 2 * margin, img.height + 2 * margin), (255, 255, 255))
                    new_img.paste(img, (margin, margin))

                    with tempfile.TemporaryFile() as new_tf:
                        new_img.save(new_tf, format='png')
                        new_tf.seek(0)

                        App.get_().fs.write_bytes(new_tf.read(), path)
                else:
                    App.get_().fs.write_bytes(tf.read(), path)
        except ImportError as e:
            print('Failed to import [dataframe_image]: {}'.format(e))
            return
        except Exception as e:
            print('Failed to save [dataframe_image]: {}'.format(e))
            return


def set_styler(df, tpl_table=None, tpl_style=None):
    tpl_dir = 'pd-templates'
    tpl_table = tpl_table or '_html_table.tpl'
    tpl_style = tpl_style or '_html_style.tpl'

    if not App.get_().fs.exists('{}/{}'.format(tpl_dir, tpl_table)):
        tpl_table = None
    if not App.get_().fs.exists('{}/{}'.format(tpl_dir, tpl_style)):
        tpl_style = None

    dfs = df.style.from_custom_template(
        '{}/{}'.format(App.get_().fspath, tpl_dir),
        html_table=tpl_table,
        html_style=tpl_style
    )(df)

    return dfs


def set_log_scale(axis='y', ax=None):
    if ax is None:
        ax = plt.gca()

    if axis in ['x', 'both']:
        ax.set_xscale('log')
        ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    if axis in ['y', 'both']:
        ax.set_yscale('log')
        ax.yaxis.set_major_formatter(mticker.ScalarFormatter())

    return ax


def set_cat_locator(rotation=45, ha='right', axis='x', ax=None):
    if ax is None:
        ax = plt.gca()

    if axis in ['x', 'both']:
        for lbl in ax.get_xticklabels():
            lbl.set_rotation(rotation)
            lbl.set_horizontalalignment(ha)
            lbl.set_rotation_mode('anchor')

    return ax


def set_num_locator(n=None, axis='y', ax=None):
    if ax is None:
        ax = plt.gca()

    # x10 -> 1 tick for each multiple of 10 (default)
    # n10 -> 10 total ticks

    ltype, lval = None, None
    if is_number(n):
        lval = float(n)
    elif n is not None:
        ltype, lval = n[0], float(n[1:])

    if ltype == 'n':
        locator = mticker.MaxNLocator(nbins=int(lval))
    else:
        locator = mticker.MultipleLocator(base=lval)

    axs = []
    if axis in ['x', 'both']:
        axs.append(ax.xaxis)
    if axis in ['y', 'both']:
        axs.append(ax.yaxis)

    for a in axs:
        if ltype == 'n':
            if axis in ['x', 'both']:
                alim = ax.get_xlim()
                ticks = get_scale_range(alim[0], alim[1], int(lval))
                ax.set_xlim(ticks[0], ticks[-1])
            if axis in ['y', 'both']:
                alim = ax.get_ylim()
                ticks = get_scale_range(alim[0], alim[1], int(lval))
                ax.set_ylim(ticks[0], ticks[-1])

        a.set_major_locator(locator)

    return ax


def set_ts_locator(freq=None, axis='x', ax=None):
    if ax is None:
        ax = plt.gca()

    freq = Freq.parse(freq)
    dt_fmt = freq.str_label

    if freq == Freq.DAY:
        locator = mdates.DayLocator()
    elif freq == Freq.WEEK:
        locator = mdates.WeekdayLocator(byweekday=mdates.MO)
    elif freq == Freq.MONTH:
        locator = mdates.MonthLocator()
    elif freq == Freq.QUARTER:
        locator = mdates.MonthLocator(bymonth=(1, 4, 7, 10))
    elif freq == Freq.YEAR:
        locator = mdates.YearLocator()
    else:
        locator = mdates.AutoDateLocator()

    if axis in ['x', 'both']:
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(mdates.DateFormatter(fmt=dt_fmt))
        ax.tick_params(axis='x', which='major', reset=True, top=False)
        for lbl in ax.get_xticklabels():
            lbl.set_rotation(45)
    if axis in ['y', 'both']:
        ax.yaxis.set_major_locator(locator)
        ax.yaxis.set_major_formatter(mdates.DateFormatter(fmt=dt_fmt))
        ax.tick_params(axis='y', which='major', reset=True, top=False)
        for lbl in ax.get_yticklabels():
            lbl.set_rotation(45)

    return ax


def set_polar_locator(freq=None, ax=None):
    if ax is None:
        ax = plt.gca()
    
    freq = Freq.parse(freq)

    if freq == Freq.DAY:
        ticks = 365
        labels = list(np.arange(1, 366).astype(str))
    elif freq == Freq.WEEK:
        ticks = 7
        labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    elif freq == Freq.MONTH:
        ticks = 12
        labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    else:
        return ax

    theta = 2 * np.pi * np.linspace(0, 1, ticks, False)
    ax.set_xticks(theta)
    ax.set_xticklabels(labels)

    return ax


def set_null_locator(axis='y', ax=None):
    if ax is None:
        ax = plt.gca()

    locator = mticker.NullLocator()

    if axis in ['x', 'both']:
        ax.xaxis.set_major_locator(locator)
    if axis in ['y', 'both']:
        ax.yaxis.set_major_locator(locator)

    return ax


def set_title(title=None, x=None, y=None, ax=None, **kwargs):
    if title is None:
        return ax

    if ax is not None:
        ax.set_title(title, y=y, **kwargs)
    else:
        y = y or 1.0

        plt.suptitle(title, x=x, y=y, **kwargs)

    return ax


def set_note(note=None, x=None, y=None, ax=None, **kwargs):
    if note is None:
        return ax

    if ax is not None:
        x = x or 0
        y = y or -60

        ax.annotate(
            text=note, xytext=(x, y),
            xy=(0, 0), ha='left', va='top', linespacing=2, style='italic',
            xycoords='axes fraction', textcoords='offset points',
            **kwargs
        )
    else:
        x = x or 0.025
        y = y or -0.015

        plt.figtext(
            s=note, x=x, y=y,
            ha='left', va='top', linespacing=2, style='italic',
            **kwargs
        )

    return ax


# Tables

def get_df_styler(
    data: pd.DataFrame,
    meta: Optional[dict[str, str]] = None,
    freq: Optional[str] = None,
    max_dec: int = 2,
    names: dict[str, str] = {},
    header: list = [],
    gradients: list = [],
    bars: list = [],
    na_rep: str = '-',
    title: Optional[str] = None,
    note: Optional[str] = None,
    styles: Optional[list] = None,
    tpl_table: Optional[str] = None,
    tpl_style: Optional[str] = None
) -> Styler:
    df = to_pandas(data, pdtype='DataFrame')

    if freq is not None:
        freq = Freq.parse(freq)
    else:
        freq = Freq.infer(df, default=Freq.DAY)

    if isinstance(df.index, pd.DatetimeIndex):
        df.index = df.index.strftime(freq.str_label)

    if meta is None:
        meta = {col: 'num' if is_numeric(df[col]) else 'str' for col in df.columns}
    elif not isinstance(meta, dict):
        meta = {col: meta for col in df.columns}

    if len(names) == 0:
        names = {col: col for col in df.columns}

    fmt = {}
    for col, dtype in meta.items():
        if dtype == 'dtd':
            fmt[names[col]] = lambda x: x.strftime(freq.str_label)
        elif dtype == 'dts':
            fmt[names[col]] = lambda x: x.strftime('{} %H:%M'.format(freq.str_label))
        elif dtype == 'int':
            fmt[names[col]] = lambda x: format_number(x, 0)
        elif dtype == 'num':
            fmt[names[col]] = lambda x: format_number(x, max_dec)
        elif dtype == 'icur':
            fmt[names[col]] = lambda x: format_number(x, 0) + '€'
        elif dtype == 'ncur':
            fmt[names[col]] = lambda x: format_number(x, max_dec) + '€'
        elif dtype == 'ipct':
            fmt[names[col]] = lambda x: format_number(x, 0) + '%'
        elif dtype == 'npct':
            fmt[names[col]] = lambda x: format_number(x, max_dec) + '%'
        elif dtype == 'isig':
            fmt[names[col]] = lambda x: '+' + format_number(x, 0) if x > 0 else format_number(x, 0)
        elif dtype == 'nsig':
            fmt[names[col]] = lambda x: '+' + format_number(x, max_dec) if x > 0 else format_number(x, max_dec)

    if len(header) > 0 and isinstance(header[0], (list, tuple)):
        df.columns = pd.MultiIndex.from_product(header)
    elif len(names) > 0:
        df = df.rename(names, axis=1)

    if styles is None:
        styles = table_styles

    dfs = set_styler(
        df,
        tpl_table=tpl_table,
        tpl_style=tpl_style
    ).set_table_styles(styles).format(fmt, na_rep=na_rep)

    for grad in gradients:
        kwargs = dict(grad)

        cmap = kwargs['cmap']
        if not isinstance(cmap, mcolors.Colormap):
            if is_array(cmap):
                cmap = build_color_cat_map(cmap)
            elif '#' in cmap or cmap in colors:
                cmap = build_color_seq_map(get_color(cmap))

        kwargs['cmap'] = cmap
        kwargs['subset'] = [names[i] for i in kwargs['subset']]
        dfs = dfs.background_gradient(**kwargs).highlight_null(get_color('grey-alpha'))

    for bar in bars:
        kwargs = dict(bar)

        if isinstance(kwargs['color'], (list, tuple)):
            kwargs['color'] = [get_color(c) for c in kwargs['color']]
        else:
            kwargs['color'] = get_color(kwargs['color'])

        kwargs['subset'] = [names[i] for i in kwargs['subset']]
        dfs = dfs.bar(**kwargs)

    if title is not None:
        dfs = dfs.set_caption(title)

    if note is not None:
        dfs = dfs.set_table_attributes(note)

    return dfs


def print_df(data, margin=0.05, show=True, path=None, **kwargs):
    dfs = get_df_styler(data, **kwargs)

    print_styler(dfs=dfs, margin=margin, show=show, path=path)


# Information


def show_summary(x, tgt=None, name=None):
    line = []

    if name is not None:
        line.append('{}'.format(name))

    line.append('Shape: {:,} x {:,}'.format(*x.shape))

    print(' | '.join(line))

    if tgt is not None:
        if isinstance(tgt, str):
            tgt = x[tgt]

        values = tgt.value_counts(dropna=False).sort_values(ascending=False)

        for v, t in values.items():
            r = 100 * t / values.sum()
            print('    {}: {:,} ({:0.2f}%)'.format(v, t, r))


def show_memory(x):
    dtypes = []
    for dtype in x.dtypes:
        if dtype.name not in dtypes:
            dtypes.append(dtype.name)

    for dtype in dtypes:
        sel = x.select_dtypes(include=[dtype])
        mean_usage_b = sel.memory_usage(deep=True).mean()
        mean_usage_mb = mean_usage_b / 1024 ** 2
        print('{}: {:03.2f}MB'.format(dtype, mean_usage_mb))

    print()
    x.info(memory_usage='deep')


def show_nulls(x, detail=False):
    m = x.isnull()
    c = m.any()

    n = len(c[c].index)
    print('Nulls: {:d}'.format(n))

    if n > 0:
        if detail:
            d = pd.Series({col: m[col].value_counts()[True] for col in c[c].index}).sort_values(ascending=False)
        else:
            d = list(c[c].index)

        print(d)


def show_duplicates(x, columns=None, show=False):
    n = x[x.duplicated(columns, keep='first')].shape[0]
    print('Duplicates: {:d}'.format(n))

    if show:
        d = x[x.duplicated(columns, keep=False)].set_index(columns)
        print(d)


def show_cv_results(cv, params=None, n_top=3):
    params = params or []

    best = {p: cv.best_params_[p] for p in list(params)}

    rank = cv.cv_results_['rank_test_score']
    mean = cv.cv_results_['mean_test_score']
    std = cv.cv_results_['std_test_score']
    sets = cv.cv_results_['params']

    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(rank == i)
        for k in candidates:
            print('{:d}: [Mean {:0.4f} Std {:0.4f}]'.format(i, mean[k], std[k]))
            print({p: sets[k][p] for p in list(params)})
            print()

    return best


# Plots


def plot_series(data, ax=None, show=True, path=None, **kwargs):
    _col_params = dict(
        kind='line', yaxis='left', agg=None, smooth=None,
        ylim=None, ymin=None, ymax=None, yticks=None, yfmt=None, ylog=False,
        cm=None, label=True, annot=False, annot_zero=True, bottom=None,
        marker=None, s=50, pw=1, lw=3, ls='-', alpha=None
    )
    _plt_params = dict(
        xlim=None, xmin=None, xmax=None, xticks=None, xfmt=None, xlog=False,
        dt_min=None, dt_max=None, freq='W',
        grid=True, disp=None, vlines=None, hlines=None,
        margin=0.05, legend=True, leg_cols=None, leg_loc=None, title=None, note=None
    )
    _fig_params = dict(
        figsize=(20, 10), projection=None
    )

    df = to_pandas(data, pdtype='DataFrame')

    series = list(df.columns)
    n_series = len(series)

    col_params = get_params(_col_params, series, 'col_params', **kwargs)
    plt_params = get_params(_plt_params, None, 'plt_params', **kwargs)
    fig_params = get_params(_fig_params, None, 'fig_params', **kwargs)

    df = ts_range(df, dt_min=plt_params['dt_min'], dt_max=plt_params['dt_max'])
    idx = df.index
    is_ts = is_timeseries(idx)
    is_num = is_numeric(idx)

    theta, width = None, None
    is_polar = fig_params['projection'] == 'polar'
    if is_polar:
        df = df.fillna(0)
        theta, width = np.linspace(0.0, 2 * np.pi, df.shape[0], endpoint=False, retstep=True)

    col_params = parse_col_params(
        col_params,
        df,
        disp=plt_params['disp'],
        margin=plt_params['margin']
    )

    fig, ax = create_figure(ax=ax, **fig_params)

    axes = {ax: series}
    if n_series > 1:
        yaxis_arr = np.unique([p['yaxis'] for p in col_params.values()])
        if len(yaxis_arr) > 1:
            yaxis = col_params[series[0]]['yaxis']
            axes[ax] = [s for s in series if col_params[s]['yaxis'] == yaxis]
            axes[ax.twinx()] = [s for s in series if col_params[s]['yaxis'] != yaxis]

    for axis, keys in axes.items():
        gs = []

        for s in keys:
            g = None
            cp = col_params[s]
            kind = cp['kind']

            if cp['agg'] is not None:
                df[s] = filter_agg(df[s], *cp['agg'])

            if cp['smooth'] is not None:
                df[s] = filter_smooth(df[s], *cp['smooth'])

            if kind in ['line', 'area']:
                idx_ = theta if is_polar else np.array(idx)

                if cp['marker'] or df[s].dropna().shape[0] < 2:
                    if not isinstance(cp['marker'], str):
                        cp['marker'] = '.'
                    cp['s'] = 3 * cp['lw']

                b = 0.0
                x = df[s].values.astype(float)
                if plt_params['disp'] == 'stacked':
                    b = df[keys].apply(lambda x: x.cumsum().shift(), axis=1).fillna(0)[s].values.astype(float)
                    x += b
                elif isinstance(cp['bottom'], str) and cp['bottom'] in series:
                    b = df[cp['bottom']].fillna(0).values.astype(float)
                elif is_number(cp['bottom']):
                    b = float(cp['bottom'])

                if kind == 'area':
                    if cp['alpha'] is None:
                        cp['alpha'] = 0.8

                    axis.fill_between(
                        idx_, b, x,
                        lw=0,
                        alpha=cp['alpha'], color=cp['cm']
                    )

                if is_polar:
                    idx_ = np.concatenate((theta, theta[:1]))
                    x = np.concatenate((x, x[:1]))

                g, = axis.plot(
                    idx_, x,
                    color=cp['cm'], lw=cp['lw'], ls=cp['ls'], alpha=cp['alpha'], label=cp['label'],
                    marker=cp['marker'], markersize=cp['s']
                )
            elif kind in ['bar', 'step', 'top', 'scatter']:
                idx_ = theta if is_polar else idx
                if width is None:
                    width = 1 if kind == 'step' else 0.8
                    if is_ts:
                        dfreq = Freq.infer(df[s], default=Freq.DAY)
                        if dfreq == Freq.WEEK:
                            width = 7 if kind == 'step' else 6
                        elif dfreq == Freq.MONTH:
                            width = 31 if kind == 'step' else 24
                        elif dfreq == Freq.QUARTER:
                            width = 93 if kind == 'step' else 75
                        elif dfreq == Freq.YEAR:
                            width = 365 if kind == 'step' else 300
                    width *= cp['pw']

                b = 0.0
                x = df[s].values.astype(float)
                if plt_params['disp'] == 'stacked':
                    b = df[keys].apply(lambda x: x.cumsum().shift(), axis=1).fillna(0)[s].values.astype(float)
                    if kind == 'top':
                        x += b
                elif isinstance(cp['bottom'], str) and cp['bottom'] in series:
                    b = df[cp['bottom']].fillna(0).values.astype(float)
                elif is_number(cp['bottom']):
                    b = float(cp['bottom'])

                if plt_params['disp'] == 'split':
                    width_ = width / n_series
                    idx_ += (series.index(s) - n_series / 2) * width_ + width_ / 2
                else:
                    width_ = width

                idx_ = idx_[~np.isnan(x)]
                x = x[~np.isnan(x)]

                if kind == 'top':
                    if cp['marker']:
                        if not isinstance(cp['marker'], str):
                            cp['marker'] = '.'
                        cp['s'] = 4 * cp['lw']

                    g = axis.errorbar(
                        idx_, x, xerr=(width_ / 2),
                        color=cp['cm'], elinewidth=cp['lw'], lw=0, alpha=cp['alpha'], label=cp['label'],
                        marker=cp['marker'], markersize=cp['s']
                    )
                elif kind == 'scatter':
                    if not isinstance(cp['marker'], str):
                        cp['marker'] = '.'

                    g = axis.scatter(
                        idx_, x,
                        color=cp['cm'], alpha=cp['alpha'], label=cp['label'],
                        marker=cp['marker'], s=cp['s']
                    )
                else:
                    g = axis.bar(
                        idx_, x, bottom=b, width=width_,
                        color=cp['cm'], alpha=cp['alpha'], label=cp['label']
                    )

                annot = cp['annot']
                if not is_empty(annot):
                    if is_array(annot):
                        d_annot = to_pandas(annot, pdtype='DataFrame')[s].fillna('').values
                    else:
                        d_annot = x

                    d_annot = np.array([format_number(v) if cp['annot_zero'] or v > 0 else '' for v in d_annot])
                    if cp['yfmt'] is not None:
                        d_annot = np.array([cp['yfmt'].format(v) if v != '' else '' for v in d_annot])

                    if is_polar:
                        rotation = (2 * np.pi / len(idx)) * np.arange(len(idx))
                        rotation_degrees = -np.rad2deg(rotation)

                        for i, (t, value, rotation) in enumerate(zip(theta, x, rotation_degrees)):
                            axis.text(
                                t, value, d_annot[i],
                                ha='center', color='k', bbox=dict(
                                    edgecolor='k', facecolor='w',
                                    boxstyle='round, pad=0.4', lw=1
                                )
                            )
                    else:
                        axis.bar_label(g, labels=d_annot, padding=3)

            if g is not None:
                gs.append(g)

            axis.set_ymargin(0)
            if cp['yaxis'] == 'right':
                axis.yaxis.tick_right()

            if cp['ylog']:
                set_log_scale(axis='y', ax=axis)

        s = keys[0]
        cp = col_params[s]

        ymin = cp['ymin']
        ymax = cp['ymax']
        ylim = list(axis.get_ylim())

        if ymin is not None:
            ylim[0] = ymin
        if ymax is not None:
            ylim[1] = ymax

        axis.set_ylim(ylim)

        yticks = cp['yticks']
        yfmt = cp['yfmt']

        if not plt_params['grid']:
            axis.set_yticks([])
        elif isinstance(yticks, dict) and s in yticks.keys():
            axis.set_yticks(yticks[s])
        elif isinstance(yticks, (list, tuple)):
            axis.set_yticks(yticks)
        elif yticks is not None:
            set_num_locator(n=yticks, ax=axis)

        if is_polar:
            ax.set_yticklabels([])
        else:
            yticklabels = [format_number(t) for t in axis.get_yticks()]
            if yfmt is not None:
                yticklabels = [yfmt.format(t) for t in yticklabels]
            axis.set_yticklabels(yticklabels)
        
        if len(axis.get_yticks()) == 0:
            axis.spines[cp['yaxis']].set_visible(False)
            axis.yaxis.set_tick_params(which='both', length=0)

        if plt_params['legend'] and n_series > 1:
            ncol = plt_params['leg_cols'] or int(np.ceil(len(gs) / 5))
            loc = plt_params['leg_loc'] or 'upper {}'.format(cp['yaxis'])
            bbox_to_anchor = (1.05, 1) if is_polar else None
            axis.legend(handles=gs, loc=loc, ncol=ncol, bbox_to_anchor=bbox_to_anchor)

    vlines = plt_params['vlines'] or []
    for line in vlines:
        if is_ts:
            line = pd.to_datetime(line)
        ax.axvline(line, color=get_color('grey-light'), ls='--', lw=2, alpha=0.5)

    hlines = plt_params['hlines'] or []
    for line in hlines:
        ax.axhline(line, color=get_color('grey-light'), ls='--', lw=2, alpha=0.5)

    if is_polar:
        ax.tick_params(labelbottom=False)
        ax.set_xticks(theta)
        ax.set_xticklabels([])

        ax.set_rorigin(-5.0)
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)

        rotation = (2 * np.pi / len(idx)) * np.arange(len(idx))
        mask_flip_label = (rotation > np.pi / 2) & (rotation < np.pi / 2 * 3)
        rotation[mask_flip_label] = rotation[mask_flip_label] + np.pi
        rotation_degrees = -np.rad2deg(rotation)

        for t, rotation, label in zip(theta, rotation_degrees, idx):
            ax.text(
                t, 110, label,
                rotation=rotation, rotation_mode='anchor', ha='center'
            )

        if plt_params['grid']:
            ax.set_thetagrids((theta + width / 2) * 180 / np.pi)

            for gl in list(ax.yaxis.get_gridlines())[:-1]:
                gl.set(ls='--', lw=.5, color='k', alpha=0.2)
            for gl in ax.xaxis.get_gridlines():
                gl.set(ls='-', lw=1, color='k', alpha=1)
        else:
            ax.grid(False)
    else:
        ax.set_xticks(idx)
        if is_ts:
            set_ts_locator(freq=plt_params['freq'], axis='x', ax=ax)
        elif is_num:
            if plt_params['xlog']:
                set_log_scale(axis='x', ax=ax)
            if plt_params['xticks'] is not None:
                if isinstance(plt_params['xticks'], (list, tuple)):
                    ax.set_xticks(plt_params['xticks'])
                else:
                    set_num_locator(n=plt_params['xticks'], axis='x', ax=ax)
                ax.set_xmargin(plt_params['margin'])
            if plt_params['xfmt'] is not None:
                ax.set_xticklabels([plt_params['xfmt'].format(format_number(t)) for t in ax.get_xticks()])
        else:
            set_cat_locator(axis='x', ax=ax)

        if plt_params['grid']:
            gaxis = plt_params['grid'] if plt_params['grid'] in ['x', 'y'] else 'both'
            ax.grid(True, axis=gaxis, ls='--', lw=.5, color='k', alpha=0.2)
        else:
            ax.grid(False)

    set_title(plt_params['title'], ax=ax)
    set_note(plt_params['note'], ax=ax)

    plot_figure(show=show, path=path, fig=fig)


def plot_series_ci(data, ax=None, show=True, path=None, **kwargs):
    _col_params = dict(
        agg=None, smooth=None, log=False, ylim=None, ymin=None, ymax=None,
        yticks=None, fmt=None, cm=None, marker=False, lw=3, ls='-', alpha=1
    )
    _plt_params = dict(
        dt_min=None, dt_max=None, freq='week', grid=True, vlines=None, hlines=None,
        margin=0.05, legend=True, leg_cols=None, title=None, note=None
    )
    _fig_params = dict(
        figsize=(20, 10)
    )

    df = to_pandas(data, pdtype='DataFrame')
    series = list(df.columns)

    col_main, col_min, col_max = series

    col_params = get_params(_col_params, series, 'col_params', **kwargs)
    plt_params = get_params(_plt_params, None, 'plt_params', **kwargs)
    fig_params = get_params(_fig_params, None, 'fig_params', **kwargs)

    df = ts_range(df, dt_min=plt_params['dt_min'], dt_max=plt_params['dt_max'])
    idx = df.index

    col_params = parse_col_params(
        col_params,
        df,
        margin=plt_params['margin']
    )

    ymin = col_params[col_main]['ymin']
    ymax = col_params[col_main]['ymax']

    ylim = [df.min().min(), df.max().max()]
    if ymin is not None:
        ylim[0] = ymin
    if ymax is not None:
        ylim[1] = ymax
    m = np.abs(ylim[1]) * plt_params['margin']
    if np.abs(ylim[0]) > 0:
        ylim[0] -= m
    if np.abs(ylim[1]) > 0:
        ylim[1] += m

    col_params[col_main]['ymin'] = ylim[0]
    col_params[col_main]['ymax'] = ylim[1]

    fig, ax = create_figure(ax=ax, **fig_params)

    plot_series(
        df[col_main],
        ax=ax,
        col_params=dict(col_params[col_main]),
        plt_params=dict(plt_params),
        fig_params=dict(fig_params)
    )

    agg = col_params[col_main]['agg']
    if agg is not None:
        if not isinstance(agg, (list, tuple)):
            agg = [agg]
        df = filter_agg(df, *agg)

    smooth = col_params[col_main]['smooth']
    if smooth is not None:
        if isinstance(smooth, bool):
            smooth = [int(smooth)]
        elif not isinstance(smooth, (list, tuple)):
            smooth = [smooth]
        df = filter_smooth(df, *smooth)

    ax.fill_between(
        idx,
        df[col_min].values,
        df[col_max].values,
        color=plt.gca().lines[-1].get_color(),
        alpha=0.2
    )

    plot_figure(show=show, path=path, fig=fig)


def plot_series_cmp(data, ax=None, show=True, path=None, **kwargs):
    _col_params = dict(
        agg=None, smooth=None, log=False, ylim=None, ymin=None, ymax=None,
        yticks=None, fmt=None, marker=None, lw=3, ls='-'
    )
    _plt_params = dict(
        cur_col=None, agg_col=None, cmp_cols=None, cmp_map=False,
        dt_min=None, dt_max=None, freq='week', grid=True, vlines=None, hlines=None,
        margin=0.05, legend=True, leg_cols=None, title=None, note=None
    )
    _fig_params = dict(
        figsize=(20, 10)
    )

    df = to_pandas(data, pdtype='DataFrame')
    series = list(df.columns)

    col_params = get_params(_col_params, series, 'col_params', **kwargs)
    plt_params = get_params(_plt_params, None, 'plt_params', **kwargs)
    fig_params = get_params(_fig_params, None, 'fig_params', **kwargs)

    agg_col = plt_params.get('agg_col')
    if agg_col is not None:
        col_params[agg_col] |= dict(
            cm='blue'
        )
        series = [col for col in series if col != agg_col]

    cur_col = plt_params.get('cur_col')
    if cur_col is not None:
        col_params[cur_col] |= dict(
            cm='grey'
        )
        series = [col for col in series if col != cur_col]

    cmp_cols = plt_params.get('cmp_cols') or series
    cmp_map = plt_params.get('cmp_map') or False

    if len(cmp_cols) > 0:
        pal = palette + ['{}-light'.format(c) for c in palette]
        cmap = get_color_map(pal[:len(cmp_cols)][::-1], cmp_cols)
        alpha = None if cur_col is None and agg_col is None else 0.4 if cmp_map else 0.2

        for col in cmp_cols:
            col_params[col] |= dict(
                cm=cmap[col] if plt_params['cmp_map'] else 'blue',
                lw=2,
                alpha=alpha,
                label=cmp_map
            )

    plot_series(
        df,
        ax=ax,
        show=show,
        path=path,
        col_params=dict(col_params),
        plt_params=dict(plt_params),
        fig_params=dict(fig_params)
    )


def plot_series_polar(data,
    ymax=None, yticks=None, grid=True,
    dt_min=None, dt_max=None, freq='month', fmt=None, cm=None,
    figsize=(10, 10), legend=True, title=None, note=None, ax=None, show=True, path=None
):
    df = to_pandas(data, pdtype='DataFrame')
    df = ts_range(df, dt_min=dt_min, dt_max=dt_max)

    cols = list(df.columns)
    n_cols = len(cols)
    idx = df.index
    is_ts = isinstance(idx, pd.DatetimeIndex)
    theta = 2 * np.pi * np.linspace(0, 1, df.shape[0], False)

    cm = get_color_map(cm=cm, keys=cols)

    fig, ax = create_figure(ax=ax, figsize=figsize, projection='polar')

    ax.set_theta_direction(-1)
    ax.set_theta_zero_location('N')
    ax.set_rlabel_position(90)

    series = []
    for col in cols:
        s, = ax.plot(theta, df[col].values, color=cm[col], lw=3, label=col)

        series.append(s)

        if ymax is not None:
            ax.set_rmax(ymax)

        if not grid:
            yticks = []
        if yticks is not None:
            if isinstance(yticks, (list, tuple)):
                ax.set_rticks(yticks)
            else:
                set_num_locator(n=yticks, ax=ax)

        yticklabels = [format_number(t) for t in ax.get_yticks()]
        if fmt is not None:
            yticklabels = [fmt.format(t) for t in yticklabels]
        ax.set_yticklabels(yticklabels)

    ax.set_xticks(theta)
    if is_ts:
        set_polar_locator(freq=freq, ax=ax)

    if grid:
        ax.grid(True, axis='both', ls='--', lw=.5, color='k', alpha=0.2)
    else:
        ax.grid(False)

    if legend and n_cols > 1:
        ncol = int(np.ceil(n_cols / 5))
        ax.legend(handles=series, loc='upper left', bbox_to_anchor=(1.05, 1), ncol=ncol)

    set_title(title, y=1.08, ax=ax)
    set_note(note, ax=ax)

    plot_figure(show=show, path=path, fig=fig)


def plot_histogram(data, ax=None, show=True, path=None, **kwargs):
    # stat -> count (default), probability
    # outliers -> keep (default), group or remove
    _plt_params = dict(
        stat='count', bins=None, weights=None, discrete=False, outliers='keep', ol_std=3,
        show_bars=True, show_kde=False, show_mean=False, show_std=False, vlines=None, hlines=None,
        xlim=None, xmin=None, xmax=None, xticks=None, ylim=None, ymin=None, ymax=None, yticks=None,
        grid=True, cm=None, legend=True, title=None, note=None
    )
    _fig_params = dict(
        figsize=(20, 10)
    )

    ds = to_pandas(data, pdtype='Series').dropna()

    plt_params = get_params(_plt_params, None, 'plt_params', **kwargs)
    fig_params = get_params(_fig_params, None, 'fig_params', **kwargs)

    if plt_params['weights'] is not None:
        weights = np.array(plt_params['weights'])
        dw = DescrStatsW(ds, weights=weights)
        mean, std = dw.mean, dw.std
    else:
        weights = None
        mean, std = ds.agg(['mean', 'std'])

    if plt_params['stat'] in ['prob', 'probability', 'density']:
        plt_params['stat'] = 'prob'

    hist, _ = get_histogram(
        ds,
        metric=plt_params['stat'],
        bins=plt_params['bins'],
        weights=weights,
        discrete=plt_params['discrete'],
        outliers=plt_params['outliers'],
        ol_std=plt_params['ol_std']
    )
    if plt_params['stat'] == 'prob':
        hist *= 100

    width = np.mean(np.diff(hist.index)) * 0.9
    cm = get_color_map(plt_params['cm'], 1)[0]

    fig, ax = create_figure(ax=ax, **fig_params)

    if plt_params['show_bars']:
        ax.bar(
            hist.index,
            hist.values,
            width=width,
            color=cm,
            alpha=0.6 if plt_params['show_kde'] else 1
        )

    if plt_params['show_kde']:
        p = np.linspace(hist.min(), hist.max(), 100)
        ax2 = ax.twinx()

        plot_kde(
            ds,
            p=p,
            fill=False,
            cm=cm,
            lw=2,
            weights=weights,
            outliers=plt_params['outliers'],
            ol_std=plt_params['ol_std'],
            ax=ax2
        )

        ax2.set_yticks([])
        ax2.set_yticklabels([])

    if plt_params['show_mean']:
        ax.axvline(
            mean,
            color=get_color('green-light'),
            ls='--',
            lw=2,
            alpha=0.5,
            label='Mean ({})'.format(format_number(mean))
        )

    if plt_params['show_std']:
        ax.axvline(
            mean + std,
            color=get_color('red-light'),
            ls='--',
            lw=2,
            alpha=0.5,
            label='Std ({})'.format(format_number(std))
        )

    vlines = plt_params['vlines'] or []
    for line in vlines:
        ax.axvline(line, color=get_color('grey-light'), ls='--', lw=2, alpha=0.5)

    hlines = plt_params['hlines'] or []
    for line in hlines:
        ax.axhline(line, color=get_color('grey-light'), ls='--', lw=2, alpha=0.5)

    if plt_params['xlim'] is not None:
        plt_params['xmin'], plt_params['xmax'] = plt_params['xlim']

    xmin = plt_params['xmin']
    xmax = plt_params['xmax']
    xlim = list(ax.get_xlim())

    if xmin is not None:
        xlim[0] = xmin
    if xmax is not None:
        xlim[1] = xmax

    if plt_params['ylim'] is not None:
        plt_params['ymin'], plt_params['ymax'] = plt_params['ylim']

    ymin = plt_params['ymin']
    ymax = plt_params['ymax']
    ylim = list(ax.get_ylim())

    if ymin is not None:
        ylim[0] = ymin
    else:
        ylim[0] = 0
    if ymax is not None:
        ylim[1] = ymax
    else:
        ylim[1] = hist.max() * 1.1

    if plt_params['stat'] == 'prob':
        if ylim[0] < 0:
            ylim[0] = 0
        if ylim[1] > 100:
            ylim[1] = 100

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    xticks = plt_params['xticks']
    yticks = plt_params['yticks']

    if not plt_params['grid']:
        xticks = []
        yticks = []

    if xticks is not None:
        if isinstance(xticks, (list, tuple)):
            ax.set_xticks(xticks)
        else:
            set_num_locator(n=xticks, axis='x', ax=ax)

    if plt_params['discrete'] and plt_params['bins'] is None:
        ax.set_xticks([i for i in ax.get_xticks() if i in hist.index])

    if yticks is not None:
        if isinstance(yticks, (list, tuple)):
            ax.set_yticks(yticks)
        else:
            set_num_locator(n=yticks, axis='y', ax=ax)

    ax.set_xlabel(None)
    ax.set_ylabel(None)

    xticklabels = ['{}'.format(format_number(i)) for i in ax.get_xticks()]
    if plt_params['discrete'] and plt_params['bins'] is None and plt_params['outliers'] == 'group':
        xticklabels[-1] = xticklabels[-1] + '+'
    ax.set_xticklabels(xticklabels)

    if plt_params['stat'] == 'prob':
        yticklabels = ['{}%'.format(format_number(i)) for i in ax.get_yticks()]
    else:
        yticklabels = ['{}'.format(format_number(i)) for i in ax.get_yticks()]
    ax.set_yticklabels(yticklabels)

    if plt_params['grid']:
        gaxis = plt_params['grid'] if plt_params['grid'] in ['x', 'y'] else 'both'
        ax.grid(True, axis=gaxis, ls='--', lw=.5, color='k', alpha=0.2)
    else:
        ax.grid(False)

    if plt_params['legend'] and plt_params['show_mean']:
        ax.legend(loc='upper right')

    set_title(plt_params['title'], ax=ax)
    set_note(plt_params['note'], ax=ax)

    plot_figure(show=show, path=path, fig=fig)


def plot_kde(x, p=100, ax=None, show=True, path=None, **kwargs):
    _plt_params = dict(
        weights=None, kernel='gaussian', bw='scott', fill=True, outliers='keep', ol_std=3,
        xlim=None, xmin=None, xmax=None, xticks=None, xfmt=None, xlog=False,
        ylim=None, ymin=None, ymax=None, yticks=None, yfmt=None, ylog=False,
        grid=True, cm=None, lw=3, ls='-', alpha=None, title=None, note=None
    )
    _fig_params = dict(
        figsize=(15, 9)
    )

    x = np.array(x)
    nobs, ndim = array_shape(x)

    plt_params = get_params(_plt_params, None, 'plt_params', **kwargs)
    fig_params = get_params(_fig_params, None, 'fig_params', **kwargs)

    if plt_params['weights'] is not None:
        weights = np.array(plt_params['weights'])
    else:
        weights = None

    fig, ax = create_figure(ax=ax, **fig_params)

    if ndim > 1:
        px = np.linspace(x[:, 0].min(), x[:, 0].max(), 100)
        py = np.linspace(x[:, 1].min(), x[:, 1].max(), 100)
        p = np.column_stack([px, py])
    else:
        p = np.linspace(x.min(), x.max(), 100)
        px, py = None, None

    kde = KernelDensityEstimator(
        x,
        weights=weights,
        kernel=plt_params['kernel'],
        bw=plt_params['bw'],
        outliers=plt_params['outliers'],
        n_dev=plt_params['ol_std']
    ).fit(p)

    cm = plt_params['cm'] or 'blue'
    if ndim > 1:
        cmap = cm
        if not isinstance(cmap, mcolors.Colormap):
            if is_array(cmap):
                cmap = build_color_cat_map(cmap)
            elif '#' in cmap or cmap in colors:
                cmap = build_color_seq_map(get_color(cmap))
    else:
        cmap = get_color(cm)

    if ndim > 1:
        if plt_params['fill']:
            ax.contourf(
                px, py, kde,
                cmap=cmap,
                alpha=plt_params['alpha']
            )
        else:
            ax.contour(
                px, py, kde,
                cmap=cmap,
                alpha=plt_params['alpha']
            )
    else:
        ax.plot(
            p, kde,
            color=cmap,
            lw=plt_params['lw'],
            ls=plt_params['ls']
        )
        if plt_params['fill']:
            ax.fill_between(
                p, 0, kde,
                color=cmap,
                lw=0,
                alpha=0.2
            )

    if plt_params['xlim'] is not None:
        plt_params['xmin'], plt_params['xmax'] = plt_params['xlim']

    xmin = plt_params['xmin']
    xmax = plt_params['xmax']
    xlim = list(ax.get_xlim())

    if xmin is not None:
        xlim[0] = xmin
    if xmax is not None:
        xlim[1] = xmax

    if plt_params['ylim'] is not None:
        plt_params['ymin'], plt_params['ymax'] = plt_params['ylim']

    ymin = plt_params['ymin']
    ymax = plt_params['ymax']
    ylim = list(ax.get_ylim())

    if ymin is not None:
        ylim[0] = np.max([ymin, 0])
    else:
        ylim[0] = 0
    if ymax is not None:
        ylim[1] = np.min([ymax, 1])
    elif ndim == 1:
        ylim[1] = kde.max() * 1.1

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    if ndim > 1 and plt_params['xlog'] and plt_params['ylog']:
        set_log_scale(axis='both', ax=ax)
    elif plt_params['xlog']:
        set_log_scale(axis='x', ax=ax)
    elif ndim > 1 and plt_params['ylog']:
        set_log_scale(axis='y', ax=ax)

    if plt_params['xticks'] is not None:
        if isinstance(plt_params['xticks'], (list, tuple)):
            ax.set_xticks(plt_params['xticks'])
        else:
            set_num_locator(n=plt_params['xticks'], axis='x', ax=ax)

    if (ndim == 1 and is_numeric(x)) or (ndim > 1 and is_numeric(x[:, 0])):
        if plt_params['xfmt'] is not None:
            ax.set_xticklabels([plt_params['xfmt'].format(format_number(t)) for t in ax.get_xticks()])
    else:
        set_cat_locator(axis='x', ax=ax)

    if ndim > 1 and plt_params['yticks'] is not None:
        if isinstance(plt_params['yticks'], (list, tuple)):
            ax.set_yticks(plt_params['yticks'])
        else:
            set_num_locator(n=plt_params['yticks'], axis='y', ax=ax)

    if ndim > 1 and is_numeric(x[:, 1]):
        if plt_params['yfmt'] is not None:
            ax.set_yticklabels([plt_params['yfmt'].format(format_number(t)) for t in ax.get_yticks()])
    else:
        set_cat_locator(axis='y', ax=ax)

    set_title(plt_params['title'], ax=ax)
    set_note(plt_params['note'], ax=ax)

    plot_figure(show=show, path=path, fig=fig)


def plot_kde_1d(data, p=1000, weights=None, ax=None, show=True, path=None, **kwargs):
    _plt_params = dict(
        kernel='gaussian', bw='scott', fill=True, outliers='keep', ol_std=3,
        xlim=None, xmin=None, xmax=None, xticks=None, xfmt=None,
        ylim=None, ymin=None, ymax=None, yticks=None, yfmt=None,
        show_mean=False, show_median=False, show_std=False, show_ci=False, show_ridge=False,
        std_n=1, ci_alpha=0.05, ridge_share=False, stats_dec=2, cm=None, lw=3, ls='-', alpha=1,
        grid=True, vlines=None, hlines=None, legend=True, title=None, note=None
    )
    _fig_params = dict(
        figsize=None
    )

    df = to_pandas(data, pdtype='DataFrame')

    items = list(df.columns)
    n_items = len(items)
    n_rows = df.shape[0]

    if weights is not None:
        weights = np.array(weights) * n_rows / np.sum(weights)
    else:
        weights = np.ones(n_rows)

    plt_params = get_params(_plt_params, None, 'plt_params', **kwargs)
    fig_params = get_params(_fig_params, None, 'fig_params', **kwargs)

    cm = get_color_map(plt_params['cm'], items)

    if plt_params['show_ridge']:
        rscale = float(1 / n_items) * 0.95
        rpos = dict(zip(items, np.linspace(0, 1, n_items, endpoint=False)[::-1]))
    else:
        rscale = 1.0
        rpos = dict(zip(items, np.repeat(0, n_items)))

    if fig_params['figsize'] is None:
        if plt_params['show_ridge']:
            fig_params['figsize'] = (20, int(n_items * 3))
        else:
            fig_params['figsize'] = (20, 10)

    fig, ax = create_figure(ax=ax, **fig_params)

    for s in items:
        samp = Stat(
            df[s].values,
            weights=weights,
            dropna=True,
            outliers=plt_params['outliers'],
            n_dev=plt_params['ol_std']
        )

        x = samp.data
        w = samp.weights
        ix = np.linspace(x.min(), x.max(), p)
        b = rpos[s]

        lines = []
        labels = [[s], [], []]
        if plt_params['show_mean']:
            mean = samp.mean()
            lines.append(mean)
            labels[1].append('Mean: {}'.format(
                format_number(mean, plt_params['stats_dec'])
            ))
        if plt_params['show_median']:
            median = samp.median()
            lines.append(median)
            labels[1].append('Median: {}'.format(
                format_number(median, plt_params['stats_dec'])
            ))
        if plt_params['show_std']:
            mean = samp.mean()
            std = samp.std() * plt_params['std_n']
            lines.append(mean - std)
            lines.append(mean + std)
            labels[1].append('Std: {}'.format(
                format_number(std, plt_params['stats_dec'])
            ))
        if plt_params['show_ci']:
            cmin, cmax = samp.ci(alpha=plt_params['ci_alpha'])
            lines.append(cmin)
            lines.append(cmax)
            labels[2].append('CI ({}%): {} - {}'.format(
                format_number(100 * (1 - plt_params['ci_alpha'])),
                format_number(cmin, plt_params['stats_dec']),
                format_number(cmax, plt_params['stats_dec'])
            ))

        for line in lines:
            if ix.min() <= line <= ix.max():
                ix = np.sort(np.append(ix, line))

        label = '\n'.join([' | '.join(i) for i in labels if len(i) > 0]).strip()

        d = KernelDensityEstimator(
            x,
            weights=w,
            kernel=plt_params['kernel'],
            bw=plt_params['bw']
        ).fit(ix).values

        if plt_params['show_ridge']:
            if plt_params['ridge_share']:
                d = d * rscale * 0.95 + b
            else:
                d = d * rscale * 0.95 / d.max() + b

        edges = sorted(np.append(lines, [x.min(), x.max()]))
        alphas = np.linspace(plt_params['alpha'] * 0.2, plt_params['alpha'] * 0.4,
            int(np.max([np.ceil((len(edges) - 1) / 2), 1])))
        if len(alphas) < len(edges) - 1:
            if len(edges) % 2 > 0:
                alphas = np.append(alphas, alphas[::-1])
            else:
                alphas = np.append(alphas, alphas[::-1][1:])

        ec = cm[s]
        fc = ec if plt_params['fill'] else 'none'

        for i in np.arange(len(edges) - 1):
            find = (ix >= edges[i]) & (ix <= edges[i + 1])
            ax.fill_between(
                ix[find], b, d[find],
                ec=ec, fc=fc, lw=(plt_params['lw'] / 2), ls=plt_params['ls'], alpha=alphas[i]
            )

        ax.plot(
            ix, d,
            color=cm[s], lw=plt_params['lw'], ls=plt_params['ls'], alpha=plt_params['alpha'], label=label
        )

    if plt_params['xlim'] is not None:
        plt_params['xmin'], plt_params['xmax'] = plt_params['xlim']

    xmin = plt_params['xmin']
    xmax = plt_params['xmax']
    xlim = list(ax.get_xlim())

    if xmin is not None:
        xlim[0] = xmin
    if xmax is not None:
        xlim[1] = xmax

    if plt_params['xticks'] is not None:
        if isinstance(plt_params['xticks'], (list, tuple)):
            ax.set_xticks(plt_params['xticks'])
        else:
            set_num_locator(n=plt_params['xticks'], axis='x', ax=ax)

    if plt_params['xfmt'] is not None:
        ax.set_xticklabels([plt_params['xfmt'].format(format_number(t)) for t in ax.get_xticks()])

    if plt_params['show_ridge']:
        ylim = (0, 1)

        for s in items:
            ax.axhline(rpos[s], color=cm[s], ls='-', lw=2, alpha=0.5)

        if not plt_params['legend']:
            yticks = [v + rscale / 2 for k, v in rpos.items()]
            ylabels = items
        else:
            yticks = []
            ylabels = []

        ax.set_yticks(yticks)
        ax.set_yticklabels(ylabels)
        ax.yaxis.set_tick_params(which='both', length=0)
    else:
        if plt_params['ylim'] is not None:
            plt_params['ymin'], plt_params['ymax'] = plt_params['ylim']

        ymin = plt_params['ymin']
        ymax = plt_params['ymax']
        ylim = list(ax.get_ylim())

        if ymin is not None:
            ylim[0] = np.max([ymin, 0])
        else:
            ylim[0] = 0
        if ymax is not None:
            ylim[1] = np.min([ymax, 1])

        if plt_params['yticks'] is not None:
            if isinstance(plt_params['yticks'], (list, tuple)):
                ax.set_yticks(plt_params['yticks'])
            else:
                set_num_locator(n=plt_params['yticks'], axis='y', ax=ax)

        if plt_params['yfmt'] is not None:
            ax.set_yticklabels([plt_params['yfmt'].format(format_number(t)) for t in ax.get_yticks()])

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    vlines = plt_params['vlines'] or []
    for line in vlines:
        ax.axvline(line, color=get_color('grey-light'), ls='--', lw=2, alpha=0.5)

    hlines = plt_params['hlines'] or []
    for line in hlines:
        ax.axhline(line, color=get_color('grey-light'), ls='--', lw=2, alpha=0.5)

    if plt_params['grid']:
        gaxis = plt_params['grid'] if plt_params['grid'] in ['x', 'y'] else 'both' if not plt_params[
            'show_ridge'] else 'x'
        ax.grid(True, axis=gaxis, ls='--', lw=.5, color='k', alpha=0.2)
    else:
        ax.grid(False)

    if not plt_params['show_ridge'] and plt_params['legend']:
        ax.legend(
            loc='best', fontsize='large', labelspacing=1
        )
    elif plt_params['show_ridge'] and plt_params['legend']:
        labels = dict(zip(items, ax.get_legend_handles_labels()[1]))
        for s in items:
            ax.text(
                0.99 * ax.get_xlim()[1], rpos[s] + rscale * 0.96, labels[s],
                va='top', ha='right', multialignment='left', color='k', size='large', linespacing=1.5, bbox=dict(
                    edgecolor='k', facecolor='w', boxstyle='round, pad=0.4', lw=1
                )
            )

    set_title(plt_params['title'], ax=ax)
    set_note(plt_params['note'], ax=ax)

    plot_figure(show=show, path=path, fig=fig)


def plot_category(data,
    stat='count', grid=True, cm=None,
    figsize=(20, 10), title=None, note=None, ax=None, show=True, path=None
):
    d = to_pandas(data, pdtype='Series')

    if stat == 'probability':
        d = d.value_counts(normalize=True, ascending=False)
    else:
        d = d.value_counts(ascending=False)

    if cm is None:
        cm = colors[palette[0]]
    else:
        cm = colors[cm] if cm in colors else cm

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = None

    ax.bar(d.index, d.values, color=cm)

    if stat == 'probability':
        ax.set_yticklabels(['{:d}%'.format(format_number(i * 100)) for i in ax.get_yticks()])

    if grid:
        ax.grid(True, axis='y', ls='--', lw=.5, color='k', alpha=0.2)
    else:
        ax.set_yticks([])

    set_title(title, ax=ax)
    set_note(note, ax=ax)

    if fig is not None:
        if show:
            plt.show()
        if path:
            fig.savefig(path, bbox_inches='tight')
        plt.close()


def plot_prevalence(data, ax=None, show=True, path=None, **kwargs):
    _plt_params = dict(
        orientation=None, drop_empty=False, show_any=False,
        vmin=None, vmax=None, vticks=None, grid=True, cm=None,
        title=None, note=None
    )
    _fig_params = dict(
        figsize=None
    )

    df = to_pandas(data, pdtype='DataFrame')

    if df.isin([0, 1]).all().sum() < df.shape[1]:
        x = df.isnull()
    else:
        x = df.copy()

    n_rows = x.shape[0]
    s = x.sum() / n_rows

    plt_params = get_params(_plt_params, None, 'plt_params', **kwargs)
    fig_params = get_params(_fig_params, None, 'fig_params', **kwargs)

    if plt_params['drop_empty']:
        c = x.any()
        s = s[c[c].index]

    s = s.sort_values()

    if plt_params['show_any']:
        s = s.append(pd.Series({
            '[any]': x.any(axis=1).sum() / n_rows
        }))

    s = s[::-1] * 100

    cm = get_color_map(plt_params['cm'], 1)[0]
    orientation = plt_params['orientation'] or 'left'

    if fig_params['figsize'] is None:
        if orientation in ('top', 'bottom'):
            fig_params['figsize'] = (20, 8)
        elif orientation in ('left', 'right'):
            fig_params['figsize'] = (20, s.shape[0] // 3)

    fig, ax = create_figure(ax=ax, **fig_params)

    if orientation in ('top', 'bottom'):
        ax.bar(
            s.index,
            s.values,
            color=cm
        )
        axis = 'y'
    elif orientation in ('left', 'right'):
        ax.barh(
            s.index,
            s.values,
            color=cm
        )
        ax.invert_yaxis()
        axis = 'x'
    else:
        axis = None

    vlim = None
    vmin = plt_params['vmin']
    vmax = plt_params['vmax']
    if axis == 'y':
        vlim = list(ax.get_ylim())
    elif axis == 'x':
        vlim = list(ax.get_xlim())

    if vmin is not None:
        vlim[0] = vmin
    if vmax is not None:
        vlim[1] = vmax

    if vlim[0] < 0:
        vlim[0] = 0
    if vlim[1] > 100:
        vlim[1] = 100

    if axis == 'y':
        ax.set_ylim(vlim)
    elif axis == 'x':
        ax.set_xlim(vlim)

    vticks = plt_params['vticks']
    if not plt_params['grid']:
        vticks = []

    if vticks is not None:
        if isinstance(vticks, (list, tuple)):
            if axis == 'y':
                ax.set_yticks(vticks)
            elif axis == 'x':
                ax.set_xticks(vticks)
        else:
            set_num_locator(n=vticks, axis=axis, ax=ax)

    ax.set_xlabel(None)
    ax.set_ylabel(None)

    if axis == 'y':
        ax.set_yticklabels(['{}%'.format(format_number(i)) for i in ax.get_yticks()])
        ax.set_xticklabels(list(s.index), rotation=45)
    elif axis == 'x':
        ax.set_xticklabels(['{}%'.format(format_number(i)) for i in ax.get_xticks()])
        ax.set_yticklabels(list(s.index))

    if plt_params['grid']:
        ax.grid(True, axis=axis, ls='--', lw=.5, color='k', alpha=0.2)
    else:
        ax.grid(False)

    set_title(plt_params['title'], ax=ax)
    set_note(plt_params['note'], ax=ax)

    plot_figure(show=show, path=path, fig=fig)


def plot_corr_matrix(data, ax=None, show=True, path=None, **kwargs):
    _plt_params = dict(
        corr_method='pearson', annot=False, fmt='0.2f', triang=True, cbar=True, cm='RdBu_r',
        title=None, note=None
    )
    _fig_params = dict(
        figsize=None
    )

    df = to_pandas(data, pdtype='DataFrame')

    plt_params = get_params(_plt_params, None, 'plt_params', **kwargs)
    fig_params = get_params(_fig_params, None, 'fig_params', **kwargs)

    # Feature correlation matrix
    corr_matrix = df.sort_index(axis=1)
    if df.shape[0] != df.shape[1]:
        corr_matrix = corr_matrix.corr(method=plt_params['corr_method'])

    # Upper triangle mask
    mask = None
    if plt_params['triang']:
        mask = np.triu(np.ones_like(corr_matrix, dtype=np.bool_))

    if fig_params['figsize'] is None:
        fig_params['figsize'] = (20, corr_matrix.shape[0] // 3)

    fig, ax = create_figure(ax=ax, **fig_params)

    sns.heatmap(
        corr_matrix,
        mask=mask,
        cmap=plt_params['cm'],
        cbar=plt_params['cbar'],
        annot=plt_params['annot'],
        fmt=plt_params['fmt'],
        vmin=-1,
        vmax=1,
        center=0,
        ax=ax
    )

    set_cat_locator(axis='x', ax=ax)

    set_title(plt_params['title'], ax=ax)
    set_note(plt_params['note'], ax=ax)

    plot_figure(show=show, path=path, fig=fig)


def plot_corr_dendogram(data, ax=None, show=True, path=None, **kwargs):
    _plt_params = dict(
        orientation=None, corr_method='pearson', linkage_method='single', cm=None,
        size=(20, 8), title=None, note=None
    )
    _fig_params = dict(
        figsize=(20, 10)
    )

    df = to_pandas(data, pdtype='DataFrame')

    plt_params = get_params(_plt_params, None, 'plt_params', **kwargs)
    fig_params = get_params(_fig_params, None, 'fig_params', **kwargs)

    corr_matrix = df.sort_index(axis=1).corr(method=plt_params['corr_method'])
    corr_linkage = hierarchy.linkage(corr_matrix, method=plt_params['linkage_method'])

    cm = get_color_map(plt_params['cm'])
    hierarchy.set_link_color_palette(cm)
    orientation = plt_params['orientation'] or 'bottom'
    leaf_rotation = 45 if orientation in ('top', 'bottom') else 0

    fig, ax = create_figure(ax=ax, **fig_params)

    hierarchy.dendrogram(
        corr_linkage,
        labels=list(df.columns),
        leaf_rotation=leaf_rotation,
        orientation=orientation,
        distance_sort='descending',
        ax=ax
    )

    hierarchy.set_link_color_palette(None)

    if orientation in ('top', 'bottom'):
        ax.set_yticklabels(['{}'.format(format_number(i)) for i in ax.get_yticks()])
        set_cat_locator(axis='x', ax=ax)
    elif orientation in ('left', 'right'):
        ax.set_xticklabels(['{}'.format(format_number(i)) for i in ax.get_xticks()])

    set_title(plt_params['title'], ax=ax)
    set_note(plt_params['note'], ax=ax)

    plot_figure(show=show, path=path, fig=fig)


def plot_scatter(x, y, ax=None, show=True, path=None, **kwargs):
    # reg_type -> ls: LeastSquares (default), lk: LocalKernel
    _plt_params = dict(
        show_reg=False, show_kde=False, show_diag=False, show_quad=False,
        weights=None, diag_cm=None, quad_cm=None,
        xlim=None, xticks=None, xfmt=None, xlog=False, ylim=None, yticks=None, yfmt=None, ylog=False,
        annot=False, annot_loc=None, annot_adjust=False, s=50, labels=True, vlines=None, hlines=None,
        grid=True, cm=None, alpha=None, legend=True, title=None, note=None
    )
    _reg_params = dict(
        reg_type='ls', poly_deg=1, cov_type='hac', reg_kernel='gaussian',
        reg_bw_type='adaptive', reg_bw='isj', reg_alpha=0.05, reg_cm=None, reg_lw=2
    )
    _kde_params = dict(
        kde_kernel='gaussian', kde_bw='scott', kde_fill=True,
        outliers='keep', ol_std=3, kde_cm=None, kde_lw=2
    )
    _fig_params = dict(
        figsize=(15, 9)
    )

    sx = to_pandas(x, pdtype='Series')
    sy = to_pandas(y, pdtype='Series')

    plt_params = get_params(_plt_params, None, 'plt_params', **kwargs)
    reg_params = get_params(_reg_params, None, 'reg_params', **kwargs)
    kde_params = get_params(_kde_params, None, 'kde_params', **kwargs)
    fig_params = get_params(_fig_params, None, 'fig_params', **kwargs)

    if plt_params['cm'] is None:
        cm = colors[palette[0]]
    elif is_array(plt_params['cm']):
        cm = [get_color(c) for c in plt_params['cm']]
    else:
        cm = colors[plt_params['cm']] if plt_params['cm'] in colors else plt_params['cm']

    fig, ax = create_figure(ax=ax, **fig_params)

    ax.scatter(sx, sy, s=plt_params['s'], c=cm, alpha=plt_params['alpha'], zorder=3)

    if plt_params['xlim'] is not None:
        ax.set_xlim(plt_params['xlim'])
    else:
        xlim_ = list(ax.get_xlim())
        if np.abs(xlim_[0]) > 0:
            xlim_[0] *= 0.95
        if np.abs(xlim_[1]) > 0:
            xlim_[1] *= 1.05
        ax.set_xlim(xlim_)

    if plt_params['ylim'] is not None:
        ax.set_ylim(plt_params['ylim'])
    else:
        ylim_ = list(ax.get_ylim())
        if np.abs(ylim_[0]) > 0:
            ylim_[0] *= 0.95
        if np.abs(ylim_[1]) > 0:
            ylim_[1] *= 1.05
        ax.set_ylim(ylim_)

    if not is_empty(plt_params['annot']):
        if is_array(plt_params['annot']):
            annot = plt_params['annot']
        else:
            annot = sx.index

        if plt_params['annot_loc'] == 'inside':
            for i in np.arange(len(sx)):
                ax.text(sx.iloc[i], sy.iloc[i], annot[i], va='center', ha='center', size='small', color='w', weight='semibold')
        else:
            annot = [
                ax.annotate(annot[i], (sx.iloc[i], sy.iloc[i]))
                for i in np.arange(len(sx))
                if annot[i]
            ]
            if plt_params['annot_adjust']:
                adjust_text(
                    annot,
                    ax=ax,
                    expand_points=(2, 2),
                    arrowprops=dict(arrowstyle='-', color='k', lw=0.5)
                )
            else:
                for text in annot:
                    offset = np.abs(np.subtract(*ax.get_ylim()) / 60)
                    pos = text.get_position()
                    text.set_position((pos[0], pos[1] + offset))

    if plt_params['xlog'] and plt_params['ylog']:
        set_log_scale(axis='both', ax=ax)
    elif plt_params['xlog']:
        set_log_scale(axis='x', ax=ax)
    elif plt_params['ylog']:
        set_log_scale(axis='y', ax=ax)

    if plt_params['xticks'] is not None:
        if isinstance(plt_params['xticks'], (list, tuple)):
            ax.set_xticks(plt_params['xticks'])
        else:
            set_num_locator(n=plt_params['xticks'], axis='x', ax=ax)

    if is_numeric(sx):
        if plt_params['xfmt'] is not None:
            ax.set_xticklabels([plt_params['xfmt'].format(format_number(t)) for t in ax.get_xticks()])
    else:
        set_cat_locator(axis='x', ax=ax)

    if plt_params['yticks'] is not None:
        if isinstance(plt_params['yticks'], (list, tuple)):
            ax.set_yticks(plt_params['yticks'])
        else:
            set_num_locator(n=plt_params['yticks'], axis='y', ax=ax)

    if is_numeric(sy):
        if plt_params['yfmt'] is not None:
            ax.set_yticklabels([plt_params['yfmt'].format(format_number(t)) for t in ax.get_yticks()])
    else:
        set_cat_locator(axis='y', ax=ax)

    if plt_params['show_reg']:
        if reg_params['reg_type'] == 'ls':
            est = LeastSquaresEstimator(
                sx,
                sy,
                weights=plt_params['weights'],
                poly_deg=reg_params['poly_deg'],
                cov_type=reg_params['cov_type']
            ).fit()
        elif reg_params['reg_type'] == 'lk':
            est = LocalKernelEstimator(
                sx,
                sy,
                weights=plt_params['weights'],
                poly_deg=reg_params['poly_deg'],
                cov_type=reg_params['cov_type'],
                kernel=reg_params['reg_kernel'],
                bw_type=reg_params['reg_bw_type'],
                bw=reg_params['reg_bw']
            ).fit()
        else:
            raise ValueError('Invalid regression type: {}'.format(reg_params['reg_type']))

        xlim_ = list(ax.get_xlim())
        px = np.linspace(xlim_[0], xlim_[1], 100)

        y_est = est.predict(px, alpha=reg_params['reg_alpha'])

        ax.plot(
            px,
            y_est['mean'],
            color=reg_params['reg_cm'] or get_color('blue'),
            lw=reg_params['reg_lw'],
            label='$R^2$ {:0.2f}'.format(est.r2_score)
        )

        if reg_params['reg_alpha'] is not None:
            ax.fill_between(
                px,
                y_est['cmin'],
                y_est['cmax'],
                color=reg_params['reg_cm'] or get_color('blue'),
                lw=0,
                alpha=0.2,
                label='CI {:d}%'.format(int((1 - reg_params['reg_alpha']) * 100))
            )

    if plt_params['show_kde']:
        xlim_ = list(ax.get_xlim())
        ylim_ = list(ax.get_ylim())
        px = np.linspace(xlim_[0], xlim_[1], 100)
        py = np.linspace(ylim_[0], ylim_[1], 100)

        plot_kde(
            np.column_stack([sx, sy]),
            p=np.column_stack([px, py]),
            weights=plt_params['weights'],
            xlim=xlim_,
            ylim=ylim_,
            bw=kde_params['kde_bw'],
            fill=kde_params['kde_fill'],
            outliers=kde_params['outliers'],
            ol_std=kde_params['ol_std'],
            cm=kde_params['kde_cm'],
            lw=kde_params['kde_lw'],
            ax=ax
        )

    if plt_params['show_diag']:
        xlim_ = list(ax.get_xlim())
        ylim_ = list(ax.get_ylim())
        lims_ = [
            np.min([xlim_[0], ylim_[0]]),
            np.max([xlim_[1], ylim_[1]])
        ]
        ax.set_xlim(lims_)
        ax.set_ylim(lims_)
        ax.set_aspect('equal')

        ax.plot(ax.get_xlim(), ax.get_ylim(), color=get_color('grey-light'), ls='--', lw=2, alpha=0.5)
        if plt_params['diag_cm']:
            cm_ = plt_params['diag_cm'] if is_array(plt_params['diag_cm']) else ['green', 'red']
            if len(cm_) > 0 and cm_[0]:
                ax.fill_between(
                    ax.get_xlim(), ax.get_ylim(), np.max(ax.get_ylim()),
                    color=colors[cm_[0]], lw=0, alpha=0.2
                )
            if len(cm_) > 1 and cm_[1]:
                ax.fill_between(
                    ax.get_xlim(), ax.get_ylim(), np.min(ax.get_ylim()),
                    color=colors[cm_[1]], lw=0, alpha=0.2
                )
    if plt_params['show_quad']:
        ax.axvline(x.quantile(0.5), color=get_color('grey-light'), ls='--', lw=2, alpha=0.5)
        ax.axhline(y.quantile(0.5), color=get_color('grey-light'), ls='--', lw=2, alpha=0.5)
        if plt_params['quad_cm']:
            cm_ = plt_params['quad_cm'] if is_array(plt_params['quad_cm']) else ['yellow', 'green', 'red', 'orange']
            if len(cm_) > 0 and cm_[0]:
                ax.fill_between(
                    [ax.get_xlim()[0], sx.quantile(0.5)], sy.quantile(0.5), ax.get_ylim()[1],
                    color=colors[cm_[0]], lw=0, alpha=0.2
                )
            if len(cm_) > 1 and cm_[1]:
                ax.fill_between(
                    [sx.quantile(0.5), ax.get_xlim()[1]], sy.quantile(0.5), ax.get_ylim()[1],
                    color=colors[cm_[1]], lw=0, alpha=0.2
                )
            if len(cm_) > 2 and cm_[2]:
                ax.fill_between(
                    [ax.get_xlim()[0], sx.quantile(0.5)], ax.get_ylim()[0], sy.quantile(0.5),
                    color=colors[cm_[2]], lw=0, alpha=0.2
                )
            if len(cm_) > 3 and cm_[3]:
                ax.fill_between(
                    [sx.quantile(0.5), ax.get_xlim()[1]], ax.get_ylim()[0], sy.quantile(0.5),
                    color=colors[cm_[3]], lw=0, alpha=0.2
                )

    if plt_params['labels']:
        ax.set_xlabel(sx.name)
        ax.set_ylabel(sy.name)

    if plt_params['vlines'] is not None:
        for p in plt_params['vlines']:
            ax.axvline(p, color=get_color('grey-light'), ls='--', lw=2, alpha=0.5)

    if plt_params['hlines'] is not None:
        for p in plt_params['hlines']:
            ax.axhline(p, color=get_color('grey-light'), ls='--', lw=2, alpha=0.5)

    if plt_params['grid']:
        gaxis = plt_params['grid'] if plt_params['grid'] in ['x', 'y'] else 'both'
        ax.grid(True, axis=gaxis, ls='--', lw=.5, color='k', alpha=0.2)
    else:
        ax.set_yticks([])

    if plt_params['legend'] and plt_params['show_reg']:
        ax.legend(loc='best')

    set_title(plt_params['title'], ax=ax)
    set_note(plt_params['note'], ax=ax)

    adjust_figure(fig=fig)
    plot_figure(show=show, path=path, fig=fig)


def plot_diverging(x, y, ax=None, show=True, path=None, **kwargs):
    _plt_params = dict(
        yaxis='left', annot=False,
        xlim=None, xticks=None, xfmt=None, xlog=False, ylim=None, yticks=None, yfmt=None, ylog=False,
        grid=True, cm=None, legend=True, title=None, note=None
    )
    _fig_params = dict(
        figsize=(15, 9)
    )

    sx = to_pandas(x, pdtype='Series')
    sy = to_pandas(y, pdtype='Series')

    plt_params = get_params(_plt_params, None, 'plt_params', **kwargs)
    fig_params = get_params(_fig_params, None, 'fig_params', **kwargs)

    cm = get_color_map(plt_params['cm'], 2)

    fig, ax = create_figure(ax=ax, **fig_params)

    if plt_params['xlim'] is not None:
        xlim_ = [0, plt_params['xlim'][1]]
    else:
        xlim_ = [0, np.max(sx) * 1.05]

    if plt_params['ylim'] is not None:
        ylim_ = [0, plt_params['ylim'][1]]
    else:
        ylim_ = [0, np.max(sy) * 1.05]

    x_norm = sx / xlim_[1]
    y_norm = sy / ylim_[1]

    py = ax.barh(y_norm.index, -y_norm.values, color=cm[1], label=sy.name)
    px = ax.barh(x_norm.index, x_norm.values, color=cm[0], label=sx.name)
    ax.axvline(0, color='k', lw=1, alpha=0.5)

    ax.set_xlim([-1, 1])
    if plt_params['yaxis'] == 'right':
        ax.yaxis.tick_right()

    if plt_params['xlog'] and plt_params['ylog']:
        set_log_scale(axis='both', ax=ax)
    elif plt_params['xlog']:
        set_log_scale(axis='x', ax=ax)
    elif plt_params['ylog']:
        set_log_scale(axis='y', ax=ax)

    xticks_ = None
    xlabels_ = None
    if plt_params['xticks'] is not None:
        xticks_ = np.array([t / plt_params['xlim'][1] for t in plt_params['xticks'] if t > 0])[::-1]
        xlabels_ = np.array([t for t in plt_params['xticks'] if t > 0])[::-1].astype(int)
        if plt_params['xfmt'] is not None:
            xlabels_ = np.array([plt_params['xfmt'].format(format_number(t)) for t in xlabels_])

    yticks_ = None
    ylabels_ = None
    if plt_params['yticks'] is not None:
        yticks_ = -np.array([t / plt_params['ylim'][1] for t in plt_params['yticks'] if t > 0])[::-1]
        ylabels_ = np.array([t for t in plt_params['yticks'] if t > 0])[::-1].astype(int)
        if plt_params['yfmt'] is not None:
            ylabels_ = np.array([plt_params['yfmt'].format(format_number(t)) for t in ylabels_])
        else:
            ylabels_ = np.array([format_number(t) for t in ylabels_])

    if plt_params['xticks'] is not None and plt_params['yticks'] is not None:
        ax.set_xticks(np.hstack([yticks_, np.array([0]), xticks_]))
        ax.set_xticklabels(np.hstack([ylabels_, np.array([0]), xlabels_]))
    else:
        ax.set_xticklabels([])

    if not is_empty(plt_params['annot']):
        xannot_ = np.array([format_number(v) if v > 0 else '' for v in sx.values])
        if plt_params['xfmt'] is not None:
            xannot_ = np.array([plt_params['xfmt'].format(v) if v != '' else '' for v in xannot_])

        yannot_ = np.array([format_number(v) if v > 0 else '' for v in sy.values])
        if plt_params['yfmt'] is not None:
            yannot_ = np.array([plt_params['yfmt'].format(v) if v != '' else '' for v in yannot_])

        ax.bar_label(px, labels=xannot_, padding=3)
        ax.bar_label(py, labels=yannot_, padding=3)

    if plt_params['grid']:
        gaxis = plt_params['grid'] if plt_params['grid'] in ['x', 'y'] else 'both'
        ax.grid(True, axis=gaxis, ls='--', lw=.5, color='k', alpha=0.2)
    else:
        ax.set_yticks([])

    if plt_params['legend']:
        ax.add_artist(ax.legend(handles=[py], bbox_to_anchor=(0, 1), loc='lower left'))
        ax.legend(handles=[px], bbox_to_anchor=(1, 1), loc='lower right')

    set_title(plt_params['title'], ax=ax)
    set_note(plt_params['note'], ax=ax)

    adjust_figure(fig=fig)
    plot_figure(show=show, path=path, fig=fig)


def plot_scores(data, ax=None, show=True, path=None, **kwargs):
    _plt_params = dict(
        abs_val=False, sort=False, vmin=None, vmax=None, vticks=None, vlines=None,
        fmt=None, grid=True, cm=None, alpha=None, legend=False, title=None, note=None
    )
    _fig_params = dict(
        figsize=None
    )

    df = to_pandas(data, pdtype='DataFrame')
    series = list(df.columns)

    col_main = series[0]
    cols_err = series[1:] if len(series) > 1 else None

    plt_params = get_params(_plt_params, None, 'plt_params', **kwargs)
    fig_params = get_params(_fig_params, None, 'fig_params', **kwargs)

    if plt_params['cm'] is not None:
        cm = plt_params['cm']
    elif plt_params['abs_val']:
        cm = df[col_main].apply(np.sign).map({
            1: 'green',
            -1: 'red'
        }).fillna('blue')
    else:
        cm = 'blue'

    cm = pd.Series(cm, index=df.index)

    if plt_params['alpha'] is not None:
        alpha = plt_params['alpha']
    else:
        alpha = 1

    alpha = pd.Series(alpha, index=df.index)

    df['cm'] = [get_color(cm[i], alpha[i]) for i in df.index]

    if plt_params['abs_val']:
        df[col_main] = df[col_main].abs()
        if cols_err is not None:
            df[cols_err] = df[cols_err[::-1]]

    if plt_params['sort']:
        df = df.sort_values(col_main)

    s = df[col_main].fillna(0).values
    idx = df.index
    err = df[cols_err].fillna(0).T.values if cols_err is not None else None
    cm = df['cm'].values

    if fig_params['figsize'] is None:
        fig_params['figsize'] = (20, s.shape[0] // 3)

    fig, ax = create_figure(ax=ax, **fig_params)

    ax.barh(idx, s, xerr=err, color=cm, align='center', tick_label=idx, label=col_main)
    ax.axvline(0, color='k', lw=1, alpha=0.5)

    if plt_params['abs_val']:
        ax.set_xlim([0, ax.get_xlim()[1]])

    vmin = plt_params['vmin']
    vmax = plt_params['vmax']
    vlim = list(ax.get_xlim())

    if vmin is not None:
        vlim[0] = vmin
    if vmax is not None:
        vlim[1] = vmax

    ax.set_xlim(vlim)

    vticks = plt_params['vticks']
    if not plt_params['grid']:
        vticks = []

    if vticks is not None:
        if isinstance(vticks, (list, tuple)):
            ax.set_xticks(vticks)
        else:
            set_num_locator(n=vticks, axis='x', ax=ax)

    fmt = plt_params['fmt']
    vticklabels = [format_number(t) for t in ax.get_xticks()]
    if fmt is not None:
        vticklabels = [fmt.format(t) for t in vticklabels]
    ax.set_xticklabels(vticklabels)

    if plt_params['vlines'] is not None:
        for p in plt_params['vlines']:
            ax.axvline(p, color=get_color('grey-light'), ls='--', lw=2, alpha=0.5)

    if plt_params['grid']:
        gaxis = plt_params['grid'] if plt_params['grid'] in ['x', 'y'] else 'both'
        ax.grid(True, axis=gaxis, ls='--', lw=.5, color='k', alpha=0.2)
    else:
        ax.grid(False)

    if plt_params['legend']:
        ax.legend(bbox_to_anchor=(0, 1), loc='lower left')

    set_title(plt_params['title'], ax=ax)
    set_note(plt_params['note'], ax=ax)

    plot_figure(show=show, path=path, fig=fig)


def plot_odds_ratio(data, ax=None, show=True, path=None, **kwargs):
    _plt_params = dict(
        sort=False, vmin=None, vmax=None, vticks=None,
        grid=True, title=None, note=None
    )
    _fig_params = dict(
        figsize=None
    )

    df = to_pandas(data)
    series = list(df.columns)

    col_main, col_min, col_max = series

    plt_params = get_params(_plt_params, None, 'plt_params', **kwargs)
    fig_params = get_params(_fig_params, None, 'fig_params', **kwargs)

    if plt_params['sort']:
        df = df.sort_values(col_main)

    s = df[col_main].values
    idx = df.index
    err_lo = s - df[col_min].values
    err_up = df[col_max].values - s

    if fig_params['figsize'] is None:
        fig_params['figsize'] = (20, s.shape[0] // 3)

    fig, ax = create_figure(ax=ax, **fig_params)

    ax.errorbar(
        s,
        idx,
        xerr=[err_lo, err_up],
        color='k',
        fmt='s',
        markerfacecolor='w',
        capsize=3
    )
    ax.axvline(1, color='k', lw=1, alpha=0.5)

    vmin = plt_params['vmin']
    vmax = plt_params['vmax']
    vlim = list(ax.get_xlim())

    if vmin is not None:
        vlim[0] = vmin
    if vmax is not None:
        vlim[1] = vmax

    ax.set_xlim(vlim)

    vticks = plt_params['vticks']
    if not plt_params['grid']:
        vticks = []

    if vticks is not None:
        if isinstance(vticks, (list, tuple)):
            ax.set_xticks(vticks)
        else:
            set_num_locator(n=vticks, axis='x', ax=ax)

    if plt_params['grid']:
        gaxis = plt_params['grid'] if plt_params['grid'] in ['x', 'y'] else 'both'
        ax.grid(True, axis=gaxis, ls='--', lw=.5, color='k', alpha=0.2)
    else:
        ax.grid(False)

    set_title(plt_params['title'], ax=ax)
    set_note(plt_params['note'], ax=ax)

    plot_figure(show=show, path=path, fig=fig)


def plot_sfs(data, ax=None, show=True, path=None, **kwargs):
    _col_params = dict(
        ylim=None, ymin=None, ymax=None, yticks=None, cm=None, marker=True, lw=3
    )
    _plt_params = dict(
        show_names=True, show_limits=True,
        grid=True, margin=0.01, title=None, note=None
    )
    _fig_params = dict(
        figsize=(20, 10)
    )

    df = to_pandas(data)
    series = ['avg_score', 'lo_score', 'hi_score']

    col_params = get_params(_col_params, series, 'col_params', **kwargs)
    plt_params = get_params(_plt_params, None, 'plt_params', **kwargs)
    fig_params = get_params(_fig_params, None, 'fig_params', **kwargs)

    if plt_params['show_names']:
        df['feature'] = df.feature.fillna('-')
        df = df.set_index('feature').rename_axis(None)

    df['lo_score'] = df.avg_score - df.ci_bound
    df['hi_score'] = df.avg_score + df.ci_bound

    fig, ax = create_figure(ax=ax, **fig_params)

    plot_series_ci(
        df[series],
        col_params=col_params,
        plt_params=plt_params,
        fig_params=fig_params,
        ax=ax
    )

    if plt_params['show_limits']:
        k1 = len(df[df.optimal]) - 1  # Optimal
        ax.axvline(k1, color=get_color('green-light'), ls='--', lw=2)

        k2 = len(df[df.best]) - 1  # Best
        ax.axvline(k2, color=get_color('purple-light'), ls='--', lw=2)

    plot_figure(show=show, path=path, fig=fig)


def plot_geo_heatmap(data, geo_map,
    labels=None, vmin=None, vmax=None, cmap='OrRd', scheme=None,
    figsize=(10, 10), legend=False, title=None, note=None, ax=None, show=True, path=None
):
    geo_map = geo_map.merge(
        to_pandas(data, pdtype='Series').rename('metric'),
        left_on='region',
        right_index=True,
        how='left'
    ).to_crs('EPSG:3395')

    if vmin is None:
        vmin = geo_map.metric.dropna().min()
    if vmax is None:
        vmax = geo_map.metric.dropna().max()

    geo_map['metric'] = geo_map.metric.where((pd.isnull(geo_map.metric)) | (geo_map.metric >= vmin), vmin)
    geo_map['metric'] = geo_map.metric.where((pd.isnull(geo_map.metric)) | (geo_map.metric <= vmax), vmax)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = None

    geo_map.plot(
        column='metric',
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        scheme=scheme,
        legend=legend,
        missing_kwds={
            'color': get_color('grey-alpha')
        },
        ax=ax
    )

    if labels is not None:
        geo_map['centroid'] = geo_map['geometry'].centroid
        geo_map = geo_map.merge(
            pd.Series(labels).rename('label'),
            left_on='region',
            right_index=True,
            how='left'
        )

        props = dict(boxstyle='round', facecolor='linen', alpha=1)
        geo_lab = geo_map[pd.notnull(geo_map.label)]
        for x, y, label in zip(geo_lab.centroid.x, geo_lab.centroid.y, geo_lab.label):
            ax.annotate(
                label,
                xy=(x, y),
                size='large',
                ha='center',
                va='center',
                bbox=props
            )

    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

    set_title(title, ax=ax)
    set_note(note, y=-30, ax=ax)

    plot_figure(show=show, path=path, fig=fig)


def plot_series_mesh(data, ax=None, show=True, path=None, **kwargs):
    # aspect -> auto (default), equal, num (height / width)
    _plt_params = dict(
        how=None, aspect='auto', vmin=None, vmax=None, cm='Reds', annot=None, dt_min=None, dt_max=None,
        freq='week', xticks=None, vlines=None, cbar=False, title=None, note=None
    )
    _fig_params = dict(
        figsize=None
    )

    df = to_pandas(data, pdtype='DataFrame')

    plt_params = get_params(_plt_params, None, 'plt_params', **kwargs)
    fig_params = get_params(_fig_params, None, 'fig_params', **kwargs)

    if plt_params['how'] is not None:
        df = df.resample('D').agg(plt_params['how'])

    cmap = plt_params['cm'] or 'blue'
    if not isinstance(cmap, mcolors.Colormap):
        if is_array(cmap):
            cmap = build_color_cat_map(cmap)
        elif '#' in cmap or cmap in colors:
            cmap = build_color_seq_map(get_color(cmap))

    df = ts_range(df, dt_min=plt_params['dt_min'], dt_max=plt_params['dt_max'])
    series = list(df.columns)
    idx = df.index
    is_ts = isinstance(idx, pd.DatetimeIndex)
    is_num = idx.is_numeric()

    vmin = plt_params.get('vmin', df.min().min())
    vmax = plt_params.get('vmax', df.max().max())

    annot = plt_params['annot']
    if annot is True:
        annot = df.applymap(format_number)
    elif annot is not None:
        annot = to_pandas(annot, pdtype='DataFrame').reindex(df.index).fillna('')

    d_plot = df.T
    d_fill = pd.DataFrame(1, columns=idx, index=series)

    x = np.arange(len(idx) + 1)
    y = np.arange(len(series) + 1)

    if fig_params['figsize'] is None:
        fig_params['figsize'] = (20, y.shape[0] // 3)

    fig, ax = create_figure(ax=ax, **fig_params)

    ax.pcolormesh(x, y, d_fill, vmin=0, vmax=1, cmap=mcolors.ListedColormap(['whitesmoke']))
    mesh = ax.pcolormesh(x, y, d_plot, vmin=vmin, vmax=vmax, cmap=cmap, linewidth=1, edgecolors='white')

    ax.set_aspect(plt_params['aspect'])
    for side in ('top', 'right', 'left', 'bottom'):
        ax.spines[side].set_visible(False)
    ax.xaxis.set_tick_params(which='both', length=0)
    ax.yaxis.set_tick_params(which='both', length=0)

    ax.set_xlabel('')
    ax.set_ylabel('')

    ax.set_xticks([i + 0.5 for i in np.arange(len(idx))])
    if is_ts:
        ts_freq = Freq.parse(plt_params['freq'])
        ts_fmt = ts_freq.str_label
        ts_ticks = pd.date_range(start=df.index[0], end=df.index[-1], freq=ts_freq.pd_period)
        ax.set_xticklabels([i.strftime(ts_fmt) if i in ts_ticks else '' for i in idx], rotation=45)
    elif is_num:
        xticks = plt_params['xticks']
        if isinstance(xticks, (list, tuple)):
            n_ticks = xticks
        elif isinstance(xticks, int):
            n_ticks = np.arange(min(idx), max(idx) + 1, xticks)
        else:
            n_ticks = np.arange(len(idx))
        ax.set_xticklabels([i if i in n_ticks else '' for i in idx])
    else:
        ax.set_xticklabels(idx, rotation=45, ha='right', rotation_mode='anchor')

    ax.invert_yaxis()
    ax.set_yticks([i + 0.5 for i in np.arange(len(series))])
    ax.set_yticklabels(series, rotation='horizontal', va='center')

    if annot is not None:
        # Force axis to redraw in order to get mesh colors
        try:
            ax.draw(ax.figure.canvas.get_renderer())
        except AttributeError:
            pass

        d_annot = annot.T.values
        d_colors = np.array([get_text_color(color) for color in mesh.get_fc()]).reshape(d_plot.shape)

        for y in np.arange(d_plot.shape[0]):
            for x in np.arange(d_plot.shape[1]):
                s = d_annot[y, x]
                c = d_colors[y, x]

                ax.text(
                    x + 0.5,
                    y + 0.5,
                    c=c,
                    s=s,
                    ha='center',
                    va='center'
                )

    vlines = plt_params['vlines'] or []
    for line in vlines:
        if is_ts:
            line = pd.to_datetime(line)
        line = ax.get_xticks()[list(idx).index(line)]
        ax.axvline(line, color=get_color('grey-light'), ls='--', lw=2, alpha=0.5)

    set_title(plt_params['title'], size='x-large', ax=ax)
    set_note(plt_params['note'], size='large', ax=ax)

    if plt_params['cbar']:
        cb = ax.figure.colorbar(mesh, ax=ax, orientation='vertical')
        cb.outline.set_linewidth(0)
        plt.tight_layout()

    plot_figure(fig=fig, show=show, path=path)


def plot_calmap_year(data, ax=None, show=True, path=None, **kwargs):
    _plt_params = dict(
        year=None, months=None, how=None, vmin=None, vmax=None, cm='Reds',
        full_year=True, month_grid=True, annot=None,
        cbar=False, title=None, note=None
    )
    _fig_params = dict(
        figsize=(20, 3)
    )

    d = to_pandas(data, pdtype='Series')

    plt_params = get_params(_plt_params, None, 'plt_params', **kwargs)
    fig_params = get_params(_fig_params, None, 'fig_params', **kwargs)

    year = plt_params['year'] or datetime.utcnow().year
    months = plt_params['months'] or list(np.arange(1, 13))

    if plt_params['how'] is not None:
        d = d.resample('D').agg(plt_params['how'])

    d = d[(d.index.year == year) & (d.index.month.isin(months))]

    vmin = plt_params['vmin'] or d.min()
    vmax = plt_params['vmax'] or d.max()

    dt_start = pd.to_datetime('{}-01-01'.format(year)) if plt_params['full_year'] else d.index[0]
    dt_end = pd.to_datetime('{}-12-31'.format(year)) if plt_params['full_year'] else d.index[-1]

    d = d.reindex(pd.date_range(start=dt_start, end=dt_end, freq='D'))

    annot = plt_params['annot']
    if annot is True:
        annot = d.apply(format_number)
    elif annot is not None:
        annot = pd.Series(annot).reindex(pd.date_range(start=d.index[0], end=d.index[-1], freq='D'))

    df = pd.DataFrame({
        'data': d,
        'annot': annot,
        'fill': 1,
        'day': d.index.dayofweek,
        'week': d.index.isocalendar().week
    })
    df.loc[(df.index.month == 1) & (df.week > 50), 'week'] = 0
    df.loc[(df.index.month == 12) & (df.week < 10), 'week'] = df.week.max() + 1

    d_plot = df.pivot(index='day', columns='week', values='data').values[::-1]
    d_plot = np.ma.masked_where(np.isnan(d_plot), d_plot)

    d_fill = df.pivot(index='day', columns='week', values='fill').values[::-1]
    d_fill = np.ma.masked_where(np.isnan(d_fill), d_fill)

    cmap = plt_params['cm'] or 'blue'
    if not isinstance(cmap, mcolors.Colormap):
        if is_array(cmap):
            cmap = build_color_cat_map(cmap)
        elif '#' in cmap or cmap in colors:
            cmap = build_color_seq_map(get_color(cmap))

    fig, ax = create_figure(ax=ax, **fig_params)

    ax.pcolormesh(d_fill, vmin=0, vmax=1, cmap=mcolors.ListedColormap(['whitesmoke']))
    mesh = ax.pcolormesh(d_plot, vmin=vmin, vmax=vmax, cmap=cmap, linewidth=1, edgecolors='white')

    ax.set(xlim=(0, d_plot.shape[1]), ylim=(0, d_plot.shape[0]))
    ax.set_aspect('equal')

    for side in ('top', 'right', 'left', 'bottom'):
        ax.spines[side].set_visible(False)
    ax.xaxis.set_tick_params(which='both', length=0)
    ax.yaxis.set_tick_params(which='both', length=0)

    ax.set_xlabel('')
    ax.set_ylabel(str(year), size='xx-large', color='k', ha='center')

    month_abbr = [m for i, m in enumerate(calendar.month_abbr) if i in months]
    month_ticks = np.arange(len(month_abbr))

    ax.set_xticks([
        date(year, i, 15).isocalendar()[1] - df.week.min() for i in months
    ])
    ax.set_xticklabels([
        month_abbr[i] for i in month_ticks
    ], ha='center', size='x-large')

    days = calendar.day_abbr[:]
    day_ticks = np.arange(len(days))

    ax.yaxis.set_ticks_position('right')
    ax.set_yticks([
        6 - i + 0.5 for i in day_ticks
    ])
    ax.set_yticklabels([
        days[i] for i in day_ticks
    ], rotation='horizontal', va='center', size='x-large')

    if annot is not None:
        # Force axis to redraw in order to get mesh colors
        try:
            ax.draw(ax.figure.canvas.get_renderer())
        except AttributeError:
            pass

        d_annot = df.pivot('day', 'week', 'annot').values[::-1]
        d_colors = np.array([get_text_color(color) for color in mesh.get_fc()]).reshape(d_plot.shape)

        for y in np.arange(d_plot.shape[0]):
            for x in np.arange(d_plot.shape[1]):
                if d_plot[y, x] is not np.ma.masked:
                    s = d_annot[y, x]
                    c = d_colors[y, x]

                    ax.text(
                        x + 0.5,
                        y + 0.5,
                        c=c,
                        s=s,
                        ha='center',
                        va='center'
                    )

    if plt_params['month_grid']:
        xticks = []
        start = datetime(year, 1, 1).weekday()

        for month in range(1, 13):
            first = datetime(year, month, 1)
            last = first + relativedelta(months=1, days=-1)
            y0 = 7 - first.weekday()
            y1 = 7 - last.weekday()
            x0 = (int(first.strftime('%j')) + start - 1) // 7
            x1 = (int(last.strftime('%j')) + start - 1) // 7

            P = [
                (x0, y0),
                (x0 + 1, y0),
                (x0 + 1, 7),
                (x1 + 1, 7),
                (x1 + 1, y1 - 1),
                (x1, y1 - 1),
                (x1, 0),
                (x0, 0)
            ]

            xticks.append(x0 + (x1 - x0 + 1) / 2)
            poly = mpatches.Polygon(
                P,
                ec='k',
                fc='None',
                ls='-',
                lw=1,
                zorder=20,
                clip_on=False
            )

            ax.add_artist(poly)

    set_title(plt_params['title'], size='x-large', ax=ax)
    set_note(plt_params['note'], size='large', ax=ax)

    if plt_params['cbar']:
        cb = ax.figure.colorbar(mesh, ax=ax, orientation='vertical')
        cb.outline.set_linewidth(0)
        plt.tight_layout()

    plot_figure(fig=fig, show=show, path=path)


def plot_calmap(data, show=True, path=None, **kwargs):
    _plt_params = dict(
        months=None, how=None, ascending=True, vmin=None, vmax=None, cm='Reds',
        full_year=True, month_grid=True, annot=None,
        cbar=False, title=None, note=None
    )
    _fig_params = dict(
        n_cols=1, figsize=(20, 3.5)
    )

    d = to_pandas(data, pdtype='Series')

    plt_params = get_params(_plt_params, None, 'plt_params', **kwargs)
    fig_params = get_params(_fig_params, None, 'fig_params', **kwargs)

    years = np.unique(d.index.year)
    if not plt_params['ascending']:
        years = years[::-1]
    n_items = len(years)

    if plt_params['how'] is not None:
        d = d.resample('D').agg(plt_params['how'])

    fig, axs = create_figure(n=n_items, **fig_params)

    max_weeks = 0
    for i, year in enumerate(years):
        item_plt_params = dict(plt_params) | dict(year=year, cbar=False, title=None, note=None)
        item_fig_params = dict(fig_params) | dict(figsize=None)

        plot_calmap_year(
            data,
            ax=axs[i],
            plt_params=item_plt_params,
            fig_params=item_fig_params
        )

        max_weeks = max(max_weeks, axs[i].get_xlim()[1])
        axs[i].set_xlim(0, max_weeks)

    set_title(plt_params['title'], size='x-large')
    set_note(plt_params['note'], size='x-large')

    adjust_figure(fig=fig, n=n_items, **fig_params)

    if plt_params['cbar']:
        fig.subplots_adjust(right=0.9)
        cax = fig.add_axes([0.95, 0.01, 0.01, 0.95])
        cb = fig.colorbar(axs[0].get_children()[1], cax=cax, orientation='vertical')
        cb.outline.set_linewidth(0)

    plot_figure(show=show, path=path, fig=fig)


# Grids


def plot_series_grid(data, show=True, path=None, **kwargs):
    _col_params = dict(
        kind='line', yaxis='left', agg=None, smooth=None, log=False, ylim=None, ymin=None, ymax=None, yticks=None,
        fmt=None, cm=None, label=True, annot=None, bottom=None, marker=False, pw=1, lw=3, ls='-', alpha=1
    )
    _plt_params = dict(
        dt_min=None, dt_max=None, freq='week', grid=True, disp=None, vlines=None, hlines=None,
        margin=0.05, legend=True, leg_cols=None, title=None, note=None
    )
    _fig_params = dict(
        n_cols=2, share=False, figsize=(20, 7)
    )

    items = list(data.keys())
    n_items = len(items)

    series = list(data[items[0]].columns)
    n_series = len(series)

    col_params = get_params(_col_params, series, 'col_params', **kwargs)
    plt_params = get_params(_plt_params, None, 'plt_params', **kwargs)
    fig_params = get_params(_fig_params, None, 'fig_params', **kwargs)

    data = {k: ts_range(d, dt_min=plt_params['dt_min'], dt_max=plt_params['dt_max']) for k, d in data.items()}

    col_params = parse_col_params(
        col_params,
        data,
        disp=plt_params['disp'],
        share=fig_params['share'],
        margin=plt_params['margin']
    )

    fig, axs = create_figure(n=n_items, **fig_params)

    for i, item in enumerate(items):
        item_col_params = dict(col_params)
        item_plt_params = dict(plt_params) | dict(legend=False, title=item, note=None)
        item_fig_params = dict(fig_params) | dict(figsize=None)

        plot_series(
            data[item],
            ax=axs[i],
            col_params=item_col_params,
            plt_params=item_plt_params,
            fig_params=item_fig_params
        )

    if plt_params['legend'] and n_series > 1:
        yaxis_arr = np.unique([p['yaxis'] for p in col_params.values()])
        lh = {
            label: handle for ax in fig.axes for handle, label in zip(*ax.get_legend_handles_labels())
        }

        if len(yaxis_arr) > 1:
            fig.add_artist(fig.legend(
                handles=[lh.get(col_params[s]['label']) for s in series if col_params[s]['yaxis'] == 'right'],
                labels=[col_params[s]['label'] for s in series if col_params[s]['yaxis'] == 'right'],
                loc='upper right',
                ncol=plt_params['leg_cols'] or 4
            ))
            fig.legend(
                handles=[lh.get(col_params[s]['label']) for s in series if col_params[s]['yaxis'] == 'left'],
                labels=[col_params[s]['label'] for s in series if col_params[s]['yaxis'] == 'left'],
                loc='upper left',
                ncol=plt_params['leg_cols'] or 4
            )
        else:
            fig.legend(
                handles=lh.values(),
                labels=lh.keys(),
                loc='upper left',
                ncol=plt_params['leg_cols'] or 4
            )

    set_title(plt_params['title'])
    set_note(plt_params['note'], size='large')

    adjust_figure(fig=fig, n=n_items, **fig_params)

    plot_figure(show=show, path=path, fig=fig)


def plot_histogram_grid(data, show=True, path=None, **kwargs):
    # stat -> count (default), probability
    # outliers -> keep (default), group or remove
    _plt_params = dict(
        stat='count', bins=None, weights=None, discrete=False, outliers='keep', ol_std=3,
        show_bars=True, show_kde=False, show_mean=False, show_std=False,
        xlim=None, xmin=None, xmax=None, xticks=None, ylim=None, ymin=None, ymax=None, yticks=None,
        grid=True, cm=None, legend=True, title=None, note=None
    )
    _fig_params = dict(
        n_cols=2, share=False, figsize=(20, 7)
    )

    items = list(data.columns)
    n_items = len(items)

    plt_params = get_params(_plt_params, None, 'plt_params', **kwargs)
    fig_params = get_params(_fig_params, None, 'fig_params', **kwargs)

    fig, axs = create_figure(n=n_items, **fig_params)

    for i, item in enumerate(items):
        item_plt_params = dict(plt_params) | dict(title=item, note=None)
        item_fig_params = dict(fig_params) | dict(figsize=None)

        plot_histogram(
            data[item],
            ax=axs[i],
            plt_params=item_plt_params,
            fig_params=item_fig_params
        )

    set_title(plt_params['title'])
    set_note(plt_params['note'], size='large')

    adjust_figure(fig=fig, n=n_items, **fig_params)

    plot_figure(show=show, path=path, fig=fig)


def plot_regression_grid(x, y, ax=None, show=True, path=None, **kwargs):
    _plt_params = dict(
        show_reg=False, show_kde=False, show_diag=False, show_quad=False,
        weights=None, diag_cm=None, quad_cm=None,
        xlim=None, xticks=None, xfmt=None, xlog=False, ylim=None, yticks=None, yfmt=None, ylog=False,
        annot=False, annot_loc=None, s=50, labels=True, vlines=None, hlines=None,
        grid=True, cm=None, alpha=None, legend=True, title=None, note=None
    )
    _reg_params = dict(
        reg_type='ls', poly_deg=1, cov_type='hac', reg_kernel='gaussian',
        reg_bw_type='fixed', reg_bw='isj', reg_alpha=0.05, reg_cm=None, reg_lw=2
    )
    _kde_params = dict(
        kde_kernel='gaussian', kde_bw_type='fixed', kde_bw='scott',
        kde_fill=True, outliers='keep', ol_std=3, kde_cm=None, kde_lw=2
    )
    _fig_params = dict(
        figsize=(20, 10)
    )

    plt_params = get_params(_plt_params, None, 'plt_params', **kwargs)
    reg_params = get_params(_reg_params, None, 'reg_params', **kwargs)
    kde_params = get_params(_kde_params, None, 'kde_params', **kwargs)
    fig_params = get_params(_fig_params, None, 'fig_params', **kwargs)

    fig, axs = create_figure(n=2, n_cols=2, **fig_params)

    item_plt_params = dict(plt_params) | dict(title=None, note=None)
    item_reg_params = dict(reg_params)
    item_kde_params = dict(kde_params)
    item_fig_params = dict(fig_params) | dict(figsize=None)

    plot_scatter(
        x, y,
        ax=axs[0],
        plt_params=item_plt_params,
        reg_params=item_reg_params,
        kde_params=item_kde_params,
        fig_params=item_fig_params
    )

    xlim = axs[0].get_xlim()
    xdiff = np.diff(axs[0].get_xticks())[0] * 2
    xticks = np.arange(0, xlim[1] + xdiff, xdiff)

    item_plt_params['xlim'] = xlim
    item_plt_params['xticks'] = xticks

    ylim = axs[0].get_ylim()
    ydiff = np.diff(axs[0].get_yticks())[0] * 2
    yticks = np.arange(0, ylim[1] + ydiff, ydiff)

    item_plt_params['ylim'] = ylim
    item_plt_params['yticks'] = yticks

    plot_diverging(
        x, y,
        yaxis='right',
        ax=axs[1],
        plt_params=item_plt_params,
        fig_params=item_fig_params
    )

    set_title(plt_params['title'])
    set_note(plt_params['note'], size='large')

    adjust_figure(fig=fig, n=2, n_cols=2, **fig_params)
    plot_figure(show=show, path=path, fig=fig)


def plot_missing_grid(data, show=True, path=None, **kwargs):
    _plt_params = dict(
        orientation='top', drop_empty=True, show_any=False,
        xmin=None, xmax=None, xticks=None, grid=True, cm=None,
        corr_method='pearson', linkage_method='single',
        title=None, note=None
    )
    _fig_params = dict(
        figsize=(20, 10)
    )

    df = pd.DataFrame(data).copy()

    plt_params = get_params(_plt_params, None, 'plt_params', **kwargs)
    fig_params = get_params(_fig_params, None, 'fig_params', **kwargs)

    x = df.isnull()

    fig, axs = create_figure(n=2, n_cols=2, **fig_params)

    plot_prevalence(
        x,
        orientation=plt_params['orientation'],
        drop_empty=plt_params['drop_empty'],
        show_any=plt_params['show_any'],
        vmin=plt_params['xmin'],
        vmax=plt_params['xmax'],
        vticks=plt_params['xticks'],
        grid=plt_params['grid'],
        cm=plt_params['cm'],
        ax=axs[0]
    )

    c = x.any()
    x = x[c[c].index]

    plot_corr_dendogram(
        x,
        orientation=plt_params['orientation'],
        corr_method=plt_params['corr_method'],
        linkage_method=plt_params['linkage_method'],
        cm=plt_params['cm'],
        ax=axs[1]
    )

    set_title(plt_params['title'])
    set_note(plt_params['note'], size='large')

    adjust_figure(fig=fig, n=2, n_cols=2, share=False, **fig_params)

    plot_figure(show=show, path=path, fig=fig)


def plot_collinearity_grid(data, ax=None, show=True, path=None, **kwargs):
    """
    # Feature collinearity analysis visualization using both methods above
    # Shows a list of features that can be linearly predicted from the others
    # Left: Eigen Values (values below threshold) | Right: Variance Inflation Facor [VIF] (values over threshold)
    """

    _plt_params = dict(
        eig_thr=0.1, vif_thr=5, sort=False,
        grid=True, title=None, note=None
    )
    _fig_params = dict(
        figsize=None
    )

    # Collinearity
    w, v, det, cond = feature_collinearity(data)

    # VIF
    vif = feature_vif(data)

    plt_params = get_params(_plt_params, None, 'plt_params', **kwargs)
    fig_params = get_params(_fig_params, None, 'fig_params', **kwargs)

    if fig_params['figsize'] is None:
        fig_params['figsize'] = (20, data.shape[1] // 3)

    fig, axs = create_figure(n=2, n_cols=2, **fig_params)

    item_plt_params = dict(plt_params) | dict(note=None)
    item_fig_params = dict(fig_params) | dict(figsize=None)

    if plt_params['eig_thr'] is not None:
        eig_ = w.where(w >= plt_params['eig_thr'], -w)
        eig_abs_val = True
    else:
        eig_ = w.copy()
        eig_abs_val = False

    if plt_params['vif_thr'] is not None:
        vif_ = vif.where(vif <= plt_params['vif_thr'], -vif)
        vif_abs_val = True
    else:
        vif_ = vif.copy()
        vif_abs_val = False

    plot_scores(
        eig_,
        abs_val=eig_abs_val,
        plt_params=item_plt_params,
        fig_params=item_fig_params,
        ax=axs[0]
    )
    plot_scores(
        vif_,
        abs_val=vif_abs_val,
        plt_params=item_plt_params,
        fig_params=item_fig_params,
        ax=axs[1]
    )

    if eig_abs_val:
        axs[0].axvline(plt_params['eig_thr'], color='k', ls='--', lw=1, alpha=0.5)
    if vif_abs_val:
        axs[1].axvline(plt_params['vif_thr'], color='k', ls='--', lw=1, alpha=0.5)

    axs[0].set_title('Correlation Matrix Eigen Values [Determinant: {:0.3f} | Condition: {:0.3f}]'.format(det, cond))
    axs[1].set_title('Variance Inflation Factor (VIF) [Mean: {:0.3f} | Max: {:0.3f}]'.format(np.mean(vif), np.max(vif)))

    set_title(plt_params['title'])
    set_note(plt_params['note'], size='large')

    adjust_figure(fig=fig, n=2, n_cols=2, **fig_params)
    plot_figure(show=show, path=path, fig=fig)


def plot_feature_rank(x, y, return_df=False, ax=None, show=True, path=None, **kwargs):
    _plt_params = dict(
        show_cv=False, abs_val=False, sort=False, cv=None, estimator=None, scoring=None,
        fit_params=None, seed=None, corr_method='pearson', p_value_thr=0.05,
        grid=True, note=None
    )
    _fig_params = dict(
        figsize=None
    )

    if isinstance(y, str):
        y = x[y]
        x = x.drop(columns=[y])

    x = to_pandas(x, pdtype='DataFrame')
    y = to_pandas(y, pdtype='Series')

    plt_params = get_params(_plt_params, None, 'plt_params', **kwargs)
    fig_params = get_params(_fig_params, None, 'fig_params', **kwargs)

    names = list(x.columns)
    rank_cols = ['fcorr', 'flreg']
    if plt_params['show_cv']:
        rank_cols += ['fimp', 'fshap']

    df = pd.DataFrame(
        columns=rank_cols,
        index=names,
        dtype=float
    )

    # Feature/Target Univariate Correlation
    fcorr = pd.DataFrame(
        feature_correlations(x, y, method=plt_params['corr_method']).rename('fcorr')
    )
    # Perform T-test on each feature and mask features with high p-value
    fcorr['mask'] = feature_stat_test(x, y) < plt_params['p_value_thr']
    df['fcorr'] = fcorr.fcorr

    # Feature/Target Multivariate Regression
    flreg = feature_regression(x, y)
    # Normalize Odds Ratio and it's limits in order to visualize the contribution of each feature
    flreg[['odds_ratio', 'or_lo', 'or_up']] = flreg[['odds_ratio', 'or_lo', 'or_up']] - 1
    flreg['or_lo'] = flreg.odds_ratio - flreg.or_lo
    flreg['or_up'] = flreg.or_up - flreg.odds_ratio
    # Mask features with high regression p-value
    flreg['mask'] = flreg.p_value < plt_params['p_value_thr']
    df['flreg'] = flreg.odds_ratio

    if plt_params['show_cv']:
        # Model CV
        cv_model = cv_model_train(
            x, y,
            cv=plt_params['cv'],
            estimator=plt_params['estimator'],
            scoring=plt_params['scoring'],
            fit_params=plt_params['fit_params'],
            seed=plt_params['seed']
        )

        # Model Feature Importances
        fimp = [i.feature_importances_ for i in cv_model['estimator']]
        fimp = pd.DataFrame([
            np.mean(fimp, axis=0),
            np.mean(fimp, axis=0) - np.min(fimp, axis=0),
            np.max(fimp, axis=0) - np.mean(fimp, axis=0)
        ], columns=x.columns).T
        fimp = fimp.where((fimp[0] == 0) | (fcorr['fcorr'] > 0), -fimp)
        df['fimp'] = fimp[0]

        # Shap Values
        fshap = [np.mean(np.abs(shap_values(x, i)), axis=0) for i in cv_model['estimator']]
        fshap = pd.DataFrame([
            np.mean(fshap, axis=0),
            np.mean(fshap, axis=0) - np.min(fshap, axis=0),
            np.max(fshap, axis=0) - np.mean(fshap, axis=0)
        ], columns=x.columns).T
        fshap = fshap.where((fshap[0] == 0) | (fcorr['fcorr'] > 0), -fshap)
        df['fshap'] = fshap[0]

    n = df.shape[1]
    if fig_params['figsize'] is None:
        fig_params['figsize'] = (20, x.shape[1] // 3)

    fig, axs = create_figure(n=n, n_cols=2, **fig_params)

    item_plt_params = dict(plt_params) | dict(note=None)
    item_fig_params = dict(fig_params) | dict(figsize=None)

    plot_scores(
        fcorr[['fcorr']],
        alpha=fcorr['mask'].map({True: 1, False: 0.1}),
        plt_params=item_plt_params,
        fig_params=item_fig_params,
        ax=axs[0]
    )
    axs[0].set_title('Univariate Correlation')

    plot_scores(
        flreg[['odds_ratio', 'or_lo', 'or_up']],
        alpha=flreg['mask'].map({True: 1, False: 0.1}),
        plt_params=item_plt_params,
        fig_params=item_fig_params,
        ax=axs[1]
    )
    axs[1].set_title('Multivariate Regression')

    if plt_params['show_cv']:
        plot_scores(
            fimp,
            plt_params=item_plt_params,
            fig_params=item_fig_params,
            ax=axs[2]
        )
        axs[2].set_title('ML Feature Importances')

        plot_scores(
            fshap,
            plt_params=item_plt_params,
            fig_params=item_fig_params,
            ax=axs[3]
        )
        axs[3].set_title('ML Shap Values')

    set_note(plt_params['note'], size='large')

    adjust_figure(fig=fig, n=n, n_cols=2, **fig_params)
    plot_figure(show=show, path=path, fig=fig)

    if return_df:
        if plt_params['show_cv']:
            return df, cv_model

        return df


def plot_geo_heatmap_grid(data, geo_map,
    n_cols=2,
    labels=None, vmin=None, vmax=None, cmap='OrRd', scheme=None,
    figsize=(10, 10), legend=False, title=None, note=None, show=True, path=None
):
    keys = list(data.keys())
    n_keys = len(keys)

    n_rows = int(np.ceil(n_keys / n_cols))
    width = figsize[0]
    height = figsize[1] * n_rows

    fig, ax = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(width, height), constrained_layout=True)
    axes = ax.flatten()

    for i, key in enumerate(keys):
        if isinstance(labels, (dict, pd.DataFrame)) and key in labels.keys():
            labels_ = labels[key]
        else:
            labels_ = labels

        if isinstance(vmin, dict) and key in vmin.keys():
            vmin_ = vmin[key]
        else:
            vmin_ = vmin

        if isinstance(vmax, dict) and key in vmax.keys():
            vmax_ = vmax[key]
        else:
            vmax_ = vmax

        plot_geo_heatmap(
            data[key], geo_map,
            labels=labels_, vmin=vmin_, vmax=vmax_, cmap=cmap, scheme=scheme,
            legend=legend, title=key, ax=axes[i]
        )

    set_title(
        title,
        y=1.05 if n_rows > 1 else 0.95
    )
    set_note(
        note,
        x=0.02,
        y=0 if n_rows > 1 else 0.08,
        size='large'
    )

    plot_figure(show=show, path=path, fig=fig)


def plot_clusters_grid(data, labels, pos=None, centroids=None, show=True, path=None, **kwargs):
    _plt_params = dict(
        s=30, cm=None
    )
    _fig_params = dict(
        figsize=(20, 10)
    )

    data = to_pandas(data, pdtype='DataFrame')
    labels = to_pandas(labels, pdtype='Series')

    if pos is not None:
        pos = to_pandas(pos, pdtype='DataFrame')
    else:
        pos = data.copy()
    
    if centroids is not None:
        centroids = to_pandas(centroids, pdtype='DataFrame')

    clustered = labels >= 0
    outliers = labels < 0

    names = labels.unique().tolist()

    n_names = len(names)
    n_nodes = len(labels)
    ratio = len(labels[clustered]) / len(labels)

    plt_params = get_params(_plt_params, None, 'plt_params', **kwargs)
    fig_params = get_params(_fig_params, None, 'fig_params', **kwargs)

    cmap = labels.map(
        get_color_map(cm=plt_params['cm'], keys=names, rtype='dict')
    )

    fig, axs = create_figure(n=2, n_cols=2, **fig_params)

    axs[0].set_xlim([-0.1, 1])
    axs[0].set_ylim([0, len(data[clustered])])

    score, samples = clusters_silhouette(data[clustered], labels[clustered])

    # Silhouette graph
    y_lower = 0
    for i in names:
        ith = samples[labels[clustered] == i]
        ith.sort()
        color = cmap[labels[clustered] == i]

        size = ith.shape[0]
        y_upper = y_lower + size

        axs[0].fill_betweenx(
            np.arange(y_lower, y_upper), 0, ith,
            facecolor=color, edgecolor=color, alpha=0.7
        )
        axs[0].text(
            -0.05, y_lower + 0.5 * size, str(i),
            color='k', weight='semibold', ha='center', va='center', alpha=1,
            bbox=dict(edgecolor='k', facecolor='w', boxstyle='circle, pad=0.4')
        )
        y_lower = y_upper

    axs[0].set_title('Silhouette')
    axs[0].axvline(x=score, color='red', ls='--')

    axs[0].set_yticks(np.linspace(0, len(data), num=11).astype(int))
    axs[0].set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
    axs[0].set_xticklabels([
        '{}%'.format(format_number(t * 100)) if t >= 0 else ''
        for t in axs[0].get_xticks()
    ])

    if pos is not None and pos.shape[1] == 2:
        # Plot data (color clustered)
        axs[1].scatter(
            pos[clustered].values[:, 0], pos[clustered].values[:, 1],
            s=plt_params['s'], c=cmap, alpha=0.7
        )

        # Plot cluster centroids
        if centroids is not None:
            for i, r in centroids.iterrows():
                axs[1].text(
                    r[0], r[1], str(i),
                    color='k', weight='semibold', ha='center', va='center', alpha=1,
                    bbox=dict(edgecolor='k', facecolor='w', boxstyle='circle, pad=0.4')
                )

        # Plot outliers (dark)
        axs[1].scatter(
            pos[outliers].values[:, 0], pos[outliers].values[:, 1],
            s=plt_params['s'], c='k', alpha=0.7
        )

        axs[1].set_yticks([])
        axs[1].set_xticks([])

        axs[1].set_title('Clusters')
    else:
        axs[1].set_axis_off()

    set_title('Sample: {} | Clusters: {} | Clustered: {}% | Score: {}'.format(
        format_number(n_nodes),
        format_number(n_names),
        format_number(100 * ratio, 0),
        '{}%'.format(format_number(100 * score, 0))
    ))

    adjust_figure(fig=fig, n=2, n_cols=2, **fig_params)
    plot_figure(show=show, path=path, fig=fig)


# Images

def plot_image(image, ax=None, show=True, path=None, **kwargs):
    _plt_params = dict(
        title=None, note=None
    )
    _fig_params = dict(
        figsize=(16, 9)
    )

    plt_params = get_params(_plt_params, None, 'plt_params', **kwargs)
    fig_params = get_params(_fig_params, None, 'fig_params', **kwargs)

    fig, ax = create_figure(ax=ax, **fig_params)

    ax.imshow(image)

    ax.set_xticks([])
    ax.set_yticks([])

    set_title(plt_params['title'], ax=ax)
    set_note(plt_params['note'], ax=ax)

    plot_figure(show=show, path=path, fig=fig)


def plot_image_grid(images, show=True, path=None, **kwargs):
    _plt_params = dict(
        title=None, note=None
    )
    _fig_params = dict(
        n_cols=2, figsize=(16, 9)
    )

    if isinstance(images, (list, tuple, np.ndarray)):
        images = {str(i): img for i, img in enumerate(images)}
    elif not isinstance(images, dict):
        images = {'': images}

    items = list(images.keys())
    n_items = len(items)

    plt_params = get_params(_plt_params, None, 'plt_params', **kwargs)
    fig_params = get_params(_fig_params, None, 'fig_params', **kwargs)

    fig, axs = create_figure(n=n_items, **fig_params)

    for i, item in enumerate(items):
        item_plt_params = dict(plt_params) | dict(title=item, note=None)
        item_fig_params = dict(fig_params) | dict(figsize=None)

        plot_image(
            images[item],
            ax=axs[i],
            plt_params=item_plt_params,
            fig_params=item_fig_params
        )

    set_title(plt_params['title'])
    set_note(plt_params['note'], size='large')

    adjust_figure(fig=fig, n=n_items, **fig_params)
    plot_figure(show=show, path=path, fig=fig)


# Tables


def print_feature_split(x, y, return_df=False, show=True, path=None, **kwargs):
    """
    Create a table of values for every feature to be filled by each hypotesis test
    """

    _plt_params = dict(
        show_stats=False, max_dec=2, grad_cm=None, sort=None,
        title=None, note=None
    )

    plt_params = get_params(_plt_params, None, 'plt_params', **kwargs)

    if isinstance(y, str):
        y = x[y]
        x = x.drop(columns=[y])

    x = to_pandas(x, pdtype='DataFrame')
    y = to_pandas(y, pdtype='Series')

    names = list(x.columns)
    splits = y.value_counts().to_dict()
    keys = [('missing', 'count')] + [('{} ({})'.format(s, n), f) for s, n in splits.items() for f in ['mean', 'std']]

    df = pd.DataFrame(
        columns=pd.MultiIndex.from_tuples(keys),
        index=names,
        dtype=float
    )

    for name in names:
        row = [
            x[x[name].isnull()].shape[0]  # Missing
        ]

        for s in splits:
            row.extend(
                x[y == s][name].agg(['nanmean', 'nanstd']).values
            )

        df.loc[name] = row

    test_ind = None
    reg_mul = None

    if plt_params['show_stats']:
        try:
            x_, _ = to_tensor(x, scale=True, impute=True)
            y_ = y.map(dict(zip(splits, np.arange(len(splits))))) if not is_numeric(y) else y.copy()

            test_ind = feature_stat_test(x_, y_)
            df.loc[:, ('test_ind', 'p_value')] = test_ind.values

            reg_mul = feature_regression(x_, y_)
            df.loc[:, ('reg_mul', 'p_value')] = reg_mul.p_value.values
            df.loc[:, ('reg_mul', 'odds_ratio')] = reg_mul.odds_ratio.values
        except Exception:
            pass

    sort = plt_params['sort']
    if sort == 'index':
        df = df.sort_index()
    elif sort is not None and sort in df.columns.get_level_values(0):
        df = df.sort_values((sort, df[sort].columns[0]))

    grad_cmap = plt_params['grad_cm'] or 'blue'
    if not isinstance(grad_cmap, mcolors.Colormap):
        grad_cmap = build_color_cat_map([
            '{}{}'.format(grad_cmap, suf) for suf in ['-bolder', '-bold', '', '-light', '-alpha']
        ], n=10, gamma=0.3)

    gradients = []
    bars = [
        {
            'color': 'purple-light',
            'subset': [('missing', 'count')],
            'vmin': 0,
            'vmax': x.shape[0]
        }
    ]

    if plt_params['show_stats']:
        if test_ind is not None:
            gradients.append({
                'cmap': grad_cmap,
                'subset': [('test_ind', 'p_value')],
                'vmin': 0,
                'vmax': 0.5
            })

        if reg_mul is not None:
            gradients.append({
                'cmap': grad_cmap,
                'subset': [('reg_mul', 'p_value')],
                'vmin': 0,
                'vmax': 0.5
            })
            bars.append({
                'color': ['red-light', 'green-light'],
                'subset': [('reg_mul', 'odds_ratio')],
                'align': 1,
                'vmin': 0,
                'vmax': 2
            })

    dfs = get_df_styler(
        df,
        max_dec=plt_params['max_dec'],
        gradients=gradients,
        bars=bars,
        title=plt_params['title'],
        note=plt_params['note']
    )

    print_styler(dfs=dfs, show=show, path=path)

    if return_df:
        return df


# Legacy


def plot_regression(x, y, ax=None, show=True, path=None, **kwargs):
    return plot_scatter(
        x,
        y,
        ax=ax,
        show=show,
        path=path,
        **kwargs
    )


def plot_feature_matrix(x, num=False, triang=True, ax=None, show=True, path=None):
    return plot_corr_matrix(
        x,
        annot=num,
        triang=triang,
        ax=ax,
        show=show,
        path=path
    )


def plot_feature_usages(x, features=None, drop_filled=False, ax=None, show=True, path=None):
    return plot_missing_grid(
        x[features or x.columns],
        drop_empty=drop_filled,
        ax=ax,
        show=show,
        path=path
    )


def plot_feature_scores(x, features=None, abs_val=False, sort=False, num_feat=None, xlim=None, ax=None, show=True, path=None):
    return plot_scores(
        x.loc[features or x.index],
        abs_val=abs_val,
        sort=sort,
        ax=ax,
        show=show,
        path=path
    )


def plot_feature_collinearity(x, eig_thr=None, vif_thr=None, sort=False, show=True, path=None):
    return plot_collinearity_grid(
        x,
        eig_thr=eig_thr,
        vif_thr=vif_thr,
        sort=sort,
        show=show,
        path=path
    )


def get_axes_list(ax, nrows, ncols):
    if nrows == 1 and ncols == 1:
        axes = [ax]
    elif nrows == 1 or ncols == 1:
        axes = [i for i in ax]
    else:
        axes = [i for rows in ax for i in rows]

    return axes


def get_color_cmap(c):
    if c is not None:
        c_num = len(sorted(set(c)))
        if c_num == 1:
            cm = 'g'
        elif c_num == 2:
            cm = c.map({0: 'r', 1: 'g'})
        else:
            cm = plt.cm.get_cmap('Spectral')(c.astype(float) / c_num)
    else:
        cm = 'b'

    return cm


def plot_stack(x, legend=True, tmask=None, ax=None, show=True, path=None):
    ncols = x.shape[1]

    labels = x.index.astype(str)
    columns = x.columns
    values = x.values

    colors = plt.get_cmap('Spectral')(np.linspace(0.85, 0.15, ncols))
    tmask = [tmask[i] if tmask is not None and len(tmask) > i else None for i in np.arange(ncols)]

    if ax is None:
        fig, ax = plt.subplots(figsize=(20, 12))
    else:
        fig = None

    for i, (column, color) in enumerate(zip(columns, colors)):
        widths = values[:, i]
        starts = values.cumsum(axis=1)[:, i] - widths
        ax.barh(labels, widths, left=starts, label=column, color=color)

        locs = starts + widths / 2
        r, g, b, _ = color
        c = 'white' if r * g * b < 0.5 else 'darkgrey'

        for j, (loc, width) in enumerate(zip(locs, widths)):
            if tmask[i] is not None:
                ax.text(loc, labels[j], tmask[i].format(width), ha='center', va='center', size='large', color=c)

    ax.invert_yaxis()
    ax.xaxis.set_visible(False)
    ax.set_xlim(0, np.sum(values, axis=1).max())

    if legend:
        ax.legend(ncol=ncols, bbox_to_anchor=(0, 1), loc='lower left')

    plot_figure(show=show, path=path, fig=fig)


def plot_surv_lines(x, y=None, xlabel=None, ylabel=None, n_span=None, lw=2, ax=None, show=True, path=None):
    n_span = n_span or x.shape[0]

    if ax is None:
        fig, ax = plt.subplots(figsize=(20, 12))
    else:
        fig = None

    x = x[x.index <= n_span]
    xticks = list(x.index)

    ax.plot(x, color='g', label=xlabel, lw=lw)

    if y is not None:
        y = y[y.index <= n_span]

        ax.plot(y, color='b', label=ylabel, lw=lw)
        ax.fill_between(xticks, x, y, where=y > x, color='g', alpha=0.2)
        ax.fill_between(xticks, x, y, where=y < x, color='r', alpha=0.2)

        ymin = math.floor(np.min([x.min(), y.min()]) / 10) * 10
        ymax = math.ceil(np.max([x.max(), y.max()]) / 10) * 10
    else:
        ymin = math.floor(x.min() / 10) * 10
        ymax = math.ceil(x.max() / 10) * 10

    ax.set_ylim([ymin - 5, ymax + 5])
    ax.set_yticks(range(ymin, ymax + 1, 10))
    ax.set_xticks(xticks)
    ax.grid(True, ls='--')

    if xlabel is not None or ylabel is not None:
        ax.legend(ncol=2, bbox_to_anchor=(0, 1), loc='lower left')

    plot_figure(show=show, path=path, fig=fig)


def plot_cv_results(cv_model, ax=None, show=True, path=None):
    cv = len(cv_model['test_score'])
    ticks = np.arange(cv)
    scores = np.abs(cv_model['test_score'])

    mean = scores.mean()
    lims = [
        scores.min() * 0.95,
        scores.max() * 1.05
    ]

    if ax is None:
        fig, ax = plt.subplots(figsize=(20, 12))
    else:
        fig = None

    bars = ax.barh(ticks, scores, align='center', tick_label=ticks)

    ax.invert_yaxis()
    ax.grid(True, ls='--')
    ax.set_yticks(ticks)
    ax.set_xlim(lims)

    for bar in bars:
        width = bar.get_width()
        yloc = bar.get_y() + bar.get_height() / 2

        ax.annotate(
            '{:0.4f}'.format(width),
            xy=(lims[0], yloc),
            xytext=(20, 0),
            textcoords='offset points',
            va='center',
            color='w',
            weight='bold',
            clip_on=True
        )

    ax.axvline(mean, color='g', lw=3, alpha=0.5, label='Mean Score: {:0.4f}'.format(mean))

    ax.legend(loc='upper right')

    plot_figure(show=show, path=path, fig=fig)


def plot_som_umat(model, ax=None, show=True, path=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(20, 12))
    else:
        fig = None

    umat = model.distance_map()
    ax.imshow(umat.T, cmap='Purples')
    ax.invert_yaxis()

    plot_figure(show=show, path=path, fig=fig)


def plot_som_map(model, x, c=None, ax=None, show=True, path=None):
    x = np.array(x)

    if ax is None:
        fig, ax = plt.subplots(figsize=(20, 12))
    else:
        fig = None

    hmap = model.activation_response(x)
    wmax = 2 ** np.ceil(np.log(np.abs(hmap).max()) / np.log(2))

    if c is not None:
        c = np.array(c)

        lmap = model.labels_map(x, c)
        colors = plt.get_cmap('Spectral')(np.linspace(0.15, 0.85, 101))

    for p, w in np.ndenumerate(hmap):
        if w > 0:
            if c is not None:
                if p in lmap:
                    tgt = 100 - int(100 * lmap[p][0] / np.sum(list(lmap[p].values())))
                    color = colors[tgt]
                else:
                    color = colors[0]
            else:
                color = 'b'
        else:
            color = 'w'

        size = np.sqrt(np.abs(w) / wmax)
        rect = plt.Rectangle([p[0] - size / 2, p[1] - size / 2], size, size, facecolor=color, edgecolor=color)
        ax.add_patch(rect)

    ax.set_aspect('equal', 'box')
    ax.autoscale_view()

    plot_figure(show=show, path=path, fig=fig)


def plot_feature_grid(x, y=None, features=None, split=False, ncols=2, show=True, path=None):
    if features is None:
        features = list(x.columns)

    if y is not None:
        if isinstance(y, str):
            d = x[pd.notnull(x[y])]
            d[y] = [1 if i else 0 for i in d[y]]

            dx = d[d[y] == 0]
            dy = d[d[y] == 1]
        else:
            d = pd.concat([x, y])

            dx = x.copy()
            dy = y.copy()

        if split:
            features = np.array([[f + '_x', f + '_y'] for f in features]).flatten().tolist()
            d = pd.merge(dx, dy, left_index=True, right_index=True, how='left')[features]

            comp = False
        else:
            comp = True
    else:
        d = x.copy()

        comp = False

    categorical = get_columns(d, dtypes=['cat'])
    d.loc[:, categorical] = d.loc[:, categorical].astype(str).replace({'nan': None})

    nrows = int(math.ceil(float(len(features)) / float(ncols)))
    width = 20
    height = 6 * nrows

    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, sharex='none', sharey='none', figsize=(width, height))
    axes = get_axes_list(ax, nrows=nrows, ncols=ncols)

    for i, col in enumerate(features):
        s = d[pd.notnull(d[col])][col]
        if comp:
            sx = dx[pd.notnull(dx[col])][col]
            sy = dy[pd.notnull(dy[col])][col]

        h = s.value_counts(normalize=True) * 100

        if np.issubdtype(s.dtype, np.number) and h.size > 20:
            bins = np.histogram_bin_edges(s[np.abs(s - s.mean()) / s.std() < 3], bins='auto')
            if bins.size > h.size:
                bins = np.histogram_bin_edges(s[np.abs(s - s.mean()) / s.std() < 3], bins='doane')

            xticks = [(bins[i] + bins[i + 1]) / 2 for i in range(len(bins) - 1)]

            h, _ = np.histogram(s, bins=bins)
            h = pd.Series(h, index=xticks) * 100 / np.sum(h)

            if comp:
                hx, _ = np.histogram(sx, bins=bins)
                hx = pd.Series(hx, index=xticks) * 100 / np.sum(hx)
                hy, _ = np.histogram(sy, bins=bins)
                hy = pd.Series(hy, index=xticks) * 100 / np.sum(hy)

                axes[i].bar(hx.index, hx.values, width=np.diff(bins), color='r', alpha=0.5)
                axes[i].bar(hy.index, hy.values, width=np.diff(bins), color='g', alpha=0.5)
            else:
                axes[i].bar(h.index, h.values, width=np.diff(bins))
        else:
            labels = sorted(set(h.index))
            xticks = np.arange(len(labels))

            h = h.reindex(labels, fill_value=0).rename(index=dict(zip(labels, xticks)))

            axes[i].set_xticks(xticks)
            axes[i].set_xticklabels(labels)

            if comp:
                hx = sx.value_counts(normalize=True) * 100
                hx = hx.reindex(labels, fill_value=0).rename(index=dict(zip(labels, xticks)))
                hy = sy.value_counts(normalize=True) * 100
                hy = hy.reindex(labels, fill_value=0).rename(index=dict(zip(labels, xticks)))

                axes[i].bar(hx.index, hx.values, color='r', alpha=0.5)
                axes[i].bar(hy.index, hy.values, color='g', alpha=0.5)
            else:
                axes[i].bar(h.index, h.values)

        axes[i].set_title(col)

    fig.tight_layout()
    plot_figure(show=show, path=path, fig=fig)


def plot_pca(x, c=None, exp_var=None, s=50, show=True, path=None):
    cm = get_color_cmap(c)
    n = x.shape[1]

    if exp_var is not None:
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20, 6))
        ax = np.append(ax, ax[1].twinx())

        ax[0].scatter(x.values[:, 0], x.values[:, 1], marker='.', c=cm, s=s)
        ax[1].plot(exp_var, color='b', lw=2.5)
        ax[2].plot(exp_var.cumsum(), color='g', lw=2.5)

        ax[1].set_xticks(np.arange(n))
        ax[1].set_xticklabels(np.arange(1, n + 1))
        ax[1].set_ylim([0, exp_var.max() * 1.1])
        ax[1].set_yticklabels(['{:0.0f}%'.format(i * 100) for i in ax[1].get_yticks()])
        ax[2].set_ylim([0, 1.1])
        ax[2].set_yticklabels(['{:0.0f}%'.format(i * 100) for i in ax[2].get_yticks()])

        ax[1].grid(axis='x', ls='--')
        ax[2].grid(axis='y', ls='--')

        ax[0].set_title('PCA')
        ax[1].set_title('Explained Variance')
    else:
        fig, ax = plt.subplots(figsize=(20, 12))

        ax.scatter(x.values[:, 0], x.values[:, 1], marker='.', c=cm, s=s)

        ax.set_title('PCA')

    fig.tight_layout()
    plot_figure(show=show, path=path, fig=fig)


def plot_scatter_grid(grid, ncols=2, s=50, show=True, path=None):
    nrows = int(math.ceil(float(len(grid)) / float(ncols)))
    width = 20
    height = 6 * nrows

    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(width, height))
    axes = get_axes_list(ax, nrows=nrows, ncols=ncols)

    for i, item in enumerate(grid):
        cm = get_color_cmap(item[1])
        axes[i].scatter(item[0].values[:, 0], item[0].values[:, 1], marker='.', c=cm, s=s)
        axes[i].set_title(item[2])

    fig.tight_layout()
    plot_figure(show=show, path=path, fig=fig)


def plot_scatter_clusters(x, g, c=None, ncols=2, s=50, show=True, path=None):
    grid = []
    labels = sorted(set(g))

    for label in labels:
        xc = x[g == label]
        rx = xc.shape[0]

        if c is not None:
            yc = c[g == label]
            ry = int(yc.value_counts(normalize=True)[1] * 100)

            grid.append([xc, yc, 'Label: {} | Rows: {:,} | Target: {:d}%'.format(label, rx, ry)])
        else:
            yc = None

            grid.append([xc, yc, 'Labels: {} | Rows: {:,}'.format(label, rx)])

    plot_scatter_grid(grid, ncols=ncols, s=s, show=show, path=path)


def plot_surv_grid(x, column, n_tte=30, n_iters=10, step=1, n_best=None,
        tgt_col='tgt', tte_col=None, show=True, path=None):
    gt = {}
    for n in range(0, n_iters * step + 1, step):
        gt[n] = (x[column] > n).value_counts(normalize=True).sort_index(ascending=False) * 100

    x_gt = pd.DataFrame.from_dict(gt, orient='index', columns=[1, 0]).fillna(0)

    gtr = {}
    for n in range(0, n_iters * step + 1, step):
        gtr[n] = x[x[column] > n][tgt_col].value_counts(normalize=True).sort_index(ascending=False) * 100

    x_gtr = pd.DataFrame.from_dict(gtr, orient='index', columns=[1, 0]).fillna(0)

    ful = {}
    for n in range(0, n_iters * step + 1, step):
        ret_true = x[(x[column] > n) & (x[tgt_col] == 1)].shape[0]
        ret_false = x[(x[column] <= n) & (x[tgt_col] == 1)].shape[0]
        total = x[x[column] > n].shape[0] + ret_false

        ful[n] = [
            100 * ret_true / total,
            100 * ret_false / total,
            100 * (total - ret_true - ret_false) / total
        ]

    x_ful = pd.DataFrame.from_dict(ful, orient='index', columns=['Target True', 'Target False', ''])
    if n_best is None:
        n_best = int(x_ful.idxmax()[0])

    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(20, 12))
    axes = get_axes_list(ax, nrows=2, ncols=2)

    plot_stack(x_gt, legend=False, tmask=['{:0.1f}%'], ax=axes[0])
    plot_stack(x_gtr, legend=False, tmask=['{:0.1f}%'], ax=axes[1])
    plot_stack(x_ful, legend=True, tmask=['{:0.2f}%'], ax=axes[2])

    if tte_col is not None:
        x_ret = pd.Series([x[x[tte_col] >= i].shape[0] for i in range(0, n_tte + 1)]) * 100 / x.shape[0]
        x_coh = pd.Series([
            x[(x[tte_col] >= i) & (x[column] > n_best)
        ].shape[0] for i in range(0, n_tte + 1)]) * 100 / x[x[column] > n_best].shape[0]

        plot_surv_lines(
            x_ret[1:], x_coh[1:],
            xlabel='All Users',
            ylabel='{} > {:d}'.format(column, n_best),
            ax=axes[3]
        )
    else:
        axes[3].set_axis_off()

    axes[0].set_title('Users with {} greater than...'.format(column))
    axes[1].set_title('Targets with {} greater than...'.format(column))

    fig.tight_layout()
    plot_figure(show=show, path=path, fig=fig)


def plot_som_grid(model, x, c=None, show=True, path=None):
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20, 12))
    axes = get_axes_list(ax, nrows=1, ncols=2)

    plot_som_umat(model, ax=axes[0])
    plot_som_map(model, x, c, ax=axes[1])

    axes[0].set_title('U-Matrix')
    axes[1].set_title('Heat Map')

    plot_figure(show=show, path=path, fig=fig)


def plot_som_split(model, x, c, show=True, path=None):
    x = np.array(x)
    c = np.array(c)

    fig = plt.figure(figsize=(20, 12))

    lmap = model.labels_map(x, c)
    labels = np.unique(c)
    lnum = len(labels)

    shape = np.max(list(lmap.keys()), axis=0) + 1
    colors = plt.get_cmap('Spectral')(np.linspace(0.15, 0.85, lnum))

    grid = fig.add_gridspec(shape[1], shape[0])
    for p in lmap.keys():
        fracs = [lmap[p][i] for i in labels]

        i = shape[1] - p[1] - 1
        j = p[0]

        ax = fig.add_subplot(grid[i, j])
        ax.pie(fracs, colors=colors)

    plot_figure(show=show, path=path, fig=fig)
