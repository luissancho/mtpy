from datetime import datetime
import pandas as pd
import requests
from tqdm import tqdm


class CurrencyLayer(object):

    def __init__(self, params):
        self.params = params

    def _request(self, method, path, headers=None, params=None, data=None):
        headers = headers or {}
        params = params or {}
        data = data or {}

        url = '{}/{}'.format(self.params['url'], path)

        r = requests.request(method, url, headers=headers, params=params, data=data)

        if not r.ok:
            raise r.raise_for_status()

        response = r.json()

        return response

    def get_quotes(self, dt_at, source='USD', quotes=None):
        quotes = quotes or []

        if not isinstance(dt_at, datetime):
            dt_at = pd.to_datetime(dt_at)

        result = {}

        response = self._request(
            'get',
            'historical',
            headers={
                'Content-Type': 'application/json; charset=utf-8'
            },
            params={
                'access_key': self.params['key'],
                'date': dt_at.strftime('%Y-%m-%d'),
                'source': source
            }
        )

        if 'quotes' in response:
            result = response['quotes']

        ds = pd.Series(result).round(2).rename(lambda x: x.replace(source, '')).sort_index()

        if ds.shape[0] > 0 and len(quotes) > 0:
            ds = ds.loc[quotes]

        return ds

    def get_quotes_range(self, dt_from, dt_to=None, source='USD', quotes=None, verbose=0):
        quotes = quotes or []

        if not isinstance(dt_from, datetime):
            dt_from = pd.to_datetime(dt_from)

        if dt_to is None:
            dt_to = pd.to_datetime(datetime.now().replace(microsecond=0, second=0, minute=0, hour=0).date())
        elif not isinstance(dt_to, datetime):
            dt_to = pd.to_datetime(dt_to)

        columns = ['rated_at', 'currency', 'value']
        df = pd.DataFrame(columns=columns)
        dates = pd.date_range(start=dt_from, end=dt_to, freq='D')

        dates_ = tqdm(dates) if verbose > 0 else dates
        for dt_at in dates_:
            ds = self.get_quotes(
                dt_at,
                source=source,
                quotes=quotes
            ).rename('value').rename_axis('currency').reset_index()
            ds['rated_at'] = dt_at

            df = pd.concat([df, ds[columns]])

        if df.shape[0] > 0:
            df['rated_at'] = pd.to_datetime(df.rated_at)
            df['value'] = df.value.astype(float)
            df = df.set_index(['rated_at', 'currency']).sort_index()

        return df
