import numpy as np
import pandas as pd
from facebook_business.api import FacebookAdsApi
from facebook_business.adobjects.adaccount import AdAccount


class FacebookAds(object):

    def __init__(self, account, params):
        self.account = account
        self.params = params['accounts'][account]

        self.params['conversion_actions'] = params.get('conversion_actions', [])
        self.params['page_limit'] = params.get('page_limit', 1000)

        self.client = FacebookAdsApi.init(
            self.params['app_id'],
            self.params['app_secret'],
            self.params['access_token']
        )

    def get_report(self, dt_from, dt_to):
        result = []

        account = AdAccount('act_{}'.format(self.params['account_id']))

        fields = [
            'adset_id',
            'adset_name',
            'campaign_id',
            'campaign_name',
            'spend',
            'clicks',
            'actions',
            'action_values'
        ]
        params = {
            'time_range': {
                'since': dt_from.strftime('%Y-%m-%d'),
                'until': dt_to.strftime('%Y-%m-%d')
            },
            'time_increment': 1,
            'level': 'adset',
            'limit': self.params['page_limit']
        }

        while True:
            response = account.get_insights(
                fields=fields,
                params=params
            )

            for row in list(response):
                r = [
                    row.get('date_start', ''),
                    row.get('campaign_id', ''),
                    row.get('campaign_name', ''),
                    row.get('adset_id', ''),
                    row.get('adset_name', ''),
                    float(row.get('spend', 0)),
                    float(row.get('clicks', 0))
                ]
                for k in ['actions', 'action_values']:
                    r.append(
                        np.sum([
                            float(s.get('value', 0))
                            if s.get('action_type', '') in self.params['conversion_actions']
                            else 0.0
                            for s in row.get(k, [])
                        ])
                    )

                result.append(r)

            if not response.load_next_page():
                break

        df = pd.DataFrame(
            result,
            columns=[
                'date', 'campaign_id', 'campaign', 'group_id', 'group',
                'cost', 'clicks', 'conversions', 'value'
            ]
        )

        if df.shape[0] > 0:
            df['date'] = pd.to_datetime(df.date)
            df = df.sort_values(['date', 'campaign_id', 'group_id'], ignore_index=True)

        return df
