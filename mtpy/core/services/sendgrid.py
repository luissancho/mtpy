import json
import pandas as pd
from sendgrid import SendGridAPIClient


class SendGrid(object):

    def __init__(self, account, params):
        self.account = account
        self.params = params['accounts'][account]

        if 'page_limit' not in self.params:
            self.params['page_limit'] = 50

        self.client = SendGridAPIClient(
            self.params['apikey']
        ).client

    def get_newsletters(self, with_stats=False, limit=None):
        per_page = limit if limit is not None and limit < self.params['page_limit'] else self.params['page_limit']

        result = {}

        columns = [
            'id', 'name', 'categories', 'date', 'status'
        ]
        page_token = ''

        while True:
            response = self.client.marketing.singlesends.get(query_params={
                'page_size': per_page,
                'page_token': page_token
            })
            response = json.loads(response.body.decode('UTF-8'))

            for row in list(response['result']):
                result[row['id']] = [
                    row['id'],
                    row['name'],
                    row['categories'],
                    row['send_at'],
                    row['status']
                ]

            if 'next' in response['_metadata']:
                page_token = response['_metadata']['next'].split('=').pop().strip()
            else:
                break

        if with_stats:
            columns.extend([
                'requests', 'delivered', 'opens', 'clicks', 'unsubscribes'
            ])
            page_token = ''

            while True:
                response = self.client.marketing.stats.singlesends.get(query_params={
                    'page_size': per_page,
                    'page_token': page_token
                })
                response = json.loads(response.body.decode('UTF-8'))

                for row in list(response['results']):
                    if row['id'] in result:
                        result[row['id']].extend([
                            row['stats']['requests'],
                            row['stats']['delivered'],
                            row['stats']['unique_opens'],
                            row['stats']['unique_clicks'],
                            row['stats']['unsubscribes']
                        ])

                if 'next' in response['_metadata']:
                    page_token = response['_metadata']['next'].split('=').pop().strip()
                else:
                    break

        df = pd.DataFrame(
            result.values(),
            columns=columns
        )

        if df.shape[0] > 0:
            df['date'] = pd.to_datetime(df.date)
            df = df.sort_values('date', ignore_index=True)

        return df

    def get_report(self, limit=None):
        per_page = limit if limit is not None and limit < self.params['page_limit'] else self.params['page_limit']

        result = []

        newsletters = self.get_newsletters(limit=limit).set_index('id').rename_axis(None)
        params = {
            'aggregated_by': 'day',
            'timezone': 'UTC',
            'page_size': per_page
        }

        for nid, nls in newsletters.iterrows():
            response = self.client.marketing.stats.singlesends._(nid).get(query_params=params)
            response = json.loads(response.body.decode('UTF-8'))

            for row in list(response['results']):
                result.append([
                    nid,
                    nls['name'],
                    nls['categories'][0] if len(nls['categories']) > 0 else None,
                    row['aggregation'],
                    row['stats']['requests'],
                    row['stats']['delivered'],
                    row['stats']['unique_opens'],
                    row['stats']['unique_clicks'],
                    row['stats']['unsubscribes']
                ])

        df = pd.DataFrame(
            result,
            columns=[
                'id', 'name', 'category', 'date',
                'requests', 'delivered', 'opens', 'clicks', 'unsubscribes'
            ]
        )

        if df.shape[0] > 0:
            df['date'] = pd.to_datetime(df.date)
            df = df.sort_values(['date', 'id'], ignore_index=True)

        return df
