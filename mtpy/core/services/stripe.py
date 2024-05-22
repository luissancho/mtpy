from datetime import datetime
import pandas as pd
import stripe


class Stripe(object):

    def __init__(self, params):
        self.params = params

        if 'page_limit' not in self.params:
            self.params['page_limit'] = 100

        stripe.api_key = self.params['api_key']

        self.resource_map = {
            'payment_intents': 'PaymentIntent',
            'charges': 'Charge',
            'customers': 'Customer',
            'disputes': 'Dispute',
            'payouts': 'Payout',
            'refunds': 'Refund',
            'subscriptions': 'Subscription',
            'transactions': 'BalanceTransaction',
            'transfers': 'Transfer'
        }

    def _request(self, resource, dt_from=None, dt_to=None, limit=None):
        per_page = limit if limit is not None and limit < self.params['page_limit'] else self.params['page_limit']

        cfilter = {}

        if dt_from is not None:
            if not isinstance(dt_from, datetime):
                dt_from = pd.to_datetime(dt_from)
            cfilter['gte'] = int(dt_from.timestamp())

        if dt_to is not None:
            if not isinstance(dt_to, datetime):
                dt_to = pd.to_datetime(dt_to)
            cfilter['lt'] = int(dt_to.timestamp())
        
        params = {
            'limit': per_page
        }
        if len(cfilter) > 0:
            params['created'] = cfilter

        results = []
        starting_after = None

        while True:
            if starting_after is not None:
                params['starting_after'] = starting_after

            result = getattr(stripe, resource).list(**params)

            results.extend([
                row.to_dict() for row in result['data']
            ])
            
            if result['has_more'] and limit is None:
                starting_after = result['data'][-1]['id']
            else:
                break

        return results
    
    def get_results(self, name, dt_from=None, dt_to=None, limit=None, expand_cols=None):
        expand_cols = expand_cols or []

        results = self._request(self.resource_map[name], dt_from, dt_to, limit)

        df = pd.DataFrame(results)

        for col in expand_cols:
            df = pd.concat([
                df, pd.json_normalize(df[col].tolist()).add_prefix(col + '__')
            ], axis=1).drop(columns=[col])

        return df
