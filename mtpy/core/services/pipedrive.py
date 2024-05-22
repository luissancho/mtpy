import pandas as pd
import requests


class Pipedrive(object):

    def __init__(self, params):
        self.params = params

        if 'page_limit' not in self.params:
            self.params['page_limit'] = 500

    def _request(self, method, path, headers=None, params=None, data=None):
        headers = headers or {}
        params = params or {}
        data = data or {}

        url = '{}/{}'.format(self.params['api_url'], path)
        params['api_token'] = self.params['token']

        r = requests.request(method, url, headers=headers, params=params, data=data)

        if not r.ok:
            raise r.raise_for_status()

        result = r.json()

        return result

    def get_results(self, resource, filter_id=None, start=0, limit=None):
        path = '{}'.format(resource)
        per_page = limit if limit is not None and limit < self.params['page_limit'] else self.params['page_limit']

        params = {}
        params['start'] = start
        params['limit'] = per_page
        if filter_id is not None:
            params['filter_id'] = filter_id

        results = []
        while True:
            result = self._request('get', path, params=params)

            if result['data'] is not None:
                results.extend(result['data'])

            params['start'] += per_page
            if not result['additional_data']['pagination']['more_items_in_collection'] or limit is not None:
                break

        return pd.DataFrame(results)

    def get(self, resource, id):
        path = '{}/{}'.format(resource, id)

        result = self._request('get', path)

        if result['data'] is None:
            return {}

        return result['data']

    def get_related(self, resource, id, name):
        path = '{}/{}/{}'.format(resource, id, name)

        result = self._request('get', path)

        if result['data'] is None:
            return {}

        if result['data'] is None:
            return pd.DataFrame.from_records({})

        return pd.DataFrame(result['data'])

    def create(self, resource, data):
        path = '{}'.format(resource)

        result = self._request('post', path, data=data)

        if result['data'] is None:
            return {}

        return result['data']

    def update(self, resource, id, data):
        path = '{}/{}'.format(resource, id)

        result = self._request('put', path, data=data)

        if result['data'] is None:
            return {}

        return result['data']

    def delete(self, resource, id):
        path = '{}/{}'.format(resource, id)

        self._request('delete', path)
