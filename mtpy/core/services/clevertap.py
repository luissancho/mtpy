from datetime import datetime
import json
import numpy as np
import pandas as pd
import requests
from time import sleep


class CleverTap(object):

    def __init__(self, params):
        self.params = params

        if 'page_limit' not in self.params:
            self.params['page_limit'] = 5000

        self.projects = list(self.params['projects'].keys())
        self.set_project()

    def set_project(self, project=None):
        self.project = project or self.projects[0]
        self.headers = {
            'X-CleverTap-Account-Id': self.params['projects'][self.project]['project_id'],
            'X-CleverTap-Passcode': self.params['projects'][self.project]['passcode'],
            'Content-Type': 'application/json; charset=utf-8'
        }

        return self

    def _request(self, method, path, headers=None, params=None, data=None):
        headers = headers or {}
        params = params or {}
        data = data or {}

        url = '{}/{}'.format(self.params['api_url'], path)

        r = requests.request(method, url, headers=headers, params=params, data=data)

        if not r.ok:
            raise r.raise_for_status()

        response = r.json()

        return response

    def get_user_count(self, event, dt_from, dt_to, filters=None):
        if not isinstance(dt_from, datetime):
            dt_from = pd.to_datetime(dt_from)
        if not isinstance(dt_to, datetime):
            dt_to = pd.to_datetime(dt_to)

        result = None

        response = self._request(
            'post',
            'counts/profiles.json',
            headers=self.headers,
            data=json.dumps({
                'event_name': event,
                'event_properties': filters,
                'from': int(dt_from.strftime('%Y%m%d')),
                'to': int(dt_to.strftime('%Y%m%d'))
            })
        )

        if response['status'] == 'partial':
            req_id = response['req_id']
            while response['status'] == 'partial':
                sleep(5)

                response = self._request(
                    'get',
                    'counts/profiles.json?req_id={}'.format(req_id),
                    headers=self.headers
                )

        if response['status'] == 'success':
            result = response['count']

        return result

    def get_event_count(self, event, dt_from, dt_to):
        if not isinstance(dt_from, datetime):
            dt_from = pd.to_datetime(dt_from)
        if not isinstance(dt_to, datetime):
            dt_to = pd.to_datetime(dt_to)

        result = None

        response = self._request(
            'post',
            'counts/events.json',
            headers=self.headers,
            data=json.dumps({
                'event_name': event,
                'from': int(dt_from.strftime('%Y%m%d')),
                'to': int(dt_to.strftime('%Y%m%d'))
            })
        )

        if response['status'] == 'partial':
            req_id = response['req_id']
            while response['status'] == 'partial':
                sleep(5)

                response = self._request(
                    'get',
                    'counts/events.json?req_id={}'.format(req_id),
                    headers=self.headers
                )

        if response['status'] == 'success':
            result = response['count']

        return result

    def get_user(self, user_id):
        result = None

        response = self._request(
            'get',
            'profile.json?identity={}'.format(user_id),
            headers=self.headers
        )

        if response['status'] == 'partial':
            req_id = response['req_id']
            while response['status'] == 'partial':
                sleep(5)

                response = self._request(
                    'get',
                    'profile.json?req_id={}'.format(req_id),
                    headers=self.headers
                )

        if response['status'] == 'success':
            result = response['record']

        return result

    def get_event(self, event, dt_from, dt_to, limit=None):
        per_page = limit if limit is not None and limit < self.params['page_limit'] else self.params['page_limit']

        if not isinstance(dt_from, datetime):
            dt_from = pd.to_datetime(dt_from)
        if not isinstance(dt_to, datetime):
            dt_to = pd.to_datetime(dt_to)

        response = self._request(
            'post',
            'events.json',
            headers=self.headers,
            params={
                'batch_size': per_page,
                'app': True,
                'events': False,
                'profile': False
            },
            data=json.dumps({
                'event_name': event,
                'from': int(dt_from.strftime('%Y%m%d')),
                'to': int(dt_to.strftime('%Y%m%d'))
            })
        )

        cursor = response['cursor']
        results = []
        while True:
            response = self._request(
                'get',
                'events.json?cursor={}'.format(cursor),
                headers=self.headers
            )

            if 'records' in response:
                results.extend(response['records'])

            cursor = response['next_cursor'] if 'next_cursor' in response else None
            if cursor is None or limit is not None:
                break

        return pd.DataFrame(results)

    def upload(self, records):
        d = []
        for sid, values in records.items():
            d.append({
                'type': 'profile',
                'identity': sid,
                'profileData': values
            })

        if len(d) > self.params['max_records']:
            n = len(d) // self.params['max_records'] + 1
            chunks = np.array_split(d, n)
        else:
            chunks = [np.array(d)]

        for c in chunks:
            data = json.dumps({
                'd': list(c)
            })

            self._request('post', 'upload', headers=self.headers, data=data)
