import requests


class Pushover(object):

    def __init__(self, params):
        self.params = params

    def send(self, message, priority=0, user=None, token=None, subject=None, **kwargs):
        user = user or self.params['user']
        token = token or self.params['token']
        subject = subject or self.params['subject']

        if not user or not token:
            return

        url = '{}/messages.json'.format(self.params['url'])
        data = {
            'user': user,
            'token': token,
            'title': subject,
            'message': message,
            'priority': priority
        }

        for key, val in kwargs.items():
            data[key] = val

        r = requests.post(url, data=data)

        if not r.status_code == requests.codes.ok:
            raise r.raise_for_status()

        result = r.json()

        return result
