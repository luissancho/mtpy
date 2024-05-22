import smtplib

from ..app import Core


class Emailer(Core):

    def __init__(self, params):
        self.params = params

        self.client = None

    def connected(self) -> bool:
        return True if self.client is not None else False

    def connect(self, force=False):
        if self.connected():
            if force:
                self.disconnect()
            else:
                return self

        self.client = smtplib.SMTP(self.params['host'], self.params['port'])
        self.client.starttls()
        self.client.login(self.params['username'], self.params['password'])

        return self

    def disconnect(self):
        if self.connected():
            self.client.quit()
            self.client = None

        return

    def send(self, message, subject=None, from_addr=None, to_addrs=None):
        subject = subject or self.params['subject']
        from_addr = from_addr or self.params['from_addr']
        to_addrs = to_addrs or self.params['to_addrs']

        if not to_addrs:
            return
        elif isinstance(to_addrs, str):
            to_addrs = [addr.strip() for addr in to_addrs.split(',')]
        
        message = 'From: {}\nTo: {}\nSubject: {}\n\n{}'.format(
            from_addr, ', '.join(to_addrs), subject, message
        )

        self.connect()

        result = self.client.sendmail(from_addr, to_addrs, message)

        self.disconnect()

        return result
