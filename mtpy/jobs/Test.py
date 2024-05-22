from ..core.worker import Job


class Test(Job):

    def run(self, msg='', **kwargs):
        self.alert(
            'Test job executed: "{}"'.format(msg)
        )
