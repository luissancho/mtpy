from ..core.worker import SequenceJob

from ..core.utils.strings import to_camel


class Pipeline(SequenceJob):

    def run(self, select=None, action=None, chunks=None, **kwargs):
        if '.' in select:
            module, name = select.split('.')
            name = '{}.{}'.format(module, to_camel(name))
        else:
            name = to_camel(select)
        
        self.sequence = {
            module: [name]
        }

        return super().run(action=action, chunks=chunks, **kwargs)
