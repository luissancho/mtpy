from ..core.worker import SequenceJob


class Update(SequenceJob):

    sequence = {}
    default_action = 'update'
