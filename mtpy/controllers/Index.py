from ..core.api import Controller


class Index(Controller):

    async def index_action(self):
        return {
            'status': 'ok',
            'message': 'API Home'
        }
