from importlib import import_module
import json
import re
from urllib.parse import parse_qsl

from .app import Core
from .utils.strings import to_camel


class Api(Core):

    async def __call__(self, scope, receive, send):
        request = Request(scope, receive)
        response = Response(send)

        self.app.set('request', request)
        self.app.set('response', response)

        router = self.app.router

        await router.handle()
        await router.dispatch()


class Request(Core):

    def __init__(self, scope, receive):
        super().__init__()

        self.scope = scope
        self.receive = receive

        self.uri = scope['path'].lstrip('/')
        self.method = scope['method'].upper()
        self.request = self.get_request()

    def get_request(self):
        qs = self.scope['query_string']

        if isinstance(qs, bytes):
            qs = qs.decode('latin-1')

        return parse_qsl(qs)

    async def get_body(self):
        body = b''
        more_body = True

        while more_body:
            message = await self.receive()
            body += message.get('body', b'')
            more_body = message.get('more_body', False)

        return body

    async def get_raw_body(self):
        if not hasattr(self, '_raw'):
            body = await self.get_body()
            self._raw = body.decode('utf-8')

        return self._raw

    async def get_json_body(self):
        if not hasattr(self, '_json'):
            body = await self.get_raw_body()
            self._json = json.loads(body) if body != '' else {}

        return self._json


class Response(Core):

    status_codes = {
        200: 'OK',
        301: 'Moved Permanently',
        302: 'Found',
        400: 'Bad Request',
        401: 'Unauthorized',
        403: 'Forbidden',
        404: 'Not Found',
        500: 'Internal Server Error'
    }

    def __init__(self, send):
        super().__init__()

        self.send = send

        self.status_code = 200
        self.headers = {}
        self.content = b''

    @property
    def raw_headers(self):
        return [(k.lower().encode('latin-1'), v.encode('latin-1')) for k, v in self.headers.items()]

    def get_header(self, key):
        return self.headers[key]

    def set_header(self, key, value):
        if value is not None:
            self.headers[key] = value
        elif key in self.headers:
            del self.headers[key]

        return self

    def set_content_type(self, content_type, charset='utf-8'):
        if charset is not None:
            self.set_header('content-type', content_type + '; charset=' + charset)
        else:
            self.set_header('content-type', content_type)

        return self

    def set_status_code(self, status_code):
        self.status_code = status_code

        return self

    def set_content(self, content):
        if content is None:
            self.set_content_type('text/plain')
            self.content = b''
        elif isinstance(content, int):
            self.set_content_type('application/json')
            self.set_status_code(content)
            self.content = json.dumps({
                'status': 'error',
                'message': str(content) + ' ' + Response.status_codes[content]
            }).encode('utf-8')
        elif isinstance(content, dict):
            self.set_content_type('application/json')
            self.content = json.dumps(content).encode('utf-8')
        elif isinstance(content, str):
            self.set_content_type('text/plain')
            self.content = content.encode('utf-8')
        elif isinstance(content, bytes):
            self.set_content_type('text/plain')
            self.content = content
        else:
            self.set_content_type('application/json')
            self.set_status_code(500)
            self.content = json.dumps({
                'status': 'error',
                'message': 'Invalid content-type'
            }).encode('utf-8')

        self.set_header('content-length', str(len(self.content)))

        return self

    async def send_content(self, content):
        self.set_content(content)

        await self.send({
            'type': 'http.response.start',
            'status': self.status_code,
            'headers': self.raw_headers,
        })
        await self.send({
            'type': 'http.response.body',
            'body': self.content
        })


class Router(Core):

    def __init__(self):
        super().__init__()

        self.controller = None
        self.action = None
        self.params = {}

        self.routes = []
        self.not_found = []

        self.regex = r'{(([0-9a-z_\-]+)(:[0-9a-z_\-]+)?)}'
        self.alias_map = {
            '': r'[0-9a-z_\-]+',
            'str': r'[a-z]+',
            'num': r'[0-9]+',
            'loc': r'[a-z]{2}',
            'uri': r'.+'
        }

        self.namespace = import_module('...controllers', package=__name__)

    def set_namespace(self, namespace):
        if isinstance(namespace, str):
            self.namespace = import_module(namespace)
        else:
            self.namespace = namespace

        return self

    def add_route(self, pattern, controller, action='index', methods=None):
        methods = methods or ['GET', 'POST']

        params = []

        for match in re.compile(self.regex).finditer(pattern):
            group, name, ctype = match.groups('')
            ctype = ctype.lstrip(':')
            alias = self.alias_map[ctype]

            pattern = pattern.replace('{' + group + '}', '(?P<' + name + '>' + alias + ')')
            params.append(name)

        methods = set(method.upper() for method in methods)

        self.routes.append({
            'pattern': pattern,
            'params': params,
            'controller': controller,
            'action': action,
            'methods': methods
        })

        return self

    def add_not_found(self, pattern, controller, action='not_found'):
        self.not_found.append({
            'pattern': pattern,
            'controller': controller,
            'action': action
        })

        return self

    @staticmethod
    def parse_route(route, uri, method):
        if not method.upper() in route['methods']:
            return False

        pattern = re.compile('^' + route['pattern'] + '$')
        match = pattern.match(uri)

        if not match:
            return False

        params = match.groupdict()

        return params

    @staticmethod
    def check_not_found(route, uri):
        return uri.startswith(route['pattern'])

    async def handle(self, uri=None):
        method = self.app.request.method
        if uri is None:
            uri = '/' + self.app.request.uri

        for route in self.routes:
            params = Router.parse_route(route, uri, method)

            if params is not False:
                name = to_camel(route['controller'])
                self.controller = getattr(self.namespace, name)()
                self.action = route['action']
                self.params = params

                return self

        for route in self.not_found:
            if Router.check_not_found(route, uri):
                name = to_camel(route['controller'])
                self.controller = getattr(self.namespace, name)()
                self.action = route['action']

                return self

        self.controller = Controller()
        self.action = 'not_found'
        self.params = {}

    async def dispatch(self):
        if self.loaded():
            await self.controller.dispatch(self.action, **self.params)

    def loaded(self):
        return True if self.controller is not None else False


class Controller(Core):

    def __init__(self):
        super().__init__()

        self.request = self.app.request
        self.response = self.app.response

        self.action = None
        self.params = {}

        self.result = None

    async def dispatch(self, action, **kwargs):
        self.action = action
        self.params = dict(kwargs)

        self.before_dispatch()

        self.result = await getattr(self, action + '_action')(**self.params)

        self.after_dispatch()

        await self.send()

    def before_dispatch(self):
        pass

    def after_dispatch(self):
        pass

    async def send(self):
        await self.response.send_content(self.result)

    async def not_found_action(self):
        return 404
