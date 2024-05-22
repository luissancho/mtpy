from dotenv import load_dotenv
import numpy as np
import os
import re
import warnings

from .core import dal
from .core.api import Api, Router
from .core.app import App, Config, Log
from .core.io import FileSystem
from .core.services.aws import CloudWatchLogHandler
from .core.services.pushover import Pushover
from .core.services.s3 import S3
from .core.utils.helpers import is_empty, is_number

warnings.filterwarnings('ignore')


def run(fspath=None):
    if App.has_():
        return App.get_()

    app = App.get_()

    abspath = os.path.abspath(os.path.dirname(__file__)) + '/..'

    app.set('abspath', abspath)

    shpath = abspath + '/../../shared'
    if not os.path.exists(shpath):
        shpath = abspath

    app.set('shpath', shpath)

    fspath = fspath or shpath + '/files'

    app.set('fspath', fspath)

    if os.path.exists(shpath + '/.env'):
        load_dotenv(dotenv_path=shpath + '/.env')

    with open(abspath + '/config.json') as fh:
        dconfig = fh.read()
    for match in re.compile(r'\${([0-9A-Za-z_\-]+)}').finditer(dconfig):
        env_var = match.group(1)
        dconfig = dconfig.replace('${' + env_var + '}', os.getenv(env_var, ''))
    config = Config(dconfig)

    app.set('config', config)

    adapter = getattr(dal, config.db.adapter)
    if config.db.adapter == 'Redshift':
        db_params = {**config.db.params.to_dict(), **config.aws.to_dict(), **config.s3.to_dict()}
    else:
        db_params = config.db.params.to_dict()
    db = adapter(db_params, config.db.database)

    app.set('db', db)

    dbs = {}
    for name, src in config.dbs.to_dict().items():
        adapter = getattr(dal, src['adapter'])
        if src['adapter'] == 'Redshift':
            src['params'] = {**src['params'], **config.aws.to_dict(), **config.s3.to_dict()}
        dbs[name] = adapter(src['params'], src['database'])

    app.set('dbs', dbs)

    verbose = int(config.app.verbose) if is_number(config.app.verbose) else 0

    app.set('verbose', verbose)

    logger = Log(shpath + '/log', verbose)
    if not is_empty(config.cwl.group):
        cwl = CloudWatchLogHandler(config.aws.to_dict(), config.cwl.to_dict())
        logger.add_handler(cwl)

    app.set('logger', logger)

    if not is_empty(config.s3.bucket):
        fs = S3(config.aws.to_dict(), config.s3.bucket)
    else:
        fs = FileSystem(fspath)

    app.set('fs', fs)

    ps = Pushover(config.pushover.to_dict())

    app.set('ps', ps)

    np.random.seed(config.app.seed)

    return app


def api():
    if not App.has_():
        return

    app = App.get_()

    router = Router()
    router.add_route('/', 'index')

    app.set('router', router)

    return Api()
