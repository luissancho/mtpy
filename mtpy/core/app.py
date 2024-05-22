import json
import logging
import sys


class App(object):

    _app = None

    @staticmethod
    def get_():
        if not App.has_():
            App()

        return App._app

    @staticmethod
    def has_():
        return True if App._app is not None else False

    def __init__(self):
        self.deps = {}
        App._app = self

    def __getattr__(self, key):
        if key in self.deps:
            return self.deps[key]

    def get(self, key):
        return getattr(self, key)

    def set(self, key, value, **kwargs):
        if callable(value):
            self.deps[key] = value(self)
        elif isinstance(value, str) and App.class_exists(value):
            self.deps[key] = App.get_class(value)(**kwargs)
        else:
            self.deps[key] = value

        return self

    def loaded(self, key):
        return True if key in self.deps else False

    @staticmethod
    def class_exists(name):
        parts = name.rsplit('.', 1)
        if len(parts) > 1:
            return hasattr(parts[0], parts[1])
        else:
            return hasattr(sys.modules[__name__], name)

    @staticmethod
    def get_class(name):
        parts = name.rsplit('.', 1)
        if len(parts) > 1:
            return getattr(parts[0], parts[1])
        else:
            return getattr(sys.modules[__name__], name)


class Config(object):

    def __init__(self, config=None):
        dconfig = {}

        if isinstance(config, dict):
            dconfig = config
        elif isinstance(config, str):
            dconfig = json.loads(config)

        for key, value in dconfig.items():
            self.offset_set(key, value)

    def offset_exists(self, key):
        return hasattr(self, key)

    def offset_get(self, key):
        if hasattr(self, key):
            return getattr(self, key)

    def offset_set(self, key, value):
        if isinstance(value, dict):
            setattr(self, key, Config(value))
        else:
            setattr(self, key, value)

    def offset_unset(self, key):
        delattr(self, key)

    def offset_is_empty(self, key):
        if not hasattr(self, key):
            return True

        val = getattr(self, key)
        if Config.is_config(val):
            if len(val.to_dict()) == 0:
                return True
        else:
            if val == '':
                return True

        return False

    def to_dict(self):
        dconfig = {}

        for key, val in self.__dict__.items():
            if Config.is_config(val):
                dconfig[key] = val.to_dict()
            else:
                dconfig[key] = val

        return dconfig

    def merge(self, config):
        return self._merge(Config(config))

    def _merge(self, config, instance=None):
        if not Config.is_config(instance):
            instance = self

        for key, val in config.__dict__.items():
            mval = getattr(instance, key)
            if Config.is_config(mval) and Config.is_config(val):
                self._merge(val, mval)
            else:
                setattr(instance, key, val)

        return instance

    @staticmethod
    def is_config(obj):
        return isinstance(obj, Config)


class Core(object):

    def __init__(self):
        self.app = App.get_()


class Log(object):

    levels = {
        logging.DEBUG: 'debug',
        logging.INFO: 'info',
        logging.ERROR: 'error'
    }

    def __init__(self, path, verbose=0):
        self.path = path
        self.verbose = verbose
        self.loggers = {}

        for level, name in Log.levels.items():
            self.add_logger(level, name)

    def add_logger(self, level, name):
        path = '{}/{}.log'.format(self.path, name)
        fmt = '[%(asctime)s] %(message)s' if level != logging.DEBUG else '%(message)s'

        logger = logging.getLogger(name)
        logger.setLevel(level)

        if self.verbose > 0:
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(logging.Formatter(fmt))

            logger.addHandler(handler)

        if level != logging.DEBUG:
            handler = logging.FileHandler(path)
            handler.setFormatter(logging.Formatter(fmt))

            logger.addHandler(handler)

        self.loggers[name] = logger

        return self

    def add_handler(self, handler):
        for level, name in Log.levels.items():
            if level != logging.DEBUG:
                self.loggers[name].addHandler(handler)

        return self

    def log(self, name, message):
        fn = getattr(self.loggers[name], name)
        tr = True if name == 'error' else False

        return fn(message, exc_info=tr)

    def __getattr__(self, key):
        if key not in list(Log.levels.values()):
            raise AttributeError(key)

        def log(message):
            return self.log(key, message)

        return log
