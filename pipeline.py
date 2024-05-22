#!/usr/bin/env python

import sys

from mtpy import mtpy, pipelines
from mtpy.core.utils.strings import to_camel


def main(name, action, chunks):
    mtpy.run()

    if '.' in name:
        module, name = name.split('.')
        name = '{}.{}'.format(module, to_camel(name))
    else:
        name = to_camel(name)

    pipeline = getattr(pipelines, name)()

    if len(chunks) > 0:
        for chunk in chunks:
            pipeline.run(action, chunk=chunk)
    else:
        pipeline.run(action)


if __name__ == '__main__':

    name = sys.argv[1]
    action = sys.argv[2] if len(sys.argv) > 2 else None
    chunks = sys.argv[3:] if len(sys.argv) > 3 else []

    main(name, action, chunks)
