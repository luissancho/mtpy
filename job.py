#!/usr/bin/env python

import json
import sys

from mtpy import jobs, mtpy
from mtpy.core.utils.strings import to_camel


def main(name, params):
    mtpy.run()

    name = to_camel(name)
    job = getattr(jobs, name)()

    if params is not None:
        params = json.loads(params)
        job.run(**params)
    else:
        job.run()


if __name__ == '__main__':

    name = sys.argv[1]
    params = sys.argv[2] if len(sys.argv) > 2 else None

    main(name, params)
