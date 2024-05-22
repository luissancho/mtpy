#!/usr/bin/env python

import sys

from mtpy import mtpy
from mtpy.core.worker import Worker


def main(queue):
    mtpy.run()

    worker = Worker(queue)
    worker.listen()


if __name__ == '__main__':

    queue = sys.argv[1] if len(sys.argv) > 1 else None

    main(queue)
