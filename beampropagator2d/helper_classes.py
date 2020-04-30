# -------------------------------------------

# Created by:               jasper
# as part of the project:   Bachelorarbeit
# Date:                     4/30/20

#--------------------------------------------

from datetime import datetime as dt
import time
import sys

class ProgressBar:

    def __init__(self, object, prefix=""):

        self.total = len(object)
        self.prefix = prefix
        self.object = object

    def __iter__(self):
        self.iteration = 0
        self.start = dt.now()
        self.object = iter(self.object)
        return self

    def __next__(self):
        self.iteration += 1
        percent = int(self.iteration / self.total * 100)
        dtime = dt.now() - self.start
        estimated_duration = dtime.total_seconds() * self.total / self.iteration
        time_left = estimated_duration - dtime.total_seconds()
        dtime = str(dtime)

        sys.stdout.write("\r# [%s] elaspsed time [%s] est. dur.[%.2f] time left [%.2f][%-100s] %d%%" % (
            self.prefix ,dtime, estimated_duration, time_left, '#' * percent, percent))
        sys.stdout.flush()
        if self.iteration > self.total:
            sys.stdout.write("\n")
        return next(self.object)


class AutoIndent(object):
    def __init__(self, stream):
        self.stream = stream
        self.offset = 0
        self.frame_cache = {}

    def indent_level(self):
        i = 0
        base = sys._getframe(2)
        f = base.f_back
        while f:
            if id(f) in self.frame_cache:
                i += 1
            f = f.f_back
        if i == 0:
            # clear out the frame cache
            self.frame_cache = {id(base): True}
        else:
            self.frame_cache[id(base)] = True
        return i

    def write(self, stuff):
        indentation = '  ' * self.indent_level()
        def indent(l):
            if l:
                return indentation + l
            else:
                return l
        stuff = '\n'.join([indent(line) for line in stuff.split('\n')])
        self.stream.write(stuff)

    def flush(self):
        self.stream.flush()
