from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import models.dense

def get_model(name, lrp=False):
    name = name.lower()
    if name == "dense":
        return models.dense.Model
    else:
        raise LookupError("Unknown model %s" % name)
