from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import models.dense_cifar

def get_model(name, lrp=False):
    name = name.lower()
    if name == "dense_cifar":
        return models.dense_cifar.Model
    else:
        raise LookupError("Unknown model %s" % name)
