def gen_hp_cartesian_product(hps):
    def helper(i, sofar={}):
        for val in hps[i].domain.values:
            next = {k:sofar[k] for k in sofar}
            next[hps[i]] = val
            if i == len(hps)-1:
                yield next
            else:
                for x in helper(i+1, next):
                    yield x
    return helper(0)