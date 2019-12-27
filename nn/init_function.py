from torch.nn import Module
from torch.nn.init import kaiming_normal_, xavier_normal_


def create_init_function(method: str = 'none'):
    def init(module: Module):
        if method == 'none':
            return module
        elif method == 'he':
            kaiming_normal_(module.weight)
            return module
        elif method == 'xavier':
            xavier_normal_(module.weight)
            return module
        else:
            raise ("Invalid initialization method %s" % method)

    return init