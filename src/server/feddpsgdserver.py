
import logging

from .fedavgserver import FedavgServer

logger = logging.getLogger(__name__)


class FedDPsgdServer(FedavgServer):
    def __init__(self, **kwargs):
        super(FedDPsgdServer, self).__init__(**kwargs)
