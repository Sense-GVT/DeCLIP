import threading
from spring.nart.tools.io import send
from prototype import __version__
import linklink as link


def send_info(info):
    PrototypeINFO = {"name": "Prototype",
                     "version": __version__}
    PrototypeINFO.update(info)
    if link.get_rank() == 0:
        t = threading.Thread(target=send, args=(PrototypeINFO, ))
        t.start()
