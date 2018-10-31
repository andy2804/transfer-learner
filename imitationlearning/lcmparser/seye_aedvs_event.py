import struct
from io import BytesIO


class SeyeAeApsEvent:
    size = 8

    def __init__(self, bytes):
        buf = BytesIO(bytes)
        self._msg_fingerprint = buf.read(8)
        self._length = struct.unpack(">i", buf.read(4))[0]
        self._nevents = self._length / self.size

        self.value = struct.unpack("<hh", buf.read(self.size))[0]
