import struct
from io import BytesIO


class GokartStatusEvent:
    size = 4

    def __init__(self, bytes):
        buf = BytesIO(bytes)
        self._msg_fingerprint = buf.read(8)
        assert struct.unpack(">i", buf.read(4))[0] == self.size
        self.value = struct.unpack("<f", buf.read(self.size))[0]
