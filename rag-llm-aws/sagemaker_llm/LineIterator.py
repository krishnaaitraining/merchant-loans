import io

class LineIterator:
    """
    Parses a byte stream, accounting for potentially incomplete JSON objects split across chunks.
    """
    def __init__(self, stream):
        self.byte_iterator = iter(stream)
        self.buffer = io.BytesIO()
        self.read_pos = 0

    def __iter__(self):
        return self

    def __next__(self):
        while True:
            try:
                self.buffer.seek(self.read_pos)
                line = self.buffer.readline()
            except ValueError:
                # Handle the case where read_pos is out of range
                self.read_pos = 0
                self.buffer.seek(self.read_pos)
                line = self.buffer.readline()

            if line:
                self.read_pos += len(line)
                return line.decode('utf-8')

            try:
                chunk = next(self.byte_iterator)
            except StopIteration:
                if self.read_pos < self.buffer.getbuffer().nbytes:
                    self.read_pos = self.buffer.getbuffer().nbytes
                    return self.buffer.getvalue()[self.read_pos:].decode('utf-8')
                raise StopIteration
            self.buffer.seek(0, io.SEEK_END)
            self.buffer.write(chunk['PayloadPart']['Bytes'])
