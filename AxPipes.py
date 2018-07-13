import struct


def process(_sin):
    return "Python got {}".format(_sin)


def receive():
    n = struct.unpack('I', pr.read(4))[0]
    return pr.read(n).decode('utf-8')


def respond(_sout):
    pw.write(struct.pack('I', len(_sout)))
    pw.write(_sout.encode('utf-8'))
    pw.seek(0)


pr = open(r'\\.\pipe\ctop', 'rb', 0)
pw = open(r'\\.\pipe\ptoc', 'wb', 0)
