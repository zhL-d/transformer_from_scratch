# 错误的根本原因在于函数decode_utf8_bytes_to_str_wrong以单个byte拿utf-8进行decode，
# 但utf-8并不是把所有codepoint都存在单个byte中，可能存在2byte或3个byte等
def decode_utf8_bytes_to_str_wrong(bytestring: bytes):
    return "".join([bytes([b]).decode("utf-8") for b in bytestring])

# 这个的输出结果对是因为，"hello" 是ascii，unicode的code point和ascii兼容，utf-8存ascii的codepoint是以单个byte来存，
# 所以错误函数以单个byte来decode成codepoint恰好没问题
print(decode_utf8_bytes_to_str_wrong("hello".encode("utf-8")))

# wrong example
print("wrong example", decode_utf8_bytes_to_str_wrong("你".encode("utf-8")))