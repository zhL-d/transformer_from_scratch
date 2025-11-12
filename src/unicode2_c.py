# string_encoded =  "你".encode("utf-8")
# print("oringinal bytes:", list(string_encoded))

# first_two = string_encoded[:2]
# print("first two bytes:", list(string_encoded))

# print("decode string:", first_two.decode("utf-16"))

seq = b'\xD8\xD8'

print("decode string:", seq.decode("utf-32"))