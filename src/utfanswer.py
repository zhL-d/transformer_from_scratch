def print_section(title: str):
    print('-' * 50)
    # print(title)
    # print('-' * 50)


print_section("test string")
test_string = "hello! こんにちは!"
print("test string:", test_string)
print("test string length:", len(test_string))

# utf-8
print_section("utf-8")

utf8_encoded =  test_string.encode("utf-8")
print("utf-8 encoded bytes: ", utf8_encoded)
print("utf-8 length:", len(utf8_encoded))

# utf-16
print_section("utf-16")

utf16_encoded = test_string.encode("utf-16")
print("utf-16 encoded bytes:", utf16_encoded)
print("utf-16 length:", len(utf16_encoded))

# utf-32
print_section("utf-32")

utf32_encoded = test_string.encode("utf-32")
print("utf-32 encoded bytes:", utf32_encoded)
print("utf-32 length:", len(utf32_encoded))
                            