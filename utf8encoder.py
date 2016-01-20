#!/usr/bin/python

import sys
import struct

trunc_args = sys.argv[1:]
file_name = trunc_args[0]
output_file = open('utf8encoder_out.txt', 'wb')

def read_two_bytes(fin):
  c = fin.read(2)
  if c == '':
   return ''
  return struct.unpack('!i', '\x00' * 2 + c)[0]

def supplementary(two_byte_int):
  if (two_byte_int >= 55296 and two_byte_int <= 56319):
    return True
  else:
    return False

def encode_non_supplementary(two_bytes):
  encode_in_utf8(two_bytes)

def encode_supplementary(two_bytes, open_file):
  second_two_bytes = read_two_bytes(open_file)
  int_val = compute_supplementary_value(two_bytes, second_two_bytes)
  encode_in_utf8(int_val)

def compute_supplementary_value(two_bytes, second_two_bytes):
  first_rel_bits = two_bytes - 55296
  second_rel_bits = second_two_bytes - 56320
  first_rel_bits = first_rel_bits << 10
  return first_rel_bits + second_rel_bits + 65536

def encode_in_utf8(int_val):
  if int_val < 128:
    write_val(int_val)
  elif int_val < 2048: 
    handle_int(int_val, 2)
  elif int_val < 65536:
    handle_int(int_val, 3)
  elif int_val < 2097152:
    handle_int(int_val, 4)
  elif int_val < 67108864:
    handle_int(int_val, 5)
  else:
    handle_int(int_val, 6)

def write_val(val):
  output_file.write(struct.pack('B', val))

def create_prefix(num):
  pre = 0
  for i in range(num):
    pre = pre | 1
    pre = pre << 1
  for j in range(7 - num):
    pre = pre << 1
  return pre

def handle_int(int_val, num):
  bytes = []
  for x in range(num - 1):
    six = int_val & 63
    byte = six | 128
    bytes.append(byte)
    int_val = int_val >> 6
  int_val = int_val & (127 >> num)
  int_val = int_val | create_prefix(num)
  bytes.append(int_val)
  bytes.reverse()
  for b in bytes:
    write_val(b)

def run(target_file):
  open_file = open(file_name, 'rb')
  try:
    two_bytes = read_two_bytes(open_file)
    while two_bytes != '':
      if supplementary(two_bytes):
        encode_supplementary(two_bytes, open_file)
      else:
        encode_non_supplementary(two_bytes)

      two_bytes = read_two_bytes(open_file)
  finally:
    open_file.close()
    output_file.close()

run(file_name)
