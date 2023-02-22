from sys import argv
from os import system

assert len(argv) == 2

num = int(argv[1])
system(f"type tools\\in\\{num:04d}.txt | clip.exe")
