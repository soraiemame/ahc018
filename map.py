from PIL import Image
from sys import argv
from subprocess import *

assert len(argv) == 2

num = int(argv[1])

img = Image.new("L",(200,200))
proc = run(f".\\tools\\tester.exe cargo run --release --features visualize < tools\\in\\{num:04d}.txt",shell=True,stdout=PIPE,stderr=PIPE,encoding='utf-8')
vis = []
f = 0
for line in proc.stderr.split("\n"):
    if line == "start visualizing":
        f = 1
    elif line == "end visualizing":
        break
    elif f:
        vis.append(list(map(int,line.split())))
for i in range(200):
    for j in range(200):
        img.putpixel((i,j),int((5000 - vis[i][j]) / 5000 * 255))
img.show()
