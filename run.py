from subprocess import *
from time import sleep

from logging import getLogger, StreamHandler, DEBUG
logger = getLogger(__name__)
handler = StreamHandler()
handler.setLevel(DEBUG)
logger.setLevel(DEBUG)
logger.addHandler(handler)
logger.propagate = False


def main():
    TESTCASES = 1000
    cnt = 0
    for i in range(TESTCASES):
        infile = f".\\tools\\in\\{i:04d}.txt"
        proc = run([".\\tools\\tester.exe",".\\target\\release\\ahc018.exe","<",infile,">","nul"],shell=True,stderr=PIPE,encoding="utf-8")
        cost = int(proc.stderr.split()[-1])
        cnt += cost
        logger.debug(f"Test {i:04d}: {str(cost).ljust(20,' ')} Cost sum: {cnt}")

if __name__ == "__main__":
    main()
