# for c2 in range(8):
#     c = 1 << c2
#     # p = 100
#     mn = 1 << 30
#     res = -1
#     for p in range(1,5001):
#         ss = [0] * 5001
#         for s in range(10,5000 + 1):
#             ss[s] = (s + p - 1) // p * (c + p)
#         print(c,p,sum(ss))
#         if mn > sum(ss):
#             mn = min(mn,sum(ss))
#             res = p
#     print(c,mn,res)

n=200
for i in range(10):
    with open(f"in/{i:04d}.txt","r") as file:
        # input()
        file.readline()
        s=[list(map(int,file.readline().split())) for i in range(n)]
        diff = [0] * 5001
        ds = 0
        for i in range(n):
            for j in range(n - 1):
                if s[i][j] < 100 and s[i][j + 1] < 100:
                    diff[abs(s[i][j] - s[i][j + 1])] += 1
                    ds += 1
        for i in range(n - 1):
            for j in range(n):
                if s[i][j] < 100 and s[i + 1][j] < 100:
                    diff[abs(s[i][j] - s[i + 1][j])] += 1
                    ds += 1
        print(*[sum(diff[:i * 10])for i in range(1,11)],ds)
        # print(sum(diff[0:100]),"/",ds)
        # for i in range(5000,-1,-1):
        #     if diff[i]:
        #         print(i)
        #         break
# import numpy as np
# import matplotlib.pyplot as plt
# n,w,h,c = map(int,input().split())
# s=[list(map(int,input().split())) for i in range(n)]
# ws = [list(map(int,input().split())) for i in range(w)]
# hs = [list(map(int,input().split())) for i in range(h)]

# print(*[s[ws[i][0]][ws[i][1]] for i in range(w)])
# print(*[s[hs[i][0]][hs[i][1]] for i in range(h)])

# a = [0] * 5001
# for i in range(n):
#     for j in range(n):
#         a[s[i][j]] += 1
# for i in range(50):
#     print(sum(a[i * 100:(i + 1) * 100]))
# print(np.array(a))
# plt.hist(np.array(a))
# plt.show()
# for i in range(50):
#     print(i * 100,"~",i * 101,sum(diff[i * 100:(i + 1) * 100]),diff[i * 100:(i + 1) * 100])
