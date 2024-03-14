def mpd(n,s):
    mp=s.count('1')
    mp1=mp
    substrs=[substr for substr in s.split('1') if substr]
    for substr in substrs:
        mp1=max(mp1, mp+len(substr))
    return mp1
t=int(input())
for _ in range(t):
    n=int(input())
    s=input()
    print(mpd(n,s))