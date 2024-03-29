# lv3

## 정수 삼각형

```python
def solution(triangle):
    for i in range(1, len(triangle)):
        for j in range(len(triangle[i])):
            if j == 0:
                triangle[i][j] += triangle[i-1][j]
            elif j == len(triangle[i])-1:
                triangle[i][j] += triangle[i-1][j-1]
            else:
                triangle[i][j] += max(triangle[i-1][j], triangle[i-1][j-1])
    return max(triangle[-1])
```



## 이중우선순위큐

```python
def solution(operations):
    import heapq
    heap = []
    for op in operations:
        x, y = op.split()
        if x == 'I':
            heapq.heappush(heap, int(y))
        elif heap:
            if y == '1':
                heap.remove(max(heap))
            else:
                heapq.heappop(heap)
    return [max(heap), heap[0]] if heap else [0, 0]
```



## 최고의 집합

```python
def solution(n, s):
    if n > s: return [-1]
    x, y = divmod(s, n)
    ans = [x]*n
    for i in range(y): ans[i] += 1
    return sorted(ans)
```



## 야근 지수

```python
# 66.6점
def solution(n, works):
    from itertools import combinations_with_replacement
    answer = 50000000000000
    for combi in combinations_with_replacement(range(len(works)), n):
        tmp = works[::]
        for i in combi: tmp[i] -= 1
        x = sum(i**2 if i>0 else 0 for i in tmp)
        if answer > x: answer = x
    return answer

# 60.0점
def solution(n, works):
    from itertools import combinations_with_replacement
    from collections import Counter
    answer = 50000000000000
    for combi in combinations_with_replacement(range(len(works)), n):
        tmp = works[::]
        cnt = Counter(combi)
        for i in cnt: tmp[i] -= cnt[i]
        x = sum(i**2 if i>0 else 0 for i in tmp)
        if answer > x: answer = x
    return answer

# 86.7점
def solution(n, works):
    for _ in range(n): works[works.index(max(works))] -= 1
    return sum(x**2 if x>0 else 0 for x in works)

# 정답
def solution(n, works):
    import heapq
    heap = sorted([-x for x in works])
    for _ in range(n): heapq.heappush(heap, heapq.heappop(heap)+1)
    return sum(x**2 if x<0 else 0 for x in heap)
```



## 네트워크

```python
def solution(n, computers):
    from collections import deque
    
    visited = [[0]*n for _ in range(n)]
    cnt = 0
    for i in range(n):
        q = deque([])
        for j in range(n):
            if computers[i][j] and not visited[i][j]:
                q.append((i, j))
        
        if q: cnt += 1

        while q:
            ni, nj = q.popleft()
            visited[ni][nj] = 1
            for k in range(n):
                if computers[nj][k] and not visited[nj][k]:
                    q.append((nj, k))
    
    return cnt

# 다른 사람의 풀이
def solution(n, computers):
    tmp = list(range(n))
    for i in range(n):
        for j in range(n):
            if computers[i][j]:
                for k in range(n):
                    if tmp[k] == tmp[i]:
                        tmp[k] = tmp[j]

    return len(set(tmp))
```



## 단어 변환

```python
def solution(begin, target, words):
    from collections import deque
    
    q = deque([(begin, 0)])
    visited = [0]*len(words)
    while q:
        word, cnt = q.popleft()
        if word == target: return cnt
        
        for i in range(len(words)):
            if not visited[i]:
                tmp = 0
                for j in range(len(word)):
                    if words[i][j] != word[j]:
                        tmp += 1
                if tmp == 1:
                    q.append([words[i], cnt+1])
                    visited[i] = 1
    
    return 0
```



## 등굣길

```python
def solution(m, n, puddles):
    dp = [[0]*(m+1) for i in range(n+1)]
    dp[1][1] = 1

    for i in range(1, n+1):
        for j in range(1, m+1):
            if not (i, j) == (1, 1):
                if [j, i] in puddles:
                    dp[i][j] = 0
                else:
                    dp[i][j] = (dp[i-1][j] + dp[i][j-1]) % 1000000007
    return dp[n][m]
```



## 단속카메라

```python
def solution(routes):
    answer, tmp = 0, -30001
    for enter, exit in sorted(routes, key = lambda x: x[1]):
        if tmp < enter:
            answer += 1
            tmp = exit
    return answer
```



## 숫자 게임

```python
def solution(A, B):
    A.sort(reverse=True)
    B.sort(reverse=True)
    cnt = 0
    while B:
        if B[-1] > A[-1]:
            A.pop()
            B.pop()
            cnt += 1
        else:
            B.pop()
    return cnt
```



## 베스트 앨범

```python
def solution(genres, plays):
    from collections import defaultdict
    
    data = defaultdict(list)
    cnt = defaultdict(int)
    for g, p, i in sorted([[genres[i], -plays[i], i] for i in range(len(genres))]):
        data[g].append((p, i))
        cnt[g] -= p
        
    answer = []
    for x in [data[genre][:min(len(data[genre]), 2)] for genre, num in sorted(cnt.items(), key=lambda x: -x[1])]:
        for a, b in x:
            answer.append(b)
    
    return answer
```



## 기지국 설치

```python
def solution(n, stations, w):
    from math import ceil
    
    arr = [-w] + stations + [n+1+w]
    return sum(ceil(x/(w+w+1)) if x>0 else 0 for x in [(arr[i]-w-1)-(arr[i-1]+w) for i in range(1, len(arr))])
```



##  가장 먼 노드

```python
def solution(n, edge):
    from collections import deque
    
    adj = [[] for _ in range(n+1)]
    for u, v in edge:
        adj[u].append(v)
        adj[v].append(u)
    
    visited = [0]*(n+1)
    visited[1] = 1
    
    q = deque([1])
    while q:
        x = q.popleft()
        for y in adj[x]:
            if not visited[y]:
                visited[y] = visited[x]+1
                q.append(y)
    
    return visited.count(max(visited))
```



## 미로 탈출 명령어

```python
def solution(n, m, x, y, r, c, k):
    w = r-x
    h = c-y
    val = k-abs(w)-abs(h)
    if val<0 or val%2:
        return 'impossible'
    else:
        dist = {'d':(1, 0), 'l':(0, -1), 'r':(0, 1), 'u':(-1, 0)}
        move = {'d':0, 'l':0, 'r':0, 'u':0}
        if w>0: move['d'] += w
        elif w<0: move['u'] -= w
        if h>0: move['r'] += h
        elif h<0: move['l'] -= h

        if n-max(x, r) > val//2:
            move['d'] += val//2
            move['u'] += val//2
        else:
            move['d'] += n-max(x, r)
            move['u'] += n-max(x, r)
            move['l'] += val//2-n+max(x, r)
            move['r'] += val//2-n+max(x, r)

        ans = ''
        for _ in range(k):
            for m in move:
                if move[m] and 1<=dist[m][0]+x<=n and 1<=dist[m][1]+y<=n:
                    move[m] -= 1
                    ans += m
                    x += dist[m][0]
                    y += dist[m][1]
                    break
    return ans
```



## 표 병합

```python
def solution(commands):
    arr = [[['', str(i)+' '+str(j)] for j in range(51)] for i in range(51)]
    res = []

    for command in commands:
        com = command.split()

        if com[0] == 'UPDATE' and len(com) == 4:
            r1, c1 = int(com[1]), int(com[2])
            for i in range(51):
                for j in range(51):
                    if arr[i][j][1] == arr[r1][c1][1]:
                        arr[i][j][0] = com[3]

        elif com[0] == 'UPDATE' and len(com) == 3:
            for i in range(51):
                for j in range(51):
                    if arr[i][j][0] == com[1]:
                        arr[i][j][0] = com[2]

        elif com[0] == 'MERGE':
            r1, c1, r2, c2 = int(com[1]), int(com[2]), int(com[3]), int(com[4])
            if arr[r2][c2][0] and not arr[r1][c1][0]:
                r1, c1, r2, c2 = r2, c2, r1, c1
            r3, c3 = arr[r1][c1][1].split()
            r4, c4 = arr[r2][c2][1].split()
            for i in range(51):
                for j in range(51):
                    if arr[i][j][1] == r4+' '+c4:
                        arr[i][j][0] = arr[int(r3)][int(c3)][0]
                        arr[i][j][1] = r3+' '+c3

        elif com[0] == 'UNMERGE':
            r1, c1 = int(com[1]), int(com[2])
            tmp = arr[r1][c1][0]
            tmp2 = arr[r1][c1][1]
            for i in range(51):
                for j in range(51):
                    if arr[i][j][1] == tmp2:
                        arr[i][j][0] = ''
                        arr[i][j][1] = str(i)+' '+str(j)
            arr[r1][c1][0] = tmp

        elif com[0] == 'PRINT':
            r1, c1 = int(com[1]), int(com[2])
            x = arr[r1][c1][0] if arr[r1][c1][0] else 'EMPTY'
            res.append(x)
    return res
```



## 디스크 컨트롤러

```python
def solution(jobs):
    import heapq
    
    n = len(jobs)
    jobs.sort(key = lambda x: x[1])
    answer = start = 0
    while jobs:
        for i in range(len(jobs)):
            if jobs[i][0] <= start:
                start += jobs[i][1]
                answer += (start-jobs[i][0])/n
                jobs.pop(i)
                break
        else:
            start += 1    
    
    return int(answer)
```



## 불량 사용자

```python
def solution(user_id, banned_id):
    from re import fullmatch
    from itertools import permutations
    
    bans = '/'.join(banned_id).replace('*','.')
    res = set()
    for per in permutations(user_id, len(banned_id)):
        if fullmatch(bans, '/'.join(per)):
            res.add(''.join(sorted(per)))

    return len(res)
```



## 보석 쇼핑

```python
def solution(gems):
    from collections import defaultdict
    
    n, m = len(gems), len(set(gems))
    answer = [0, n-1]
    s = e = 0
    dic = defaultdict(int)
    dic[gems[0]] += 1
    while e<n:
        if len(dic)<m:
            e += 1
            if e<n: dic[gems[e]] += 1

        else:
            x = gems[s]
            
            if answer[1]-answer[0] > e-s:
                answer = [s, e]
            
            if dic[x]==1:
                del dic[x]
            else:
                dic[x] -= 1

            s += 1

    return [x+1 for x in answer]
```



## 입국심사

```python
def solution(n, times):
    s, e = 1, max(times)*n
    while s <= e:
        mid = (s+e)//2
        if sum(mid//time for time in times) >= n:
            answer = mid
            e = mid-1
        else:
            s = mid+1

    return answer
```



## 징검다리 건너기

```python
def solution(stones, k):
    answer, s, e = 0, 1, max(stones)
    while s <= e:
        mid = (s+e)//2
        cnt = 0
        for stone in stones:
            if stone <= mid:
                cnt += 1
            else:
                cnt = 0

            if cnt >= k:
                answer = mid
                e = mid-1
                break
        else:
            s = mid+1
        
    return answer
```



## 스티커 모으기(2)

```python
def solution(sticker):
    N = len(sticker)
    if N == 1: return sticker[0]

    dp1, dp2 = [0]*N, [0]*N
    
    dp1[0] = sticker[0]
    dp1[1] = sticker[0]
    for i in range(2, N-1):
        dp1[i] = max(dp1[i-2]+sticker[i], dp1[i-1])
    
    dp2[1] = sticker[1]
    for i in range(2, N):
        dp2[i] = max(dp2[i-2]+sticker[i], dp2[i-1])

    return max(dp1[-1], dp1[-2], dp2[-1], dp2[-2])
```



## 섬 연결하기

```python
def solution(n, costs):
    def find(x):
        if x == rep[x]:
            return x
        rep[x] = find(rep[x])
        return rep[x]
    
    def union(a, b):
        a = find(a)
        b = find(b)
        if b < a:
            rep[a] = b
        else:
            rep[b] = a
    
    costs.sort(key = lambda x: x[2])
    rep = list(range(n))
    
    res = 0
    for u, v, w in costs:
        if find(u) != find(v):
            union(u, v)
            res += w

    return res

## 다른 풀이
def solution(n, costs):
    costs.sort(key = lambda x: x[2])
    visited = set([costs[0][0]])
    
    answer = 0
    while len(visited) < n:
        for u, v, w in costs:
            if (u in visited and v not in visited) or (u not in visited and v in visited):
                visited.update([u, v])
                answer += w
                break

    return answer
```



## 여행경로

```python
def solution(tickets):
    from collections import defaultdict
    
    paths = defaultdict(list)
    for s, e in tickets: paths[s].append(e)
    for k in paths: paths[k].sort(reverse=True)

    stack = ["ICN"]
    answer = []
    while stack:
        top = stack.pop()
        if paths[top]:
            stack.append(top)
            stack.append(paths[top].pop())
        else:
            answer.append(top)

    return answer[::-1]
```



## 가장 긴 펠린드롬

```python
def solution(s):
    answer = 0
    for i in range(len(s)):
        for j in range(i+1, len(s)+1):
            x = s[i:j]
            if x == x[::-1]:
                answer = max(answer, len(x))
    return answer
```



## 순위

```python
def solution(n, results):
    from collections import defaultdict
    
    wins, loses = defaultdict(set), defaultdict(set)
    for winner, loser in results:
        wins[loser].add(winner)
        loses[winner].add(loser)
    
    for x in range(1, n+1):
        for winner in wins[x]: loses[winner].update(loses[x])
        for loser in loses[x]: wins[loser].update(wins[x])

    return [len(wins[x])+len(loses[x])==n-1 for x in range(1, n+1)].count(True)
```



## 경주로 건설

```python
def solution(board):
    from collections import deque
    
    n = len(board)
    answer = 600000
    visited = {}
    q = deque([(0, 0, 0, -1)])
    while q:
        i, j, cnt, d = q.popleft()
        for idx, (di, dj) in enumerate([(1, 0), (-1, 0), (0, 1), (0, -1)]):
            ni, nj = i+di, j+dj
            if 0<=ni<n and 0<=nj<n and not board[ni][nj]:
                ncnt = cnt + (100 if d == idx or d == -1 else 600)
                
                if ni == n-1 and nj == n-1:
                    answer = min(answer, ncnt)
                elif (ni, nj, idx) not in visited or visited[(ni, nj, idx)] > ncnt:
                    visited[(ni, nj, idx)] = ncnt
                    q.append((ni, nj, ncnt, idx))

    return answer
```



## 거스름돈

```python
def solution(n, money):
    dp = [1]+[0]*n
    
    for m in money:
        for cost in range(m, n+1):
            dp[cost] += dp[cost-m]
    
    return dp[n]%1000000007
```



## 풍선 터트리기

```python
def solution(a):
    check = [0]*len(a)
    l = r = 1000000001
    for i in range(len(a)):
        if l > a[i]:
            l = a[i]
            check[i] = 1
        if r > a[-1-i]:
            r = a[-1-i]
            check[-1-i] = 1

    return sum(check)
```



## 합승 택시 요금

```python
def solution(n, s, a, b, fares):
    import heapq
    
    def dijkstra(start, end):
        D = [20000001]*(n+1)
        heap = []
        D[start] = 0
        heapq.heappush(heap, [0, start])
        while heap:
            val, i = heapq.heappop(heap)
            if i == end:
                return D[end]
            for v, w in adj[i]:
                tmp = w + val
                if D[v] > tmp:
                    D[v] = tmp
                    heapq.heappush(heap, [tmp, v])
        return 20000001

    adj = [[] for _ in range(n+1)]
    for u, v, w in fares:
        adj[u].append([v, w])
        adj[v].append([u, w])
    heap = []
    
    return min(dijkstra(s, i) + dijkstra(i, a) + dijkstra(i, b) for i in range(1, n+1))
```



## 셔틀버스

```python
def solution(n, t, m, timetable):
    times = sorted([int(time[:2])*60+int(time[3:]) for time in timetable])
    bus_times = [9*60+t*i for i in range(n)]
    
    i = 0
    for time in bus_times:
        cnt = 0
        while cnt<m and i<len(times) and times[i]<=time:
            i += 1
            cnt += 1
        
        if cnt<m:
            answer = time
        else:
            answer = times[i-1]-1

    return str(answer//60).zfill(2)+":"+str(answer%60).zfill(2)
```



## 부대복귀

```python
def solution(n, roads, sources, destination):
    from collections import deque
    
    adj = [[] for _ in range(n+1)]
    for u, v in roads:
        adj[u].append(v)
        adj[v].append(u)
    
    dist = [-1]*(n+1)
    dist[destination] = 0
    q = deque([destination])

    while q:
        x = q.popleft()
        for y in adj[x]:
            if dist[y] == -1:
                dist[y] = dist[x]+1
                q.append(y)

    return [dist[x] for x in sources]
```



## 인사고과

```python
def solution(scores):
    x, y = scores[0]
    scores.sort(key=lambda x: (-x[0], x[1]))
    mx, answer = 0, 1
    for a, b in scores:
        if x<a and y<b:
            return -1
        if mx<=b:
            if x+y<a+b:
                answer += 1
            mx = b
    return answer
```



## 자물쇠와 열쇠

```python
def rotate(arr, m, d):
    res = [[0]*m for _ in range(m)]
    
    if d == 0:
        for i in range(m):
            for j in range(m):
                res[i][j] = arr[i][j]

    elif d == 1:
        for i in range(m):
            for j in range(m):
                res[j][m-i-1] = arr[i][j]

    elif d == 2:
        for i in range(m):
            for j in range(m):
                res[m-i-1][m-j-1] = arr[i][j]

    elif d == 3:
        for i in range(m):
            for j in range(m):
                res[m-j-1][i] = arr[i][j]
    
    return res

def check(arr, n):
    for i in range(n, n*2):
        for j in range(n, n*2):
            if arr[i][j] != 1:
                return False
    
    return True

def solution(key, lock):
    from copy import deepcopy
    
    m, n = len(key), len(lock)
    new_lock = [[0]*(n*3) for _ in range(n*3)]
    for i in range(n):
        for j in range(n):
            new_lock[n+i][n+j] = lock[i][j]
    
    for i in range(1, n*2):
        for j in range(1, n*2):
            for d in range(4):
                tmp_lock = deepcopy(new_lock)
                tmp_key = rotate(key, m, d)
                
                for r in range(m):
                    for c in range(m):
                        tmp_lock[i+r][j+c] += tmp_key[r][c]
                
                if check(tmp_lock, n):
                    return True
                
    return False
```



## 110 옮기기

```python
def solution(s):
    answer = []
    for t in s:
        stack = []
        cnt_110 = 0
        for x in t:
            if x == '0' and stack[-2:] == ['1', '1']:
                cnt_110 += 1
                stack.pop();stack.pop()
            else:
                stack.append(x)
        
        cnt_1 = 0
        while stack:
            if stack[-1] == '1':
                stack.pop()
                cnt_1 += 1
            else:
                break

        answer.append(''.join(stack)+'110'*cnt_110+'1'*cnt_1)

    return answer
```



## 스타 수열

```python
def solution(a):
    from collections import Counter
    
    answer = -1
    c = Counter(a)
    for k, v in c.items():
        if v*2 > answer:
            cnt = idx = 0
            while idx < len(a)-1:
                if (a[idx] != k and a[idx+1] != k) or a[idx] == a[idx+1]:
                    idx += 1
                    continue
                cnt += 2
                idx += 2
            answer = max(answer, cnt)
    return answer
```



## 다단계 칫솔 판매

```python
def solution(enroll, referral, seller, amount):
    answer = [0]*len(enroll)
    dic = {v:i for i, v in enumerate(enroll)}
    for s, a in zip(seller, amount):
        earning = a*100
        while s != '-' and earning > 0:
            idx = dic[s]
            answer[idx] += (earning - earning//10)
            earning //= 10
            s = referral[idx]
    
    return answer
```



## 길 찾기 게임

```python
# 모르겠어서 인터넷 베낌

def solution(nodeinfo):
    from collections import deque
    
    nodes = sorted([v+[i+1] for i, v in enumerate(nodeinfo)], key=lambda x: (-x[1], x[0]))
    dic = {}
    for x, y, idx in nodes:
        dic[idx] = [x, y, -1, -1]
        now = nodes[0][2]
        while 1:
            if x > dic[now][0]:
                if dic[now][3] == -1:
                    dic[now][3] = idx
                    break
                now = dic[now][3]
            elif x < dic[now][0]:
                if dic[now][2] == -1:
                    dic[now][2] = idx
                    break
                now = dic[now][2]
            else:
                break

    answer = []
    for x, y, idx in [(3, 2, 0), (2, 3, 1)]:
        s = [nodes[0][2]]
        res = deque()
        while s:
            now = s.pop()
            if idx == 0:
                res.append(now)
            elif idx == 1:
                res.appendleft(now)
            
            for c in (x, y):
                if dic[now][c] != -1:
                    s.append(dic[now][c])
        
        answer.append(list(res))

    return answer
```



## 표 편집

```python
def solution(n, k, cmd):
    link = {i: [i-1, i+1] for i in range(1, n+1)}
    answer = ['O']*n
    s = []
    now = k+1
    for c in cmd:
        if c[0] == 'U':
            for _ in range(int(c[2:])):
                now = link[now][0]

        elif c[0] == 'D':
            for _ in range(int(c[2:])):
                now = link[now][1]
        
        elif c[0] == 'C':
            prev, nxt = link[now]
            s.append((prev, now, nxt))
            answer[now-1] = 'X'
            now = link[now][0] if nxt == n+1 else link[now][1]
            
            if prev != 0:
                link[prev][1] = nxt
            if nxt != n+1:
                link[nxt][0] = prev
        
        elif c[0] == 'Z':
            prev, last, nxt = s.pop()
            answer[last-1] = 'O'
            
            if prev != 0:
                link[prev][1] = last
            if nxt != n+1:
                link[nxt][0] = last
    
    return "".join(answer)
```



## 파괴되지 않은 건물

```python
def solution(board, skill):
    n, m = len(board), len(board[0])
    arr = [[0]*(m+1) for _ in range(n+1)] 
    for typ, r1, c1, r2, c2, degree in skill:
        arr[r1][c1] += -degree if typ == 1 else degree
        arr[r1][c2+1] += degree if typ == 1 else -degree
        arr[r2+1][c1] += degree if typ == 1 else -degree
        arr[r2+1][c2+1] += -degree if typ == 1 else degree

    for i in range(n):
        for j in range(m):
            arr[i][j+1] += arr[i][j]

    for j in range(m):
        for i in range(n):
            arr[i+1][j] += arr[i][j]

    answer = 0
    for i in range(n):
        for j in range(m):
            board[i][j] += arr[i][j]
            if board[i][j] > 0:
                answer += 1

    return answer
```


## 파이썬

```python
def check(arr):
    for x, y, a in arr:
        if (a == 0 and not (y == 0 or [x-1, y, 1] in arr or [x, y, 1] in arr or [x, y-1, 0] in arr)) or (a == 1 and not ([x, y-1, 0] in arr or [x+1, y-1, 0] in arr or ([x-1, y, 1] in arr and [x+1, y, 1] in arr))):
            return False
    return True

def solution(n, build_frame):
    answer = []
    for x, y, a, b in build_frame:
        if b == 0:
            answer.remove([x, y, a])
        else:
            answer.append([x, y, a])
        
        if not check(answer):
            if b == 0:
                answer.append([x, y, a])
            else:
                answer.remove([x, y, a])

    return sorted(answer)
```



## N으로 표현

```python
def solution(N, number):
    dp = []
    for i in range(1, 9):
        tmp = set([int(str(N)*i)])
        for j in range(i-1):
            for x in dp[j]:
                for y in dp[-1-j]:
                    tmp.update([x+y, x-y, x*y, x//y if y else x])
                    
        if number in tmp: return i
    
        dp.append(tmp)

    return -1
```



## 최적의 행렬 곱셈

```python
# 도움 받음

def solution(matrix_sizes):
    n = len(matrix_sizes)
    dp = [[10**9]*n for _ in range(n)]
    for i in range(n):
        dp[i][i] = 0

    for tmp in range(1, n):
        for i in range(n-tmp):
            j = i+tmp
            for k in range(i, j):
                dp[i][j] = min(dp[i][j], dp[i][k]+dp[k+1][j]+matrix_sizes[i][0]*matrix_sizes[k][1]*matrix_sizes[j][1])

    return dp[0][-1]
```



## 연속 펄스 부분 수열의 합

```python
def solution(sequence):
    sums = [0]
    for i, v in enumerate(sequence):
        sums.append(sums[-1]+v*(i%2*2-1))
    return max(sums)-min(sums)
```



## 광고 삽입

```python
def time_cal(time):
    h, m, s = time.split(":")
    return int(h)*3600+int(m)*60+int(s)

def solution(play_time, adv_time, logs):
    logs2 = [tuple(map(time_cal, log.split("-"))) for log in logs]
    pt, at = time_cal(play_time), time_cal(adv_time)
    arr = [0]*(pt+1)
    for s, e in logs2:
        arr[s] += 1
        arr[e] -= 1
    
    for i in range(pt):
        arr[i+1] += arr[i]
    
    pre_sum = [0]
    cnt = 0
    for x in arr:
        cnt += x
        pre_sum.append(cnt)
    
    max_val = answer = 0
    for i in range(pt-at+1):
        val = pre_sum[i+at]-pre_sum[i]
        if val > max_val:
            max_val = val
            answer = i
    
    h, answer = divmod(answer, 3600)
    m, answer = divmod(answer, 60)
    s = answer
    return str(h).zfill(2)+':'+str(m).zfill(2)+':'+str(s).zfill(2)
```



## 양과 늑대

```python
def solution(info, edges):
    def dfs(sheep, wolf):
        nonlocal answer
        if sheep>wolf:
            answer = max(answer, sheep)
            for u, v in edges:
                if visited[u] and not visited[v]:
                    visited[v] = 1
                    if info[v]:
                        dfs(sheep, wolf+1)
                    else:
                        dfs(sheep+1, wolf)
                    visited[v] = 0
    
    visited = [0]*len(info)
    visited[0] = 1
    answer = 0
    dfs(1, 0)
    return answer
```



## 외벽 점검

```python
def solution(n, weak, dist):
    from itertools import permutations
    
    dist.sort()
    m, l = len(weak), len(dist)
    weak2 = weak + [w+n for w in weak]
    for i in range(l):
        for per in permutations(dist[l-i-1:]):
            for j in range(m):
                d = list(per)
                w = weak2[j:j+m][::-1]
                
                while d and w:
                    can = d.pop()+w.pop()
                    while w and w[-1]<=can:
                        w.pop()
                
                if not w:
                    return i+1
    
    return -1
```

