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

