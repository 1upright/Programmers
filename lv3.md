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

