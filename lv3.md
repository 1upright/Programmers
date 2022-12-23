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

