# lv4

## 도둑질

```python
def solution(money):
    n = len(money)
    dp1 = [0]*n
    dp2 = [0]*n
    
    dp1[0] = 0
    for i in range(1, n):
        dp1[i] = max(dp1[i-1], dp1[i-2]+money[i])
    
    dp2[0] = money[0]
    for i in range(1, n-1):
        dp2[i] = max(dp2[i-1], dp2[i-2]+money[i])
    
    return max(dp1[-1], dp2[-2])
```



## 올바른 괄호의 개수

```python
def solution(n):
    from math import factorial
    
    return factorial(n*2)/factorial(n)/factorial(n+1)
```



## 호텔 방 배정

```python
import sys
sys.setrecursionlimit(10000)

def find(x, room):
    if x not in room:
        room[x] = x+1
        return x
    
    y = find(room[x], room)
    room[x] = y+1
    return y

def solution(k, room_number):
    answer = []
    room = {}
    for x in room_number:
        answer.append(find(x, room))

    return answer
```



## 무지의 먹방 라이브

```python
def solution(food_times, k):
    import heapq
    
    if sum(food_times) <= k: return -1
    
    heap = []
    for i in range(len(food_times)):
        heapq.heappush(heap, (food_times[i], i+1))
    
    prev, leftover = 0, len(food_times)
    while (heap[0][0]-prev)*leftover <= k:
        now = heapq.heappop(heap)[0]
        k -= (now-prev)*leftover
        leftover -= 1
        prev = now
    
    return sorted(heap, key=lambda x: x[1])[k%leftover][1]
```



## 쿠키 구입

```python
def solution(cookie):
    n, answer = len(cookie), 0
    for i in range(n-1):
        l, r, l_sum, r_sum = i, i+1, cookie[i], cookie[i+1]
        
        while 1:
            if l_sum == r_sum:
                answer = max(answer, l_sum)
            
            if l>0 and l_sum<=r_sum:
                l -= 1
                l_sum += cookie[l]
            elif r<n-1 and l_sum>=r_sum:
                r += 1
                r_sum += cookie[r]
            else:
                break
    
    return answer
```



## 자동완성

```python
## 시간 초과
def solution(words):
    from collections import defaultdict
    dic = defaultdict(int)
    for word in words:
        for i in range(len(word)):
            dic[word[:i+1]] += 1
    
    cnt = 0
    for word in words:
        for i in range(len(word)):
            if dic[word[:i+1]]==1:
                cnt += i+1
                break
        else:
            cnt += len(word)
    return cnt
  
## 정답
def get_prefix_len(x, y):
    for i in range(len(x)):
        if not y.startswith(x[:i+1]):
            return i+1
    return len(x)

def solution(words):
    n = len(words)
    words.sort()
    words = [""]+words+[""]
    cnt = 0
    for i in range(1, n+1):
        now = words[i]
        prv, nxt = words[i-1], words[i+1]
        cnt += max(get_prefix_len(now, prv), get_prefix_len(now, nxt))
    return cnt
```

