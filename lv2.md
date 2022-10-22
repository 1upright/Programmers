# lv2

## 최댓값과 최솟값

```python
def solution(s):
    arr = list(map(int, s.split()))
    return f'{min(arr)} {max(arr)}'
```



## JadenCase 문자열 만들기

```python
def solution(s):
    return ' '.join(x.capitalize() for x in s.split(' '))
```



## 이진 변환 반복하기

```python
def solution(s):
    answer = [0, 0]
    while s != '1':
        answer[1] += s.count('0')
        s = s.replace('0', '')
        s = bin(len(s))[2:]
        answer[0] += 1
    return answer
```



## 최솟값 만들기

```python
def solution(A,B):
    return sum(sorted(A, reverse=True)[i]*sorted(B)[i] for i in range(len(A)))
```



## 올바른 괄호

```python
def solution(s):
    stack = []
    for x in s:
        if x == '(':
            stack.append(x)
        if x == ')':
            if not stack:
                return False
            stack.pop()
    if stack:
        return False
    return True
```



## 피보나치 수

```python
def solution(n):
    fib = [0, 1]
    for i in range(2, n+1):
        fib.append((fib[-1]+fib[-2])%1234567)
    return fib[-1]
```



## 숫자의 표현

```python
def solution(n):
    return len([i for i in range(1, n+1, 2) if not n%i])
```



## 다음 큰 숫자

```python
def solution(n):
    cnt = bin(n).count('1')
    while 1:
        n += 1
        if bin(n).count('1') == cnt:
            break    
    return n
```



## 카펫

```python
def solution(brown, yellow):
    N = brown + yellow
    for i in range(3, N//3+1):
        if not N%i and (i-2)*(N//i-2) == yellow:
            return [N//i, i]
```



## 영어 끝말잇기

```python
def solution(n, words):
    for i in range(1, len(words)):
        if words[i] in words[:i] or words[i-1][-1] != words[i][0]:
            return [i%n+1, i//n+1]
    return [0, 0]
```



## 짝지어 제거하기

```python
def solution(s):
    stack = []
    for x in s:
        if stack and stack[-1] == x:
            stack.pop()
        else:
            stack.append(x)

    return 0 if stack else 1
```



## 구명 보트

```python
def solution(people, limit):
    people.sort()
    i, j, cnt = 0, len(people)-1, 0
    while i <= j:
        cnt += 1
        if people[i] + people[j] <= limit:
            i += 1
        j -= 1
    return cnt
```



## N개의 최소공배수

```python
def solution(arr):
    from math import gcd
    answer = arr[0]
    for x in arr:
        answer *= x//gcd(answer, x)
    return answer
```



## 예상 대진표

```python
def solution(n,a,b):
    cnt = 0
    while a != b:
        a, b = (a+1)//2, (b+1)//2
        cnt += 1
    return cnt
```



## 멀리 뛰기

```python
def solution(n):
    dp = [0]*(n+1)
    for i in range(n+1):
        if i <= 3:
            dp[i] = i
        else:
            for j in range(4, n+1):
                dp[j] = dp[j-1] + dp[j-2]
    return dp[n]%1234567
```



## 점프와 순간 이동

```python
def solution(n):
    return bin(n).count('1')
```



## 행렬의 곱셈

```python
def solution(arr1, arr2):
    answer = [[0]*len(arr2[0]) for _ in range(len(arr1))]
    for i in range(len(answer)):
        for j in range(len(answer[0])):
            for k in range(len(arr1[0])):
                answer[i][j] += arr1[i][k] * arr2[k][j]
    return answer
```



## H-Index

```python
def solution(citations):
    citations.sort()
    l = len(citations)
    for i in range(l):
        if citations[i] >= l-i:
            return l-i
    return 0
```



## 캐시

```python
def solution(cacheSize, cities):
    from collections import deque
    q = deque()
    cnt = 0
    
    for city in cities:
        city = city.lower()
        if cacheSize:
            if city in q:
                q.remove(city)
                q.append(city)
                cnt += 1
            else:
                if len(q) == cacheSize:
                    q.popleft()
                q.append(city)
                cnt += 5
        else:
            cnt += 5
    return cnt
```



## 괄호 회전하기

```python
def solution(s):
    answer = 0
    ls = list(s)
    for i in range(len(s)):
        stack = []
        for x in ls:
            if stack:
                if (stack[-1] == '[' and x == ']') or (stack[-1] == '(' and x == ')') or (stack[-1] == '{' and x == '}'):
                    stack.pop()
                else:
                    stack.append(x)
            else:
                stack.append(x)
        if not stack:
            answer += 1
        ls.append(ls.pop(0))
    return answer
```



## 기능개발

```python
def solution(progresses, speeds):
    from math import ceil

    remains = []
    N = len(progresses)
    for i in range(N):
        remains.append(ceil((100-progresses[i])/speeds[i]))

    cnt = 0
    answer = []
    for i in range(N):
        if remains[cnt] < remains[i]:
            answer.append(i-cnt)
            cnt = i
    answer.append(N-cnt)
        
    return answer
```



## 프린터

```python
def solution(priorities, location):
    answer = 0
    while 1:
        tmp = max(priorities)
        for i in range(len(priorities)):
            if tmp == priorities[i]:
                answer += 1
                priorities[i] = 0
                tmp = max(priorities)
                if i == location:
                    return answer
```



## 위장

```python
def solution(clothes):
    dic = {}
    for x, y in clothes:
        if y in dic:
            dic[y] += 1
        else:
            dic[y] = 1
    
    cnt = 1
    for i in dic.values():
        cnt *= (i+1)
    return cnt -1
```



## 전화번호 목록

```python
def solution(phone_book):
    phone_book.sort()
    for i in range(len(phone_book)-1):
        if len(phone_book[i])<=len(phone_book[i+1]) and phone_book[i+1][:len(phone_book[i])]==phone_book[i]:
            return False
    return True

# 다른 풀이(startswith)
def solution(phone_book):
    phone_book.sort()
    for i in range(len(phone_book)-1):
        if phone_book[i+1].startswith(phone_book[i]):
            return False
    return True
```



## 뉴스 클러스터링

```python
def solution(str1, str2):
    from collections import Counter
    
    str1, str2 = str1.lower(), str2.lower()
    c1 = Counter([str1[i:i+2] for i in range(len(str1)-1) if str1[i:i+2].isalpha()])
    c2 = Counter([str2[i:i+2] for i in range(len(str2)-1) if str2[i:i+2].isalpha()])
    
    return int(sum((c1&c2).values())/sum((c1|c2).values())*65536) if sum((c1|c2).values()) else 65536
```



## 타겟 넘버

```python
def solution(numbers, target):
    def dfs(i, s):
        if i == n:
            if s == target:
                nonlocal answer
                answer += 1
        else:
            dfs(i+1, s+numbers[i])
            dfs(i+1, s-numbers[i])
            
    n = len(numbers)
    answer = 0
    dfs(0, 0)
    return answer
```



## k진수에서 소수 개수 구하기

```python
def solution(n, k):
    def is_prime(x):
        if x == 1:
            return False
        for i in range(2, int(x**0.5)+1):
            if not x%i:
                return False
        return True
    
    tmp = ''
    while n:
        tmp += str(n%k)
        n = n//k
    tmp = tmp[::-1]
    
    result = 0
    for num in tmp.split('0'):
        if num and is_prime(int(num)):
            result += 1
    return result
```



## 더 맵게

```python
def solution(scoville, K):
    import heapq
    heapq.heapify(scoville)
    
    cnt = 0
    while scoville[0] < K:
        cnt += 1
        heapq.heappush(scoville, heapq.heappop(scoville)+heapq.heappop(scoville)*2)
        if len(scoville) == 1 and scoville[0] < K:
            return -1
    
    return cnt
```



## 주식가격

```python
def solution(prices):
    l = len(prices)
    answer = [0]*l
    for i in range(l):
        for j in range(i+1, l):
            if prices[i] <= prices[j]:
                answer[i] += 1
            else:
                answer[i] += 1
                break
    return answer
```



## 주차 요금 계산

```python
def solution(fees, records):
    import math
    from collections import deque
    
    time, fee, unit_time, unit_fee = fees
    
    dic = {}
    for x in records:
        t, n, y = x.split()
        if n in dic:
            dic[n].append(t)
        else:
            dic[n] = deque([t])
    
    for x in dic:
        if len(dic[x])%2:
            dic[x].append('23:59')
    
    dic2 = {}
    for x in sorted(dic):
        dic2[x] = 0
        y = dic[x]
        while y:
            a, b = y.popleft(), y.popleft()
            p, q = a.split(':')
            r, s = b.split(':')
            dic2[x] += int(r)*60+int(s)-int(p)*60-int(q)
    
    answer = []
    for x in dic2:
        y = dic2[x]
        answer.append(fee if y<time else fee+math.ceil((y-time)/unit_time)*unit_fee)

    return answer
```



## 압축

```python
def solution(msg):

    dic = {}
    for i in range(26):
        dic[chr(65+i)] = i+1
    
    answer = []
    word = ''
    i, no = 0, 26
    while i < len(msg):
        word += msg[i]
        if word in dic:
            i += 1
        else:
            no += 1
            dic[word] = no
            answer.append(dic[word[:-1]])
            word = ''

    answer.append(dic[word])
    return answer
```



## 오픈채팅방

```python
def solution(record):
    dic = {}
    for rec in record:
        r = rec.split()
        if len(r) == 3:            
            dic[r[1]] = r[2]
    
    answer = []    
    for rec in record:
        r = rec.split()
        if r[0] == 'Enter':
            answer.append(f'{dic[r[1]]}님이 들어왔습니다.')
        elif r[0] == 'Leave':
            answer.append(f'{dic[r[1]]}님이 나갔습니다.')
    
    return answer
```



## n진수 게임

```python
def foo(i, n):
    tmp = "0123456789ABCDEF"
    x, y = divmod(i, n)
    return foo(x, n) + tmp[y] if x else tmp[y]

def solution(n, t, m, p):
    s = ''
    for i in range(t*m):
        s += foo(i, n)
    
    answer = ''
    for i in range(t):
        answer += s[i*m+p-1]
    
    return answer
```



## 피로도

```python
def solution(k, dungeons):
    def dfs(cnt, s):
        nonlocal answer
        if answer < cnt:
            answer = cnt

        for i in range(l):
            if not visited[i] and s >= dungeons[i][0]:
                visited[i] = 1
                dfs(cnt+1, s-dungeons[i][1])
                visited[i] = 0

    answer = 0
    l = len(dungeons)
    visited = [0]*l
    dfs(0, k)
    
    return answer
```



## 땅따먹기

```python
def solution(land):
    for i in range(1, len(land)):
        for j in range(len(land[0])):
            land[i][j] += max(land[i-1][:j]+land[i-1][j+1:])
    return max(land[-1])
```



## 연속 부분 수열 합의 개수

```python
def solution(elements):
    new_elements = elements * 2
    l = len(elements)
    ans = set()

    for i in range(l):
        for j in range(i, i+l):
            ans.add(sum(new_elements[i:j+1]))
    return len(ans)

# 숏코딩
def solution(elements):
    return len(set(sum((elements*2)[i:i+j]) for i in range(len(elements)) for j in range(1, len(elements)+1)))
```



## 프렌즈4블록

```python
def solution(m, n, board):
    from collections import deque
    
    for i in range(m):
        board[i] = list(board[i])
    
    answer = 0
    while 1:
        check = set()
        for i in range(m-1):
            for j in range(n-1):
                if board[i+1][j] == board[i][j+1] == board[i+1][j+1] == board[i][j] != '0':
                    check.add((i,j)); check.add((i+1,j)); check.add((i,j+1)); check.add((i+1,j+1))
        if check:
            answer += len(check)
            for i, j in check:
                board[i][j] = '0'
            
            for j in range(n):
                q = deque()
                for i in range(m-1, -1, -1):
                    if board[i][j] == '0':
                        q.append((i, j))
                    else:
                        if q:
                            ni, nj = q.popleft()
                            board[ni][nj], board[i][j] = board[i][j], '0'
                            q.append((i, j))
            check = set()
        
        else:
            break

    return answer
```

