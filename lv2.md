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



## 스킬트리

```python
def solution(skill, skill_trees):
    answer = 0
    for s in skill_trees:
        tmp = ''
        for x in s:
            if x in skill:
                tmp += x

        if tmp in skill and (tmp.startswith(skill[0]) or not tmp):
            answer += 1

    return answer
```



## 방문 길이

```python
def solution(dirs):
    visited = set()
    x = y = 0
    for d in dirs:
        if d == 'U' and y < 5:
            visited.add((x, y, x, y+1))
            y += 1
        elif d == 'D' and y > -5:
            visited.add((x, y-1, x, y))
            y -= 1
        elif d == 'L' and x > -5:
            visited.add((x-1, y, x, y))
            x -= 1
        elif d == 'R' and x < 5:
            visited.add((x, y, x+1, y))
            x += 1

    return len(visited)
```



## 모음 사전

```python
def solution(word):
    dic = {'A':0, 'E':1, 'I':2, 'O':3, 'U':4}
    
    answer = 0
    for i in range(len(word)):
        N = dic[word[i]]
        for j in range(4, i, -1):
            answer += 5 ** (j-i) * N
        answer += 1 + N

    return answer
```



## 게임 맵 최단거리

```python
def solution(maps):
    from collections import deque
    q = deque([(0, 0)])
    N, M = len(maps), len(maps[0])
    while q:
        i, j = q.popleft()
        if i == N-1 and j == M-1:
            return maps[i][j]
        for di, dj in [(1, 0), (0, 1), (-1, 0), (0, -1)]:
            ni, nj = i+di, j+dj
            if 0<=ni<N and 0<=nj<M and maps[ni][nj] and maps[ni][nj] == 1:
                maps[ni][nj] = maps[i][j] + 1
                q.append((ni, nj))
    return -1
```



## 2개 이하로 다른 비트

```python
def solution(numbers):
    answer = []
    for num in numbers:
        if num % 2:
            tmp = '0'+bin(num)[2:]
            i = tmp.rfind('0')
            tmp_list = list(tmp)
            tmp_list[i] = '1'
            tmp_list[i+1] = '0'
            x = ''.join(tmp_list)
            answer.append(int(x, 2))
        else:
            answer.append(num+1)
    return answer
```



## 다리를 지나는 트럭

```python
def solution(bridge_length, weight, truck_weights):
    from collections import deque
    answer = 1
    truck_weights = deque(truck_weights)
    bridge = deque([(truck_weights.popleft(), 1)])
    while bridge:
        tmp = []
        while bridge:
            x, cnt = bridge.popleft()
            if cnt+1 <= bridge_length:
                tmp.append((x, cnt+1))
        bridge = deque(tmp)
        s = sum([x[0] for x in bridge])
        if truck_weights and truck_weights[0]+s <= weight:
            bridge.append((truck_weights.popleft(), 1))
        answer += 1
    return answer
```



## 2 x n 타일링

```python
def solution(n):
    res = [1, 2]
    while len(res) < n:
        res.append((res[-1]+res[-2])%1000000007)
    return res[-1]
```



## 가장 큰 수

```python
def solution(numbers):
    return str(int("".join(sorted([str(x) for x in numbers], key = lambda x : x*3, reverse = True))))
```



## 소수 찾기

```python
def solution(numbers):
    from itertools import permutations
    
    ans = set()
    for i in range(len(numbers)):
        for p in permutations(list(numbers), i+1):
            ans.add(int("".join(p)))
    
    cnt = 0
    for x in ans:
        if x>= 2:
            for i in range(2, int(x**0.5)+1):
                if not x%i: break
            else:
                cnt += 1

    return cnt
```



## 쿼드압축 후 개수 세기

```python
def solution(arr):
    cnt = [0, 0]
    def comp(ni, nj, n):
        x = arr[ni][nj]
        for i in range(ni, ni+n):
            for j in range(nj, nj+n):
                if arr[i][j] != x:
                    m = n//2
                    comp(ni, nj, m)
                    comp(ni, nj+m, m)
                    comp(ni+m, nj, m)
                    comp(ni+m, nj+m, m)
                    return
        cnt[x] += 1
    comp(0, 0, n = len(arr))
    return cnt
```



## 할인 행사

```python
def solution(want, number, discount):
    dic = {}
    for i in range(len(want)):
        dic[want[i]] = number[i]
    
    cnt = 0
    for i in range(len(discount)-9):
        
        check = {}
        for j in range(i, i+10):
            if discount[j] in check:
                check[discount[j]] += 1
            else:
                check[discount[j]] = 1
        
        for x in dic:
            if x not in check or (x in check and dic[x] > check[x]):
                break
        else:
            cnt += 1

    return cnt
```



## 124 나라의 숫자

```python
def solution(n):
    answer = ''
    while n:
        n, x = divmod(n-1, 3)
        answer += ['1', '2', '4'][x]
    return answer[::-1]
```



## 큰 수 만들기

```python
def solution(number, k):
    answer = []
    for x in number:
        while k and answer and answer[-1] < x:
            answer.pop()
            k -= 1
        answer.append(x)
    return "".join(answer[:len(answer)-k])
```



## 삼각 달팽이

```python
def solution(n):
    answer = [[0]*(i+1) for i in range(n)]
    i, j, x = -1, 0, 1
    for k in range(n):
        for _ in range(n-k):
            if k%3 == 0:
                i += 1
            elif k%3 == 1:
                j += 1
            elif k%3 == 2:
                i -= 1
                j -= 1
            answer[i][j] = x
            x += 1
    return sum(answer, [])
```



## 롤케이크 자르기

```python
def solution(topping):
    from collections import Counter
    cnts = Counter(topping)
    cnts2 = set()
    ans = 0
    for x in topping:
        cnts[x] -= 1
        if cnts[x] == 0: cnts.pop(x)
        
        cnts2.add(x)
        
        if len(cnts) == len(cnts2):
            ans += 1
        
    return ans
```



## 메뉴 리뉴얼

```python
def solution(orders, course):
    from itertools import combinations
    from collections import Counter
    
    answer = []
    for i in course:
        menus = []
        for order in orders:
            menus += list(combinations(sorted(order), i))
        cnts = Counter(menus)
        
        if cnts and max(cnts.values()) > 1:
            answer += [''.join(c) for c in cnts if cnts[c] == max(cnts.values())]
        
    return sorted(answer)
```



## 괄호 변환

```python
def is_right(s):
    check = []
    for x in s:
        if x == '(':
            check.append(x)
        elif x == ')':
            if not check:
                return 0
            check.pop()
    return 1

def solution(p):
    if not p:
        return p
    
    for i in range(len(p)):
        tmp = p[:i+1]
        if tmp.count('(') == tmp.count(')'):
            u, v = p[:i+1], p[i+1:]
            break
    
    if is_right(u):
        return u + solution(v)
    
    return '(' + solution(v) + ')' + u[1:len(u)-1].replace('(', 'x').replace(')', '(').replace('x', ')')
```



## 수식 최대화

```python
def solution(expression):
    from itertools import permutations
    from copy import deepcopy
    
    answer = 0
    
    exp = []
    s = 0
    for i, v in enumerate(expression):
        if v in ["*", "+", "-"]:
            exp.append(expression[s:i])
            exp.append(v)
            s = i+1
    exp.append(expression[s:])

    
    for p in permutations(["*", "+", "-"], 3):
        tmp_exp = deepcopy(exp)
        for op in p:
            tmp = []
            while tmp_exp:
                x = tmp_exp.pop(0)
                if x == op:
                    if x == '+':
                        tmp.append(int(tmp.pop())+int(tmp_exp.pop(0)))
                    elif x == '-':
                        tmp.append(int(tmp.pop())-int(tmp_exp.pop(0)))
                    elif x == '*':
                        tmp.append(int(tmp.pop())*int(tmp_exp.pop(0)))
                else:
                    tmp.append(x)
            tmp_exp = tmp
        answer = max(abs(tmp_exp[0]), answer)

    return answer
```



## N-Queen

```python
def solution(n):
    def dfs(i, n):
        nonlocal res
        if i == n:
            res += 1
            return

        for j in range(n):
            if v1[j]==v2[i+j]==v3[i-j]==0:
                v1[j] = v2[i+j] = v3[i-j] = 1
                dfs(i+1, n)
                v1[j] = v2[i+j] = v3[i-j] = 0
    
    res = 0
    v1, v2, v3 = [0]*30, [0]*30, [0]*30
    dfs(0, n)
    return res
```



## 행렬 테두리 회전하기

```python
def solution(rows, columns, queries):
    arr = [[j*columns+i+1 for i in range(columns)] for j in range(rows)]
    answer = []
    for q in queries:
        si, sj, ei, ej = q
        i, j = si, sj
        move = []
        while j < ej:
            move.append((i-1, j-1))
            j += 1
        while i < ei:
            move.append((i-1, j-1))
            i += 1
        while j > sj:
            move.append((i-1, j-1))
            j -= 1
        while i > si:
            move.append((i-1, j-1))
            i -= 1
        
        ti, tj = move[-1]
        tmp = arr[ti][tj]
        for i, j in move:
            tmp, arr[i][j] = arr[i][j], tmp
            
        answer.append(min(arr[i][j] for i, j in move))
    return answer
```



## 방금 그곡

```python
def solution(m, musicinfos):
    answer = []
    m = m.replace("C#", "c").replace("D#", "d").replace("F#", "f").replace("G#", "g").replace("A#", "a")
    
    cnt = 0
    for info in musicinfos:
        start, end, title, sound = info.split(',')
        start_hour, start_minute = map(int, start.split(':'))
        end_hour, end_minute = map(int, end.split(':'))
        time = (end_hour-start_hour)*60+(end_minute-start_minute)
        
        sound = sound.replace("C#", "c").replace("D#", "d").replace("F#", "f").replace("G#", "g").replace("A#", "a")
        sound *= time//len(sound)+1
        sound = sound[:time]
        
        if m in sound:
            cnt -= 1
            answer.append((time, cnt, title))
    
    return sorted(answer)[-1][2] if answer else '(None)'
```



## 배달

```python
def solution(N, road, K):
    import heapq
    
    INF = 20000000
    D = [INF]*(N+1)
    D[1] = 0
    
    adj = [[] for _ in range(N+1)]
    for u, v, w in road:
        adj[u].append([v, w])
        adj[v].append([u, w])
    
    heap = []
    heapq.heappush(heap, [0,1])
    while heap:
        val, i = heapq.heappop(heap)
        for v, w in adj[i]:
            tmp = w + val
            if D[v] > tmp:
                D[v] = tmp
                heapq.heappush(heap, [tmp, v])

    return len([x for x in D if x <=K])
```



## 줄 서는 방법

```python
def solution(n, k):
    from math import factorial
    
    nums = list(range(1, n+1))
    answer = []
    for i in range(n, 0, -1):
        x, k = divmod(k, factorial(i-1))
        answer.append(nums.pop(x) if k else nums.pop(x-1))
    return answer
```



## 가장 큰 정사각형 찾기

```python
def solution(board):
    for i in range(1, len(board)):
        for j in range(1, len(board[0])):
            if board[i][j]:
                board[i][j] = min(board[i-1][j], board[i][j-1], board[i-1][j-1])+1
    return max(map(max, board))**2
```



## 두 큐 합 같게 만들기

```python
def solution(queue1, queue2):
    from collections import deque
    
    queue1, queue2 = deque(queue1), deque(queue2)
    answer, s1, s2 = 0, sum(queue1), sum(queue2)
    for _ in range(len(queue1)*3):
        if s1 > s2:
            x = queue1.popleft()
            s1 -= x
            s2 += x
            queue2.append(x)
        elif s1 < s2:
            x = queue2.popleft()
            s1 += x
            s2 -= x
            queue1.append(x)
        else:
            return answer
        answer += 1
        
    return -1

```



## 멀쩡한 사각형

```python
def solution(w,h):
    from math import gcd
    return w*h-w-h+gcd(w, h)
```



## 거리두기 확인하기

```python
def bfs(place):
    from collections import deque
    ppl = []
    for i in range(5):
        for j in range(5):
            if place[i][j] == 'P':
                ppl.append([i, j])
    
    for si, sj in ppl:
        q = deque([(si, sj)])
        visited = [[0]*5 for _ in range(5)]
        visited[si][sj] = 1
        dist = [[0]*5 for _ in range(5)]
        
        while q:
            i, j = q.popleft()
            if dist[i][j] > 1:
                break

            for di, dj in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                ni, nj = i+di, j+dj
                if 0<=ni<5 and 0<=nj<5 and not visited[ni][nj]:
                    if place[ni][nj] == 'O':
                        q.append((ni, nj))
                        visited[ni][nj] = 1
                        dist[ni][nj] = dist[i][j]+1
                    
                    elif place[ni][nj] == 'P':
                        return 0
    return 1

def solution(places):
    return [bfs(x) for x in places]
```



## 하노이의 탑

```python
def solution(n):
    def move(start, end, mid, n):
        nonlocal answer
        if n == 1:
            answer.append([start, end])
            return

        move(start, mid, end, n-1)
        answer.append([start, end])
        move(mid, end, start, n-1)
    
    answer = []
    move(1, 3, 2, n)
    return answer
```



## 전력망을 둘로 나누기

```python
def select(tree, n, a, b):
    for i in range(1, n+1):
        for j in tree[i]:
            if sorted([i, j]) != sorted([a, b]):
                return i

def solution(n, wires):
    from collections import deque
    tree = [[] for _ in range(n+1)]
    for a, b in wires:
        tree[a].append(b)
        tree[b].append(a)
    
    answer = 100
    for a, b in wires:
        q = deque([select(tree, n, a, b)])
        cnt = 1
        visited = [0]*(n+1)
        while q:
            node = q.popleft()
            visited[node] = 1
            for x in tree[node]:
                if not visited[x] and sorted([node, x]) != sorted([a, b]):
                    q.append(x)
                    cnt += 1
        answer = min(answer, abs(cnt*2-n))
    return answer
```



## 문자열 압축

```python
def solution(s):
    res = []
    for i in range(1, len(s)+1):
        new = ''
        cnt = 1
        tmp = s[:i]
        
        for j in range(i, len(s)+i, i):
            if tmp == s[j:i+j]:
                cnt += 1
            else:
                if cnt == 1:
                    new += tmp
                else:
                    new += str(cnt) + tmp
                tmp = s[j:i+j]
                cnt = 1

        res.append(len(new))
    return min(res)
```



## 후보키

```python
def solution(relation):
    from itertools import combinations
    N, M = len(relation), len(relation[0])
    res = []
    for i in range(1, M+1):
        for comb in combinations(range(M), i):
            tmp = [tuple(relation[k][j] for j in comb) for k in range(N)]
            
            if len(set(tmp)) == N:
                for x in res:
                    if set(x).issubset(set(comb)):
                        break
                else:
                    res.append(comb)    
    return len(res)
```



## 귤 고르기

```python
def solution(k, tangerine):
    from collections import Counter
    cnt = answer = 0
    for kind, n in Counter(tangerine).most_common():
        cnt += n
        answer += 1
        if cnt >= k:
            return answer
```



## 점 찍기

```python
def solution(k, d):
    return sum(int((d**2-i**2)**0.5)//k+1 for i in range(0, d+1, k))
```



## 택배 상자

```python
def solution(order):
    sub = []
    now = 0
    for i in range(1, len(order)+1):
        sub.append(i)
        while sub[-1] == order[now]:
            sub.pop()
            now += 1
            if not sub: break
    return now
```



## 조이스틱

```python
def solution(name):
    answer = 0
    move = len(name)-1
    for i, c in enumerate(name):
        answer += min(ord(c)-ord('A'), ord('Z')-ord(c)+1)
        next = i+1
        while next<len(name) and name[next]=='A':
            next += 1
        move = min(move, 2*i+len(name)-next, 2*(len(name)-next)+i)
    answer += move
    return answer
```



## 숫자 블록

```python
def solution(begin, end):
    answer = []
    for i in range(begin, end+1):
        for j in range(2, int(i**0.5)+1):
            if not i%j and i//j<=10000000:
                answer.append(i//j)
                break
        else:
            answer.append(int(i!=1))
    return answer
```



## 숫자 카드 나누기

```python
def solution(arrayA, arrayB):
    from functools import reduce
    from math import gcd
    x, y = reduce(gcd, arrayA), reduce(gcd, arrayB)
    return max(x if all(n%x for n in arrayB) else 0, y if all(n%y for n in arrayA) else 0)
```



## 순위 검색

```python
# 시간 초과
def solution(info, query):
    result = []
    candidate = [x.split() for x in info]
    condition = [x.replace("and ", "").split() for x in query]
    for x in condition:
        cnt = 0
        for y in candidate:
            if all(x[i] == '-' or x[i] == y[i] for i in range(4)) and int(x[4]) <= int(y[4]):
                cnt += 1
        result.append(cnt)
        
    return result

# 정답
def solution(info, query):
    from itertools import combinations
    from collections import defaultdict
    from bisect import bisect_left

    candidate = [x.split() for x in info]
    dic = defaultdict(list)
    for x in candidate:
        for i in range(5):
            for combi in combinations(x[:-1], i):
                dic[''.join(combi)].append(int(x[-1]))
    
    for d in dic.values():
        d.sort()
    
    condition = [x.replace("and ", "").replace('-', '').split() for x in query]
    answer = []
    for x in condition:
        con, score = ''.join(x[:-1]), int(x[-1])
        if con not in dic:
            answer.append(0)
            continue
        answer.append(len(dic[con])-bisect_left(dic[con], score))

    return answer
```



## 혼자 놀기의 달인

```python
def solution(cards):
    boxes = {i+1: card for i, card in enumerate(cards)}
    groups = []
    
    while boxes:
        target = list(boxes.keys())[0]
        visited = []
        while target not in visited:
            visited.append(target)
            tmp = boxes[target]
            del boxes[target]
            target = tmp
        groups.append(len(visited))
    
    groups.sort(reverse=True)
    return groups[0]*groups[1] if len(groups)>1 else 0
```



## 3 x n 타일링

```python
def solution(n):
    if n%2: return 0
    n //= 2
    dp = [0, 3, 11]
    if n<3: return dp[n]
    
    for i in range(3, n+1):
        dp.append((3*dp[i-1]+sum(dp[1:i-1])*2+2)%1000000007)
    return dp[n]

# 다른 사람의 풀이
def solution(n):
    if n%2: return 0
    x = y = 1
    for _ in range(n//2): x, y = y, (4*y-x)%1000000007
    return y
```



## 양궁 대회

```python
def solution(n, info):
    from itertools import combinations_with_replacement
    
    answer = [-1]
    max_gap = 0
    for combi in combinations_with_replacement(range(11), n):
        tmp = [0]*11
        for i in combi:
            tmp[10-i] += 1
        
        ryan = apeach = 0
        for i in range(11):
            if tmp[i] > info[i]:
                ryan += 10-i
            elif tmp[i] <= info[i] and info[i]:
                apeach += 10-i
        
        gap = ryan - apeach
        if ryan > apeach and max_gap < gap:
            max_gap = gap
            answer = tmp
    
    return answer
```



## 우박수열 정적분

```python
def solution(k, ranges):
    nums = [k]
    while k != 1:
        if k%2:
            k = k*3+1
        else:
            k //= 2
        nums.append(k)
        
    areas = [(nums[i]+nums[i+1])/2 for i in range(len(nums)-1)]
    N = len(areas)
    return [-1.0 if i-j>N else sum(areas[i:N+j]) if i-j<N else 0.0 for i, j in ranges]
```



## 교점에 별 만들기

```python
def solution(line):
    from itertools import combinations
    
    star = []
    for com in combinations(range(len(line)), 2):
        a, b, e = line[com[0]]
        c, d, f = line[com[1]]
        if a*d-b*c:
            x, y = (b*f-e*d)/(a*d-b*c), (e*c-a*f)/(a*d-b*c)
            u, v = int(x), int(y)
            if x==u and y==v and (u, v) not in star:
                star.append((u, v))
    
    x_min, x_max, y_min, y_max = min(star)[0], max(star)[0], min(star, key = lambda x: x[1])[1], max(star, key = lambda x: x[1])[1]
    result = [['.']*(abs(x_max-x_min)+1) for _ in range((abs(y_max-y_min)+1))]
    for x, y in star:
        result[abs(y_max-y)][abs(x_min-x)] = '*'

    return [''.join(s) for s in result]
```



## 디펜스 게임

```python
def solution(n, k, enemy):
    import heapq
    
    heap = []
    for i, v in enumerate(enemy):
        heapq.heappush(heap, -v)
        if v <= n:
            n -= v
        elif k and v > n:
            k -= 1
            n -= heapq.heappop(heap)+v
        else:
            return i
    return len(enemy)
```



## 빛의 경로 사이클

```python
def solution(grid):
    N, M = len(grid), len(grid[0])
    visited = [[[0]*4 for _ in range(M)] for _ in range(N)]
    answer = []
    for i in range(N):
        for j in range(M):
            for d in range(4):
                if not visited[i][j][d]:
                    cnt, ni, nj = 0, i, j
                    while not visited[ni][nj][d]:
                        visited[ni][nj][d] = 1
                        cnt += 1
                        if grid[ni][nj] == "L":
                            d = (d-1)%4
                        elif grid[ni][nj] == "R":
                            d = (d+1)%4
                        ni = (ni + [1, 0, -1, 0][d]) % N
                        nj = (nj + [0, -1, 0, 1][d]) % M

                    answer.append(cnt)
    return sorted(answer)
```



## 테이블 해시 함수

```python
def solution(data, col, row_begin, row_end):
    from functools import reduce
    
    return reduce(lambda x, y: x^y, [sum(x%(i+1) for x in v) for i, v in enumerate(sorted(data, key = lambda x :(x[col-1], -x[0])))][row_begin-1:row_end])
```



## 마법의 엘리베이터

```python
def solution(storey):
    answer = i = 0
    while storey:
        x = storey%10
        if x < 5 or (x==5 and storey%100<50):
            answer += x
        else:
            answer += 10-x
            storey += 10
        i += 1
        storey //= 10
    
    return answer
```



## 유사 칸토어 비트열

```python
def solution(n, l, r):
    def cnt(x):
        from math import log
        
        if x <= 5: return '11011'[:x].count('1')
    
        time = int(log(x, 5))
        a, b = divmod(x, 5**time)
        if a == 1:
            return a*(4**time)+cnt(b)
        elif a == 2:
            return 2*(4**time)
        else:
            return (a-1)*(4**time)+cnt(b)

    return cnt(r)-cnt(l-1)
```



## 이모티콘 할인 행사

```python
def solution(users, emoticons):
    from itertools import product
    
    answer = [-1, -1]
    for discounts in product([10, 20, 30, 40], repeat=len(emoticons)):
        tmp = [0, 0]
        for rate, price in users:
            cost = sum((100-discounts[i])/100*emoticons[i] if discounts[i] >= rate else 0 for i in range(len(emoticons)))
            if cost >= price:
                tmp[0] += 1
            else:
                tmp[1] += cost
                
        answer = max(answer, tmp)

    return answer
```



## 택배 배달과 수거하기

```python
def solution(cap, n, deliveries, pickups):
    def foo(arr):
        indices = []
        box, idx = 0, n-1
        while idx > -1:
            if not arr[idx]:
                idx -= 1
                continue
            if not box:
                indices.append(idx)
            spare = cap - box
            if arr[idx] <= spare:
                box += arr[idx]
                idx -= 1
            else:
                arr[idx] -= spare
                box = 0
        return indices
    
    indices_d, indices_p = foo(deliveries)[::-1], foo(pickups)[::-1]
    answer = 0
    while indices_d or indices_p:
        answer += (max(indices_d.pop() if indices_d else 0, indices_p.pop() if indices_p else 0)+1)*2
    return answer

# 다른 사람의 풀이 참고
```



## 시소 짝꿍

```python
def solution(weights):
    from collections import defaultdict
    
    dic = defaultdict(int)
    for weight in weights:
        dic[weight] += 1
    
    cnt = 0
    for x in dic:
        for y in [x/2*3, x/3*2, x/2, x*2, x/4*3, x/3*4]:
            if y in dic:
                cnt += dic[x]*dic[y]
    
        cnt += dic[x]*(dic[x]-1)
    
    return cnt//2
```



## 무인도 여행

```python
def solution(maps):
    from collections import deque
    
    n, m = len(maps), len(maps[0])
    result = []
    visited = [[0]*m for _ in range(n)]
    for i in range(n):
        for j in range(m):
            if maps[i][j] != 'X' and not visited[i][j]:
                q = deque([(i, j)])
                visited[i][j] = 1
                cnt = int(maps[i][j])
                while q:
                    si, sj = q.popleft()
                    for di, dj in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                        ni, nj = si+di, sj+dj
                        if 0<=ni<n and 0<=nj<m and maps[ni][nj] != 'X' and not visited[ni][nj]:
                            q.append((ni, nj))
                            visited[ni][nj] = 1
                            cnt += int(maps[ni][nj])
                result.append(cnt)
    
    return sorted(result) if result else [-1]
```



## 숫자 변환하기

```python
def solution(x, y, n):
    from collections import deque
    
    q = deque([x])
    cnt = [0]*1000001
    cnt[x] = 1
    while q:
        u = q.popleft()
        for v in u+n, u*2, u*3:
            if 1<=v<=1000000 and not cnt[v]:
                cnt[v] = cnt[u]+1
                q.append(v)

    return cnt[y]-1
```



## 뒤에 있는 큰 수 찾기

```python
# 풀이 1
def solution(numbers):
    result = [-1]*len(numbers)
    stack = []
    
    for i, v in enumerate(numbers):
        while stack and numbers[stack[-1]] < v:
            result[stack.pop()] = v

        stack.append(i)

    return result

# 풀이 2
def solution(numbers):
    from heapq import heappush, heappop
    
    result = [-1]*len(numbers)
    heap = []
    
    for i, v in enumerate(numbers):
        while heap and heap[0][0] < v:
            val, idx = heappop(heap)
            result[idx] = v

        heappush(heap, (v, i))

    return result
```



## 호텔 대실

```python
def solution(book_time):
    import heapq

    times = sorted([(int(a[:2])*60+int(a[3:]), int(b[:2])*60+int(b[3:])) for a, b in book_time])
    heap = []
    cnt = 1
    for s, e in times:
        if not heap:
            heapq.heappush(heap, e)
            continue

        if heap[0] <= s:
            heapq.heappop(heap)
        else:
            cnt += 1

        heapq.heappush(heap, e+10)

    return cnt
```



## 미로 탈출

```python
def solution(maps):
    from collections import deque
    
    n, m = len(maps), len(maps[0])
    visited = [[0]*m for _ in range(n)]
    
    q = deque([])
    for i in range(n):
        for j in range(m):
            if maps[i][j] == 'S':
                q.append((i, j, 0))
                visited[i][j] = 1

    flag = 0
    q2 = deque([])
    visited2 = [[0]*m for _ in range(n)]
    while q:
        i, j, cnt = q.popleft()
        if maps[i][j] == 'L':
            flag = 1
            q2.append((i, j, cnt))
            visited2[i][j] = 1
            break

        for di, dj in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            ni, nj = i+di, j+dj
            if 0<=ni<n and 0<=nj<m and maps[ni][nj] != 'X' and not visited[ni][nj]:
                q.append((ni, nj, cnt+1))
                visited[ni][nj] = 1

    if flag:
        while q2:
            i, j, cnt = q2.popleft()
            if maps[i][j] == 'E':
                return cnt

            for di, dj in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                ni, nj = i+di, j+dj
                if 0<=ni<n and 0<=nj<m and maps[ni][nj] != 'X' and not visited2[ni][nj]:
                    q2.append((ni, nj, cnt+1))
                    visited2[ni][nj] = 1

    return -1
```



## 덧칠하기

```python
def solution(n, m, section):
    now = answer = 0
    for x in section:
        if now < x:
            now = x+m-1
            answer += 1
    return answer
```



## 혼자서 하는 틱택토

```python
def solution(board):
    def check(x):
        for i in range(3):
            if board[i] == x*3:
                return True
        
        for j in range(3):
            if [board[i][j] for i in range(3)] == [x]*3:
                return True
        
        if [board[i][i] for i in range(3)] == [x]*3:
            return True
        
        if [board[i][2-i] for i in range(3)] == [x]*3:
            return True
    
    cnt_o = sum(r.count('O') for r in board)
    cnt_x = sum(r.count('X') for r in board)
    
    if not 0 <= cnt_o-cnt_x <= 1:
        return 0
    
    if (check('O') and cnt_o == cnt_x) or (check('X') and cnt_o == cnt_x+1):
        return 0

    return 1
```



## 연속된 부분 수열의 합

```python
def solution(sequence, k):
    l, r, check, n, result = 0, -1, 0, len(sequence), []
    
    while 1:
        if check<k:
            r += 1
            if r>=n: break
            check += sequence[r]
        else:
            check -= sequence[l]
            l += 1
            if l>=n: break
        if check==k:
            result.append([l, r])

    return sorted(result, key=lambda x: (x[1]-x[0], x[0]))[0]
```



## 리코쳇 로봇

```python
def solution(board):
    from collections import deque
    
    n, m = len(board), len(board[0])
    q = deque()
    visited = [[-1]*m for _ in range(n)]
    
    for i in range(n):
        for j in range(m):
            if board[i][j]=='R':
                q.append((i, j))
                visited[i][j] = 0
            elif board[i][j]=='G':
                gi, gj = i, j
    
    while q:
        i, j = q.popleft()
        for di, dj in [(-1, 0), (1, 0), (0, 1), (0, -1)]:
            ni, nj = i, j
            while 0<=ni+di<n and 0<=nj+dj<m and board[ni+di][nj+dj] != 'D':
                ni += di
                nj += dj
            if visited[ni][nj] == -1:
                q.append((ni, nj))
                visited[ni][nj] = visited[i][j]+1
    
    return visited[gi][gj]
```



## 광물 캐기

```python
def solution(picks, minerals):
    from itertools import permutations
    
    minerals.extend([""]*(5-len(minerals)%5))
    n = min(sum(picks), len(minerals)//5)
    uses = []
    while len(uses)<n:
        if picks[0]:
            picks[0] -= 1
            uses.append("diamond")
        elif picks[1]:
            picks[1] -= 1
            uses.append("iron")
        elif picks[2]:
            picks[2] -= 1
            uses.append("stone")
    
    answer = 50*25
    dic = {("diamond", "diamond"):1, ("diamond", "iron"):1, ("diamond", "stone"):1, ("iron", "diamond"):5, ("iron", "iron"):1, ("iron", "stone"):1, ("stone", "diamond"):25, ("stone", "iron"):5, ("stone", "stone"):1}
    for per in set(permutations(uses, n)):
        fatigue = 0
        for i, v in enumerate(per):
            for j in range(i*5, i*5+5):
                if minerals[j]:
                    fatigue += dic[(v, minerals[j])]
        
        answer = min(answer, fatigue)
    
    return answer
```



## 과제 진행하기

```python
def solution(plans):
    plans.sort(key = lambda x: x[1])
    for plan in plans:
        n, s, pt = plan
        h, m = s.split(":")
        plan[1] = int(h)*60+int(m)
        plan[2] = int(pt)

    result, stack, now = [], [plans[0]], plans[0][1]
    for plan in plans[1:]:
        nxt = plan[1]
        while stack:
            n, s, pt = stack.pop()
            if now < s: now = s
            
            if now + pt <= nxt:
                now += pt
                result.append(n)
            else:
                stack.append([n, s, now+pt-nxt])
                now = nxt
                break
        
        stack.append(plan)

    while stack:
        result.append(stack.pop()[0])
    
    return result
```



## 요격 시스템

```python
def solution(targets):
    cnt = now = 0
    for s, e in sorted(targets):
        if s < now:
            now = min(now, e)
        else:
            cnt += 1
            now = e
    return cnt
```



## 당구 연습

```python
def solution(m, n, startX, startY, balls):
    reverses = [[-startX, startY], [startX, -startY], [m*2-startX, startY], [startX, n*2-startY]]
    result = []
    for x, y in balls:
        distances = [(x-a)**2+(y-b)**2 for a, b in reverses]
        if x==startX:
            if y<startY: distances[1] = float('inf')
            elif y>startY: distances[3] = float('inf')
        elif y==startY:
            if x<startX: distances[0] = float('inf')
            elif x>startX: distances[2] = float('inf')
        result.append(min(distances))
    return result
```



## 퍼즐 게임 챌린지

```python
def solution(diffs, times, limit):
    s, e = 1, max(diffs)
    while s<e:
        x = (s+e)//2
        cnt = diffs[0]*times[0]
        for i in range(1, len(diffs)):
            cnt += times[i]+(times[i]+times[i-1])*max(0, diffs[i]-x)
        
        if cnt <= limit:
            e = x
        else:
            s = x+1
    
    return s
```



## 석유 시추

```python
def solution(land):
    from collections import deque
    n, m = len(land), len(land[0])
    visited = [[0]*m for _ in range(n)]
    group_dic = {0:0}
    group_num = 1
    
    for i in range(n):
        for j in range(m):
            if land[i][j] and not visited[i][j]:
                cnt = 1
                q = deque([(i, j)])
                visited[i][j] = group_num
                while q:
                    si, sj = q.popleft()
                    for di, dj in [(-1,0),(1,0),(0,1),(0,-1)]:
                        ni, nj = si+di, sj+dj
                        if 0<=ni<n and 0<=nj<m and land[ni][nj] and not visited[ni][nj]:
                            visited[ni][nj] = group_num
                            cnt += 1
                            q.append((ni, nj))
                group_dic[group_num] = cnt
                group_num += 1
    
    answer = 0
    for j in range(m):
        groups = set()
        for i in range(n):
            groups.add(visited[i][j])
        
        oil_cnt = 0
        for num in list(groups):
            oil_cnt += group_dic[num]
        answer = max(answer, oil_cnt)
        
    return answer
```

