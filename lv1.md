# lv1

## 신고 결과 받기

```python
def solution(id_list, report, k):
    import re
    
    answer = []
    report = list(set(report))
    id_dic = {}
    for id in id_list:
        id_dic[id] = [0]
    
    for rep in report:
        a, b = re.split(' ', rep)
        id_dic[b].append(a)
    
    for id in id_list:
        reporters = id_dic[id]
        M = len(reporters)
        if M > k:
            for i in range(1, M):
                id_dic[reporters[i]][0] += 1
    
    for id in id_list:
        answer.append(id_dic[id][0])
    
    return answer
```



## 로또의 최고 순위와 최저 순위

```python
def solution(lottos, win_nums):
    answer = []
    min_cnt = 0
    mystery = 0
    
    for num in lottos:
        if num in win_nums:
            min_cnt += 1
        if not num:
            mystery += 1
    
    rank = [6, 6, 5, 4, 3, 2, 1]
    answer.append(rank[min_cnt+mystery])
    answer.append(rank[min_cnt])
    
    return answer
```



## 신규 아이디 추천

```python
def solution(new_id):
    #1
    new_id = new_id.lower()

    #2
    tmp = list(new_id)
    for i in range(len(tmp)):
        if tmp[i] not in ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','-','_','.','0','1','2','3','4','5','6','7','8','9']:
            tmp[i] = ''
    tmp = list(''.join(tmp))

    #3
    if tmp:
        for i in range(len(tmp)-1):
            if tmp[i] == '.' and tmp[i+1] == '.':
                tmp[i] = ''
        tmp = list(''.join(tmp))

    #4
    if tmp:
        if tmp[0] == '.':
            tmp[0] = ''
        if tmp[-1] == '.':
            tmp[-1] = ''
        tmp = list(''.join(tmp))

    #5
    if not tmp:
        tmp = ['a']

    #6
    if len(tmp) >= 16:
        tmp = tmp[:15]
        if tmp[-1] == '.':
            tmp[-1] = ''
            tmp = list(''.join(tmp))

    #7
    while len(tmp) <= 2:
        tmp.append(tmp[-1])

    answer = ''.join(tmp)
    return answer

## 정규식
from re import sub

def solution(new_id):
    new_id = new_id.lower()
    new_id = sub("[^a-z0-9-_.]", "", new_id)
    new_id = sub("\.+", ".", new_id)
    new_id = sub("(^\.|\.$)", "", new_id)
    new_id = new_id if new_id else "a"
    new_id = sub("\.$", "", new_id[:15])
    new_id = new_id if len(new_id) > 2 else new_id + new_id[-1] * (3 - len(new_id))
    return new_id
```



## 숫자 문자열과 영단어

```python
def solution(s):
    nums = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
    for i in range(10):
        if nums[i] in s:
            s = s.replace(nums[i], str(i))
    answer = int(s)
    return answer
```



## 키패드 누르기

```python
def solution(numbers, hand):
    answer = ''
    keypad = [(3, 1), (0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)]
    now = [(3, 0), (3, 2)]
    
    for n in numbers:
        x, y = keypad[n]
        if n in [1, 4, 7]:
            answer += 'L'
            now[0] =(x, y)
        elif n in [3, 6, 9]:
            answer += 'R'
            now[1] = (x, y)
        else:
            left = abs(now[0][0]-x)+abs(now[0][1]-y)
            right = abs(now[1][0]-x)+abs(now[1][1]-y)
            if left < right:
                answer += 'L'
                now[0] = (x, y)
            elif left > right:
                answer += 'R'
                now[1] = (x, y)
            else:
                if hand == 'left':
                    answer += 'L'
                    now[0] = (x, y)
                else:
                    answer += 'R'
                    now[1] = (x, y)
    
    return answer
```



## 크레인 인형뽑기 게임

```python
def solution(board, moves):
    answer = 0
    N = len(board)
    
    s = []
    for x in moves:
        for i in range(N):
            if board[i][x-1]:
                tmp = board[i][x-1]
                if s and s[-1] == tmp:
                    s.pop()
                    answer += 2
                else:
                    s.append(tmp)
                board[i][x-1] = 0
                break
    return answer
```



## 없는 숫자 더하기

```python
def solution(numbers):
    return 45 - sum(numbers)
```



## 음양 더하기

```python
def solution(absolutes, signs):
    N = len(absolutes)
    answer = 0
    for i in range(N):
        if signs[i]:
            answer += absolutes[i]
        else:
            answer -= absolutes[i]
    return answer
```



## 내적

```python
def solution(a, b):
    return sum(a[i]*b[i] for i in range(len(a)))
```



## 소수 만들기

```python
from itertools import combinations

def solution(nums):
    answer = 0
    for combi in combinations(nums, 3):
        x = sum(combi)
        for i in range(2, int(x**0.5)+1):
            if not x%i: break
        else:
            answer += 1
    return answer
```



## 완주하지 못한 선수

```python
def solution(participant, completion):
    participant.sort()
    completion.sort()
    for i in range(len(completion)):
        if participant[i] != completion[i]:
            return participant[i]
    return participant[-1]
```



## 폰켓몬

```python
def solution(nums):
    return min(len(nums)//2, len(set(nums)))
```



## K번째 수

```python
def solution(array, commands):
    return [sorted(array[com[0]-1:com[1]])[com[2]-1] for com in commands]
```



## 모의고사

```python
def solution(answers):
    x1 = [1, 2, 3, 4, 5]
    x2 = [2, 1, 2, 3, 2, 4, 2, 5]
    x3 = [3, 3, 1, 1, 2, 2, 4, 4, 5, 5]
    check = [0, 0, 0]
    answer = []
    
    for i in range(len(answers)):
        if x1[i%5] == answers[i]:
            check[0] += 1
        if x2[i%8] == answers[i]:
            check[1] += 1
        if x3[i%10] == answers[i]:
            check[2] += 1
    
    for i in range(3):
        if check[i] == max(check):
            answer.append(i+1)
    return answer
```



## 체육복

```python
def solution(n, lost, reserve):
    real_lost = set(lost)-set(reserve)
    real_reserve = set(reserve)-set(lost)
    
    for l in real_lost:
        for r in real_reserve:
            if l == r+1:
                real_reserve.remove(r)
                break
            if l == r-1:
                real_reserve.remove(r)
                break
        else:
            n -= 1
    return n
```



## 실패율

```python
def solution(N, stages):
    res = {}
    k = len(stages)
    for i in range(1, N+1):
        if k:
            cnt = stages.count(i)
            res[i] = cnt/k
            k -= cnt
        else:
            res[i] = 0
    return sorted(res, key=lambda x: -res[x])
```



## 약수의 개수와 덧셈

```python
def solution(left, right):
    answer = 0
    for i in range(left, right+1):
        cnt = 0
        for j in range(1, i+1):
            if not i%j:
                cnt += 1
        answer += -i if cnt%2 else i
    return answer

# 숏코딩
def solution(left, right):
    return sum([-x if int(x**0.5)==x**0.5 else x for x in range(left, right+1)])
```



## 3진법 뒤집기

```python
def solution(n):
    answer = ''
    while n > 0:
        n, r = divmod(n, 3)
        answer += str(r)
    return int(answer, 3)
```



## 예산

```python
def solution(d, budget):
    d.sort()
    while budget < sum(d): d.pop()
    return len(d)
```



## 두 개 뽑아서 더하기

```python
from itertools import combinations

def solution(numbers):
    answer = set()
    for combi in combinations(numbers, 2):
        answer.add(sum(combi))    
    return sorted(list(answer))
```



## 2016년

```python
from datetime import datetime, date
def solution(a, b): return ['MON', 'TUE', 'WED', 'THU', 'FRI', 'SAT', 'SUN'][date(2016, a, b).weekday()]
```



## 최소 직사각형

```python
def solution(sizes):
    w = h = 0
    for size in sizes:
        if w < max(size):
            w = max(size)
        if h < min(size):
            h = min(size)
    return w*h
```
