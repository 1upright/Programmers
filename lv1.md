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



## 나머지가 1이 되는 수 찾기

```python
def solution(n):
    for i in range(2, n+1):
        if not (n-1)%i:
            return i
```



## 부족한 금액 계산하기

```python
def solution(price, money, count):
    return max(price*sum(range(count+1))-money, 0)
```



## 가운데 글자 가져오기

```python
def solution(s):
    N = len(s)
    return s[N//2:N//2+1] if N%2 else s[N//2-1:N//2+1]
```



## 비밀지도

```python
def solution(n, arr1, arr2):
    arr3 = [list(bin(x)[2:].zfill(n)) for x in arr1]
    arr4 = [list(bin(x)[2:].zfill(n)) for x in arr2]
    tmp = [[' ']*n for _ in range(n)]
    
    for i in range(n):
        for j in range(n):
            if arr3[i][j] == '1' or arr4[i][j] == '1':
                tmp[i][j] = '#'
    
    answer = []        
    for l in tmp:
        answer.append(''.join(map(str, l)))

    return answer
```



## 다트 게임

```python
def solution(dartResult):
    dartResult = dartResult.replace('10', 'X')
    dic = {'S' : 1, 'D' : 2, 'T' : 3}
    s = []
    
    for x in dartResult:
        if '0'<=x<='9':
            s.append(int(x))
        elif x == 'X':
            s.append(10)
        elif x in ['S', 'D', 'T']:
            y = s.pop()
            s.append(y**dic[x])
        elif x == '*':
            y = s.pop()
            if s:
                z = s.pop()
                s.append(z*2)
            s.append(y*2)
        elif x == '#':
            y = s.pop()
            s.append(y*(-1))

    return sum(s)
```



## 나누어 떨어지는 숫자 배열

```python
def solution(arr, divisor):
    answer = sorted([x for x in arr if not x%divisor])
    return answer if answer else [-1]
```



## 두 정수 사이의 합

```python
def solution(a, b):
    return sum(range(min(a, b), max(a, b)+1))
```



## 문자열 내 마음대로 정렬하기

```python
def solution(strings, n):
    return sorted(sorted(strings), key=lambda x: x[n])
```



## 문자열 내 p와 y의 개수

```python
def solution(s):
    return not bool(s.count('p')+s.count('P')-s.count('y')-s.count('Y'))
```



## 문자열 내림차순으로 배치하기

```python
def solution(s):
    return "".join(sorted(s, reverse=True))
```



## 문자열 다루기 기본

```python
def solution(s):
    return (len(s) == 4 or len(s) == 6) and s.isdigit()
```



## 서울에서 김서방 찾기

```python
def solution(seoul):
    return f'김서방은 {seoul.index("Kim")}에 있다'
```



## 소수 찾기

```python
# 내 풀이(효율성 3/4)
def solution(n):
    cnt = 0
    for x in range(2, n+1):
        for i in range(2, int(x**0.5)+1):
            if not x%i: break
        else:
            cnt += 1
    return cnt

# 다른 사람의 풀이 - 에라스토테네스의 체
def solution(n):
    nums = set(range(2, n+1))
    for x in range(2, n+1):
        if x in nums:
            nums -= set(range(x*2, n+1, x))
    return len(nums)
```



## 수박수박수박수박수박수?

```python
def solution(n):
    return '수박'*(n//2) + '수'*(n%2)
```



## 문자열을 정수로 바꾸기

```python
def solution(s): return int(s)
```



## 시저 암호

```python
def solution(s, n):
    answer = ''
    for x in s:
        if x == ' ':
            answer += ' '
            continue
        v = ord(x)
        w = v+n-26 if (65<=v<91 and v+n>=91) or (97<=v<123 and v+n>=123) else v+n
        answer += chr(w)
    return answer
```



## 약수의 합

```python
def solution(n):
    answer = 0
    for x in range(1, n+1):
        if not n%x:
            answer += x
    return answer
```



## 이상한 문자 만들기

```python
def solution(s):
    answer = ''
    words = s.split(' ')
    for word in words:
        for i in range(len(word)):
            answer += word[i].lower() if i%2 else word[i].upper()
        answer += ' '
    return answer[:-1]
```



## 자릿수 더하기

```python
def solution(n):
    return sum(int(x) for x in list(str(n)))
```



## 자연수 뒤집어 배열로 만들기

```python
def solution(n):
    return [int(x) for x in reversed(list(str(n)))]
```



## 정수 내림차순으로 배치하기

```python
def solution(n):
    return int("".join(sorted(list(str(n)), reverse=True)))
```



## 정수 제곱근 판별

```python
def solution(n):
    return int((n**0.5+1)**2) if int(n**0.5) == n**0.5 else -1
```



## 제일 작은 수 제거하기

```python
def solution(arr):
    arr.remove(min(arr))
    return arr if arr else [-1]
```



## 짝수와 홀수

```python
def solution(num):
    return "Odd" if num%2 else "Even"
```



## 최대공약수와 최소공배수

```python
def solution(n, m):
    x, y = n, m
    while y > 0: x, y = y, x%y
    return [x, n*m//x]
```



## 콜라츠 추측

```python
def solution(num):
    cnt = 0
    while cnt < 500:
        if num == 1:
            return cnt
            break
        if num%2:
            num = num*3+1
        else:
            num //= 2    
        cnt += 1
    return -1
```



## 평균 구하기

```python
def solution(arr):
    return sum(arr)/len(arr)
```



## 하샤드 수

```python
def solution(x):
    return not x % sum(int(y) for y in str(x))
```



## x만큼 간격이 있는 n개의 숫자

```python
def solution(x, n):
    return [x+x*i for i in range(n)]
```



## 직사각형 별찍기

```python
a, b = map(int, input().strip().split(' '))
for i in range(b):
    print('*'*a)
```



## 핸드폰 번호 가리기

```python
def solution(phone_number):
    nums = list(phone_number)
    for i in range(len(nums)-5, -1, -1):
        nums[i] = '*'
    return "".join(nums)
```



## 행렬의 덧셈

```python
def solution(arr1, arr2):
    N = len(arr1)
    M = len(arr1[0])
    arr3 = [[0]*M for _ in range(N)]
    for i in range(N):
        for j in range(M):
            arr3[i][j] += arr1[i][j] + arr2[i][j]
    return arr3
```



## 성격 유형 검사하기

```python
def solution(survey, choices):
    score = {
        'R' : 0, 'T' : 0, 'C' : 0, 'F' : 0,
        'J' : 0, 'M' : 0, 'A' : 0, 'N' : 0,
    }

    for i in range(len(survey)):
        c = choices[i]-4
        if c<0:
            score[survey[i][0]] -= c
        elif c>0:
            score[survey[i][1]] += c

    answer = ''
    answer += 'T' if score['T']>score['R'] else 'R'
    answer += 'F' if score['F']>score['C'] else 'C'
    answer += 'M' if score['M']>score['J'] else 'J'
    answer += 'N' if score['N']>score['A'] else 'A'
    return answer
```



## 같은 숫자는 싫어

```python
def solution(arr):
    answer = []
    for x in arr:
        if not answer or x != answer[-1]:
            answer.append(x)
    return answer

## 다른 풀이
def solution(arr):
    return [arr[i] for i in range(len(arr)) if [arr[i]] != arr[i+1:i+2]]
```



## 삼총사

```python
def solution(number):
    from itertools import combinations
    return [1 if sum(combi) else 0 for combi in combinations(number, 3)].count(0)
```



## 콜라 문제

```python
def solution(a, b, n):
    return (n-b)//(a-b)*b
```



## 숫자 짝꿍

```python
def solution(X, Y):
    from collections import Counter
    cnt_x, cnt_y = Counter(X), Counter(Y)
    answer = ''
    for x in sorted(cnt_x, reverse=True):
        answer += x*min(cnt_x[x], cnt_y[x])
    return ('0' if len(answer)==answer.count('0') else answer) if answer else '-1'
```



## 푸드 파이트 대회

```python
def solution(food):
    tmp = "".join([str(i)*(food[i]//2) for i in range(4)])
    return tmp+'0'+tmp[::-1]
```



## 과일 장수

```python
def solution(k, m, score):
    apples = sorted(score, reverse=True)[:len(score)//m*m]
    res = 0
    for i in range(0, len(apples), m):
        tmp = apples[i:i+m]
        res += min(tmp)*m
    return res

# 숏코딩
def solution(k, m, score):
    return sum(sorted(score)[len(score)%m::m])*m
```



## 문자열 나누기

```python
def solution(s):
    first = ''
    check = []
    result = 0
    for x in s:
        if not first:
            first = x

        check.append(x)

        if check.count(first) == len(check)-check.count(first):
            first = ''
            check = []
            result += 1
            
    if check:
        result += 1

    return result
```



## 옹알이 (2)

```python
def solution(babbling):
    cnt = 0
    for b in babbling:
        for babble in ["aya", "ye", "woo", "ma"]:
            if babble*2 not in b:
                b = b.replace(babble, " ")
        if not b.strip():
            cnt += 1
    return cnt
```



## 명예의 전당 (1)

```python
def solution(k, score):
    answer = []
    result = []
    for s in score:
        answer.append(s)
        answer = sorted(answer, reverse=True)[:k]
        result.append(answer[-1])
    return result
```



## 기사단원의 무기

```python
def solution(number, limit, power):
    cnt = [1 for _ in range(number+1)]
    for i in range(2, number+1):
        for j in range(i, number+1, i):
            cnt[j] += 1
        if cnt[i] > limit:
            cnt[i] = power
    return sum(cnt)-1
```



## 햄버거 만들기

```python
def solution(ingredient):
    cnt = 0
    s = []
    for x in ingredient:
        s.append(x)
        if len(s) >= 4 and s[-4:] == [1, 2, 3, 1]:
            for _ in range(4):
                s.pop()
            cnt += 1
    return cnt
```



## 가장 가까운 같은 글자

```python
def solution(s):
    answer = []
    dic = {}
    for i, v in enumerate(s):
        answer.append(i-dic[v] if v in dic else -1)
        dic[v] = i
    return answer
```



## 크기가 작은 부분의 문자열

```python
def solution(t, p):
    return [int(t[i:i+len(p)])>int(p) for i in range(len(t)-len(p)+1)].count(False)
```



## 개인정보 수집 유효기간

```python
def solution(today, terms, privacies):
    standard = int(today[2:4])*28*12+int(today[5:7])*28+int(today[8:10])
    dic = {}
    for term in terms:
        x, y = term.split()
        dic[x] = standard - int(y)*28
        
    answer = []
    for i, v in enumerate(privacies):
        day, term = v.split()
        if int(day[2:4])*28*12+int(day[5:7])*28+int(day[8:10]) <= dic[term]:
            answer.append(i+1)

    return answer
```



## 둘만의 암호

```python
def solution(s, skip, index):
    answer = ''
    for x in s:
        cnt = 0
        while cnt < index:
            x = 'a' if x == 'z' else chr(ord(x)+1)
            if x not in skip:
                cnt += 1
        answer += x
    
    return answer

# 다른 풀이
def solution(s, skip, index):
    arr = [chr(i) for i in range(97, 123) if chr(i) not in skip]*3
    return "".join([arr[arr.index(i)+index] for i in s])
```

