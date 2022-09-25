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

