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
