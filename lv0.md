# lv0

## 몫 구하기

```python
def solution(num1, num2):
    return num1//num2
```



## 두 수의 곱

```python
def solution(num1, num2):
    return num1*num2
```



## 나머지 구하기

```python
def solution(num1, num2):
    return num1%num2
```



## 두 수의 차

```python
def solution(num1, num2):
    return num1-num2
```



## 두 수의 합

```python
def solution(num1, num2):
    return num1+num2
```



## 숫자 비교하기

```python
def solution(num1, num2):
    return 1 if num1==num2 else -1
```



## 나이 출력

```python
def solution(age):
    return 2023-age
```



## 각도기

```python
def solution(angle):
    return 1 if angle<90 else 2 if angle==90 else 3 if angle<180 else 4
```



## 두 수의 나눗셈

```python
def solution(num1, num2):
    return int(num1/num2*1000)
```



## 짝수의 합

```python
def solution(n):
    return n//2*(n//2+1)
```



## 양꼬치

```python
def solution(n, k):
    return n*12000+(k-n//10)*2000
```



## 배열의 평균값

```python
def solution(numbers):
    return sum(numbers)/len(numbers)
```



## 배열 뒤집기

```python
def solution(num_list):
    return num_list[::-1]
```



## 배열 원소의 길이

```python
def solution(strlist):
    return [len(x) for x in strlist]
```



## 점의 위치 구하기

```python
def solution(dot):
    return 1+int(dot[0]*dot[1]<0)+int(dot[1]<0)*2
```



## 중복된 숫자 개수

```python
def solution(array, n):
    return array.count(n)
```



## 배열 자르기

```python
def solution(numbers, num1, num2):
    return numbers[num1:num2+1]
```



## 배열 두 배 만들기

```python
def solution(numbers):
    return [x*2 for x in numbers]
```



## 삼각형의 완성 조건 (1)

```python
def solution(sides):
    a, b, c = sorted(sides)
    return 1 if a+b>c else 2
```



## 아이스 아메리카노

```python
def solution(money):
    return divmod(money, 5500)
```



## 피자 나눠 먹기 (1)

```python
def solution(n):
    return (n+6)//7
```



## 머쓱이보다 키 큰 사람

```python
def solution(array, height):
    return [x>height for x in array].count(True)
```

