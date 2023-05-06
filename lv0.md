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



## 중앙값 구하기

```python
def solution(array):
    return sorted(array)[len(array)//2]
```



## 짝수 홀수 개수

```python
def solution(num_list):
    n = [x%2 for x in num_list].count(0)
    return [n, len(num_list)-n]
```



## 문자열 뒤집기

```python
def solution(my_string):
    return my_string[::-1]
```



## 피자 나눠 먹기 (3)

```python
def solution(slice, n):
    return (n+slice-1)//slice
```



## 최댓값 만들기 (1)

```python
def solution(numbers):
    numbers.sort()
    return numbers[-1]*numbers[-2]
```



## 배열의 유사도

```python
def solution(s1, s2):
    return len(set(s1)&set(s2))
```



## 특정 문자 제거하기

```python
def solution(my_string, letter):
    return my_string.replace(letter, '')
```



## 옷 가게 할인 받기

```python
def solution(price):
    return price*80//100 if price>=500000 else price*90//100 if price>=300000 else price*95//100 if price >= 100000 else price
```



## 편지

```python
def solution(message):
    return len(message)*2
```



## 순서쌍의 개수

```python
def solution(n):
    return len([x for x in range(1, n+1) if not n%x])
```



## 모음 제거

```python
def solution(my_string):
    return my_string.replace('a', '').replace('e', '').replace('i', '').replace('o', '').replace('u', '')
```



## 숨어 있는 숫자의 덧셈 (1)

```python
def solution(my_string):
    return sum(int(x) for x in my_string if str.isdigit(x))
```



## 짝수는 싫어요

```python
def solution(n):
    return list(range(1, n+1, 2))
```



## 자릿수 더하기

```python
def solution(n):
    return sum(int(x) for x in str(n))
```



## 문자열 안에 문자열

```python
def solution(str1, str2):
    return (int(str2 in str1)-1)*(-1)+1
```



## 개미 군단

```python
def solution(hp):
    return hp//5+hp%5//3+hp%5%3
```



## 제곱수 판별하기

```python
def solution(n):
    return 1 if int(n**0.5)**2==n else 2
```



## 암호 해독

```python
def solution(cipher, code):
    return cipher[code-1::code]
```



## 종이 자르기

```python
def solution(M, N):
    return M*N-1
```



## 대문자와 소문자

```python
def solution(my_string):
    return "".join(x.lower() if x.isupper() else x.upper() for x in my_string)

# 다른 풀이
def solution(my_string):
    return my_string.swapcase()
```



## 가위 바위 보

```python
def solution(rsp):
    return "".join({'2':'0', '0':'5', '5':'2'}[x] for x in rsp)
```



## 세균 증식

```python
def solution(n, t):
    return n*2**t
```



## 문자열 정렬하기 (1)

```python
def solution(my_string):
    return sorted(int(x) for x in my_string if str.isdigit(x))
```



## 주사위의 개수

```python
def solution(box, n):
    return eval('*'.join(str(x//n) for x in box))
```



## 직각삼각형 출력하기

```python
for i in range(int(input())): print('*'*(i+1))
```



## 인덱스 바꾸기

```python
def solution(my_string, num1, num2):
    tmp = list(my_string)
    tmp[num1], tmp[num2] = tmp[num2], tmp[num1]
    return "".join(tmp)
```



## 최댓값 만들기 (2)

```python
def solution(numbers):
    numbers.sort()
    return max(numbers[0]*numbers[1], numbers[-1]*numbers[-2])
```



## n의 배수 만들기

```python
def solution(n, numlist):
    return [x for x in numlist if not x%n]
```



## 배열 회전시키기

```python
def solution(numbers, direction):
    return numbers[-1:]+numbers[:-1] if direction=="right" else numbers[1:]+numbers[:1]
```



## 외계행성의 나이

```python
def solution(age):
    return "".join(chr(int(x)+97) for x in str(age))
```



## 가장 큰 수 찾기

```python
def solution(array):
    return max([[v, i] for i, v in enumerate(array)])
```



## 피자 나눠 먹기 (2)

```python
def solution(n):
    import math
    return n//math.gcd(n, 6)
```



## 정수 부분

```python
def solution(flo):
    return int(flo)
```



## n의 배수

```python
def solution(num, n):
    return 0 if num%n else 1
```



## 대문자로 바꾸기

```python
def solution(myString):
    return myString.upper()
```



## 문자열로 변환

```python
def solution(n):
    return str(n)
```



## 문자열 곱하기

```python
def solution(my_string, k):
    return my_string*k
```



## 문자 리스트를 문자열로 변환하기

```python
def solution(arr):
    return "".join(arr)
```



## 공배수

```python
def solution(number, n, m):
    return 0 if number%n or number%m else 1
```



## 소문자로 바꾸기

```python
def solution(myString):
    return myString.lower()
```



## 문자열을 정수로 변환하기

```python
def solution(n_str):
    return int(n_str)
```



## 문자열의 앞의 n글자

```python
def solution(my_string, n):
    return my_string[:n]
```



## n번째 원소까지

```python
def solution(num_list, n):
    return num_list[:n]
```



## flag에 따라 다른 값 반환하기

```python
def solution(a, b, flag):
    return a+b if flag else a-b
```



## 길이에 따른 연산

```python
def solution(num_list):
    return sum(num_list) if len(num_list)>10 else eval('*'.join([str(n) for n in num_list]))
```



## 정수 찾기

```python
def solution(num_list, n):
    return 1 if n in num_list else 0
```



## 카운트 업

```python
def solution(start, end):
    return list(range(start, end+1))
```



## n보다 커질 때까지 더하기

```python
def solution(numbers, n):
    answer = 0
    for num in numbers:
        answer += num
        if answer > n:
            return answer
```

