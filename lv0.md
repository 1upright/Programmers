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



## 문자열의 뒤의 n글자

```python
def solution(my_string, n):
    return my_string[-n:]
```



## n 번째 원소부터

```python
def solution(num_list, n):
    return num_list[n-1:]
```



## 부분 문자열

```python
def solution(str1, str2):
    return int(str1 in str2)
```



## 부분 문자열인지 확인하기

```python
def solution(my_string, target):
    return int(target in my_string)
```



## 첫 번째로 나오는 음수

```python
def solution(num_list):
    for i, v in enumerate(num_list):
        if v<0: return i
    return -1
```



## rny_string

```python
def solution(rny_string):
    return rny_string.replace('m', 'rn')
```



## 카운트 다운

```python
def solution(start, end):
    return list(range(start, end-1, -1))
```



## 조건에 맞게 수열 변환하기 3

```python
def solution(arr, k):
    return [x*k if k%2 else x+k for x in arr]
```



## 문자열 정수의 합

```python
def solution(num_str):
    return sum(int(x) for x in num_str)
```



## 공백으로 구분하기 1

```python
def solution(my_string):
    return my_string.split()
```



## 원소들의 곱과 합

```python
def solution(num_list):
    return int(eval('*'.join(str(n) for n in num_list)) < sum(num_list)**2)
```



## 주사위 게임 1

```py
def solution(a, b):
    return a*a+b*b if a%2 and b%2 else abs(a-b) if not a%2 and not b%2 else (a+b)*2
```



## 뒤에서 5등 위로

```python
def solution(num_list):
    return sorted(num_list)[5:]
```



## 이어 붙인 수

```python
def solution(num_list):
    return int("".join([str(x) for x in num_list if x%2]))+int("".join([str(x) for x in num_list if not x%2]))
```



## 조건에 맞게 수열 변환하기 1

```python
def solution(arr):
    return [x//2 if (x>=50 and not x%2) else x*2 if (x<50 and x%2) else x for x in arr]
```



## 글자 이어 붙여 문자열 만들기

```python
def solution(my_string, index_list):
    return "".join([my_string[n] for n in index_list])
```



## n개 간격의 원소들

```python
def solution(num_list, n):
    return num_list[::n]
```



## 문자열 붙여서 출력하기

```python
str1, str2 = input().strip().split(' ')
print(str1+str2)
```



## 뒤에서 5등까지

```python
def solution(num_list):
    return sorted(num_list)[:5]
```



## 마지막 두 원소

```python
def solution(num_list):
    return num_list + [num_list[-1]-num_list[-2] if num_list[-1]>num_list[-2] else num_list[-1]*2]
```



## 원하는 문자열 넣기

```python
def solution(myString, pat):
    return int(pat.lower() in myString.lower())
```



## A 강조하기

```python
def solution(myString):
    return myString.lower().replace('a', 'A')
```



## 배열에서 문자열 대소문자 변환하기

```python
def solution(strArr):
    return [s.upper() if i%2 else s.lower() for i, s in enumerate(strArr)]
```



## 수 조작하기 1

```python
def solution(n, control):
    return sum(1 if x=='w' else -1 if x=='s' else 10 if x=='d' else -10 for x in control)+n
```



## 배열 만들기

```python
def solution(n, k):
    return list(range(k, n+1, k))
```



## 접두사인지 확인하기

```python
def solution(my_string, is_prefix):
    return int(my_string.startswith(is_prefix))
```



## 더 크게 합치기

```python
def solution(a, b):
    return max(int(str(a)+str(b)), int(str(b)+str(a)))
```



## 홀짝 구분하기

```python
a = int(input())
print(f'{a} is {"odd" if a%2 else "even"}')
```



## 꼬리 문자열

```python
def solution(str_list, ex):
    return "".join(x for x in str_list if ex not in x)
```



## 홀수 vs 짝수

```python
def solution(num_list):
    return max(sum(num_list[::2]), sum(num_list[1::2]))
```



## 배열의 원소만큼 추가하기

```python
def solution(arr):
    return sum([[x]*x for x in arr], [])
```



## 접미사인지 확인하기

```python
def solution(my_string, is_suffix):
    return int(my_string[-len(is_suffix):]==is_suffix)
```



## 배열의 길이에 따라 다른 연산하기

```python
def solution(arr, n):
    return [v if i%2 else v+n for i, v in enumerate(arr)] if len(arr)%2 else [v if not i%2 else v+n for i, v in enumerate(arr)]
```



## 공백으로 구분하기 2

```python
def solution(my_string):
    return my_string.split()
```



## 특정한 문자를 대문자로 바꾸기

```python
def solution(my_string, alp):
    return my_string.replace(alp, alp.upper())
```



## 문자열 바꿔서 찾기

```python
def solution(myString, pat):
    return int(pat in myString.replace('A', 'C').replace('B', 'A').replace('C', 'B'))
```



## 덧셈식 출력하기

```python
a, b = map(int, input().strip().split(' '))
print(f'{a} + {b} = {a+b}')
```



## 배열 비교하기

```python
def solution(arr1, arr2):
    if len(arr1)>len(arr2):
        return 1
    elif len(arr1)<len(arr2):
        return -1
    elif sum(arr1)>sum(arr2):
        return 1
    elif sum(arr1)<sum(arr2):
        return -1
    return 0
```



## 0 떼기

```python
def solution(n_str):
    return str(int(n_str))
```



## l로 만들기

```python
def solution(myString):
    return "".join(chr(max(ord(x), 108)) for x in myString)
```



## 홀짝에 따라 다른 값 반환하기

```python
def solution(n):
    return sum(x*2+1 for x in range(n//2+1)) if n%2 else sum(x**2 for x in range(n//2+1))*4
```



## 5명씩

```python
def solution(names):
    return names[::5]
```



## 할 일 목록

```python
def solution(todo_list, finished):
    return [v for i, v in enumerate(todo_list) if not finished[i]]
```



## 배열의 원소 삭제하기

```python
def solution(arr, delete_list):
    return [x for x in arr if x not in delete_list]
```



## 순서 바꾸기

```python
def solution(num_list, n):
    return num_list[n:]+num_list[:n]
```



## 접미사 배열

```python
def solution(my_string):
    return sorted([my_string[i:] for i in range(len(my_string))])
```



## 특별한 이차원 배열 2

```python
def solution(arr):
    for i in range(len(arr)):
        for j in range(len(arr)):
            if arr[i][j] != arr[j][i]:
                return 0
    return 1
```



## 부분 문자열 이어 붙여 문자열 만들기

```python
def solution(my_strings, parts):
    return "".join(v[parts[i][0]:parts[i][1]+1] for i, v in enumerate(my_strings))
```



## 특별한 이차원 배열 1

```python
def solution(n):
    return [[int(i==j) for i in range(n)] for j in range(n)]
```



## 배열 만들기 3

```python
def solution(arr, intervals):
    return arr[intervals[0][0]:intervals[0][1]+1]+arr[intervals[1][0]:intervals[1][1]+1]
```



## 간단한 식 계산하기

```python
def solution(binomial):
    return eval(binomial)
```



## ad 제거하기

```python
def solution(strArr):
    return [s for s in strArr if "ad" not in s]
```



## x 사이의 개수

```python
def solution(myString):
    return [len(v) for v in myString.split('x')]
```



## 9로 나눈 나머지

```python
def solution(number):
    return sum(int(x) for x in number)%9
```



## 369 게임

```python
def solution(order):
    return str(order).count('3')+str(order).count('6')+str(order).count('9')
```



## 문자열 돌리기

```python
for c in input(): print(c)
```



## 두 수의 연산값 비교하기

```python
def solution(a, b):
    return max(int(str(a)+str(b)), 2*a*b)
```



## 숫자 찾기

```python
def solution(num, k):
    return str(num).index(str(k))+1 if str(k) in str(num) else -1
```



## 약수 구하기

```python
def solution(n):
    return [x for x in range(1, n+1) if not n%x]
```



## 콜라츠 수열 만들기

```python
def solution(n):
    answer = [n]
    while n != 1:
        if n%2:
            n = n*3+1
        else:
            n //= 2
        answer.append(n)
    return answer
```



## 가까운 1 찾기

```python
def solution(arr, idx):
    for i, v in enumerate(arr):
        if i>=idx and v==1:
            return i
    return -1
```



## 문자열 잘라서 정렬하기

```python
def solution(myString):
    return sorted([x for x in myString.split('x') if x])
```



## 문자열 정렬하기 (2)

```python
def solution(my_string):
    return "".join(sorted(my_string.lower()))
```



## 주사위 게임 2

```python
def solution(a, b, c):
    return (a+b+c)*(a**2+b**2+c**2)*(a**3+b**3+c**3) if a==b==c else (a+b+c)*(a**2+b**2+c**2) if a==b or b==c or a==c else a+b+c
```



## 문자열 섞기

```python
def solution(str1, str2):
    return "".join(str1[i]+str2[i] for i in range(len(str1)))
```



## 중복된 문자 제거

```python
def solution(my_string):
    return "".join(dict.fromkeys(my_string))
```



## 합성수 찾기

```python
def solution(n):
    cnt = 0
    for i in range(2, n+1):
        for j in range(2, i):
            if not i%j:
                cnt += 1
                break
    return cnt
```



## 수 조작하기

```python
def solution(numLog):
    dic = {-1:'w', 1:'s', -10:'d', 10:'a'}
    return "".join(dic[numLog[i]-numLog[i+1]] for i in range(len(numLog)-1))
```



## 배열 만들기 5

```python
def solution(intStrs, k, s, l):
    return [int(x[s:s+l]) for x in intStrs if int(x[s:s+l])>k]
```



## 모스부호 (1)

```python
def solution(letter):
    morse = { 
        '.-':'a','-...':'b','-.-.':'c','-..':'d','.':'e','..-.':'f',
        '--.':'g','....':'h','..':'i','.---':'j','-.-':'k','.-..':'l',
        '--':'m','-.':'n','---':'o','.--.':'p','--.-':'q','.-.':'r',
        '...':'s','-':'t','..-':'u','...-':'v','.--':'w','-..-':'x',
        '-.--':'y','--..':'z'
    }
    return "".join(morse[x] for x in letter.split())
```



## 날짜 비교하기

```python
def solution(date1, date2):
    return int(date1<date2)
```



## 2차원으로 만들기

```python
def solution(num_list, n):
    return [[num_list[i*n+j] for j in range(n)] for i in range(len(num_list)//n)]
```



## 팩토리얼

```python
def solution(n):
    val = 1
    for i in range(1, 12):
        val *= i
        if val>n: return i-1
```



## A를 B로 바꾸기

```python
def solution(before, after):
    return int(sorted(before)==sorted(after))
```



## 등차수열의 특정한 항만 더하기

```python
def solution(a, d, included):
    return sum(a+i*d for i, v in enumerate(included) if v)
```



## k의 개수

```python
def solution(i, j, k):
    return sum(str(x).count(str(k)) for x in range(i, j+1))
```



## 세로 읽기

```python
def solution(my_string, m, c):
    return my_string[c-1::m]
```



## 수열과 구간 쿼리 1

```python
def solution(arr, queries):
    for s, e in queries:
        for i in range(s, e+1):
            arr[i] += 1
    return arr
```



## 이차원 배열 대각선 순회하기

```python
def solution(board, k):
    answer = 0
    for i in range(len(board)):
        for j in range(len(board[0])):
            if i+j<=k:
                answer += board[i][j]
    return answer
```



## 진료순서 정하기

```python
def solution(emergency):
    return [sorted(emergency, reverse=True).index(x)+1 for x in emergency]
```



## 숨어있는 숫자의 덧셈 (2)

```python
def solution(my_string):
    import re
    return sum(int(x) for x in re.split('[A-Za-z]', my_string) if x)
```



## 문자열 뒤집기

```python
def solution(my_string, s, e):
    return my_string[:s]+my_string[s:e+1][::-1]+my_string[e+1:]
```



## 가까운 수

```python
def solution(array, n):
    return sorted(array, key = lambda x: (abs(x-n), x-n))[0]
```



## 빈 배열에 추가, 삭제하기

```python
def solution(arr, flag):
    answer = []
    for x, y in zip(arr, flag):
        if y:
            answer.extend([x]*2*x)
        else:
            answer = answer[:-x]
    return answer
```



## 간단한 논리 연산

```python
def solution(x1, x2, x3, x4):
    return (x1|x2)&(x3|x4)
```



## 수열과 구간 쿼리 3

```python
def solution(arr, queries):
    for i, j in queries: arr[i], arr[j] = arr[j], arr[i]
    return arr
```



## 문자열 반복해서 출력하기

```python
a, b = input().strip().split(' ')
b = int(b)
print(a*b)
```



## 세 개의 구분자

```python
def solution(myStr):
    arr = myStr.replace('a', ' ').replace('b', ' ').replace('c', ' ').split()
    return arr if arr else ["EMPTY"]
```



## 한 번만 등장한 문자

```python
def solution(s):
    return "".join(sorted(x for x in set(s) if s.count(x)==1))
```



## 7의 개수

```python
def solution(array):
    return str(array).count('7')
```



## 특수문자 출력하기

```python
print("!@#$%^&*(\\'\"<>?:;")
```



## 영어가 싫어요

```python
def solution(numbers):
    return int(numbers.replace("one", "1").replace("two", "2").replace("three", "3").replace("four", "4").replace("five", "5").replace("six", "6").replace("seven", "7").replace("eight", "8").replace("nine", "9").replace("zero", "0"))
```



## 잘라서 배열로 저장하기

```python
def solution(my_str, n):
    return [my_str[i:i+n] for i in range(0, len(my_str), n)]
```



## 이진수 더하기

```python
def solution(bin1, bin2):
    return bin(int(bin1, 2)+int(bin2, 2))[2:]
```



## 컨트롤 제트

```python
def solution(s):
    stack = []
    for x in s.split():
        if x == 'Z':
            stack.pop()
        else:
            stack.append(int(x))
    return sum(stack)
```



## 문자열 계산하기

```python
def solution(my_string):
    return eval(my_string)
```



## 특정 문자열로 끝나는 가장 긴 부분 문자열 찾기

```python
def solution(myString, pat):
    return pat.join(myString.split(pat)[:-1])+pat
```



## 글자 지우기

```python
def solution(my_string, indices):
    arr = list(my_string)
    for i in indices: arr[i] = ""
    return "".join(arr)
```



## 1로 만들기

```python
def solution(num_list):
    return sum(len(bin(x))-3 for x in num_list)
```



## 문자열이 몇 번 등장하는지 세기

```python
def solution(myString, pat):
    return sum(myString[i:i+len(pat)]==pat for i in range(len(myString)-len(pat)+1))
```



## 배열의 길이를 2의 거듭제곱으로 만들기

```python
def solution(arr):
    n, m = len(arr), 1
    while n > m: m *= 2
    return arr+[0]*(m-n)
```



## 2의 영역

```python
def solution(arr):
    s = e = -1
    for i in range(len(arr)):
        if arr[i] == 2:
            s = i
            break
    for i in range(len(arr)-1, -1, -1):
        if arr[i] == 2:
            e = i
            break
    return [-1] if s==-1 else arr[s:e+1]
```



## 리스트 자르기

```python
def solution(n, slicer, num_list):
    a, b, c = slicer
    return [num_list[:b+1], num_list[a:], num_list[a:b+1], num_list[a:b+1:c]][n-1]
```



## a와 b 출력하기

```python
a, b = map(int, input().strip().split(' '))
print(f'a = {a}')
print(f'b = {b}')
```



## 수열과 구간 쿼리 4

```python
def solution(arr, queries):
    for s, e, k in queries:
        for i in range(s, e+1):
            if not i%k:
                arr[i] += 1
    return arr
```



## 문자열 묶기

```python
def solution(strArr):
    from collections import Counter
    return Counter(len(x) for x in strArr).most_common()[0][1]
```



## 소인수분해

```python
def solution(n):
    arr, k = [], 2
    while n>1:
        if not n%k:
            n //= k
            arr.append(k)
        else:
            k += 1
    return sorted(list(set(arr)))
```



## 조건에 맞게 수열 변환하기 2

```python
def solution(arr):
    from copy import deepcopy
    cnt = 0
    while 1:
        arr2 = deepcopy(arr)
        for i, v in enumerate(arr):
            if v>=50 and not v%2:
                arr[i] = v//2
            elif v<50 and v%2:
                arr[i] = v*2+1
        if arr == arr2:
            return cnt
        cnt += 1
```



## 커피 심부름

```python
def solution(order):
    return sum(5000 if 'latte' in o else 4500 for o in order)
```



## qr code

```python
def solution(q, r, code):
    return code[r::q]
```



## 공 던지기

```python
def solution(numbers, k):
    return [(x-1)%len(numbers)+1 for x in range(1, k*2+1, 2)][-1]
```



## 문자 개수 세기

```python
def solution(my_string):
    result = [0]*52
    for x in my_string:
        if x.isupper():
            result[ord(x)-65] += 1
        else:
            result[ord(x)-71] += 1
    return result
```



## 구슬을 나누는 경우의 수

```python
def solution(balls, share):
    from math import comb
    return comb(balls, share)
```



## 배열 만들기 4

```python
def solution(arr):
    stk = []
    i = 0
    while i<len(arr):
        x = arr[i]
        if not stk:
            stk.append(x)
            i += 1
        elif stk[-1]<arr[i]:
            stk.append(x)
            i += 1
        else:
            stk.pop()
    return stk
```



## 두 수의 합

```python
def solution(a, b):
    return str(int(a)+int(b))
```



## 삼각형의 완성조건 (2)

```python
def solution(sides):
    return min(sides)*2-1
```



## 대소문자 바꿔서 출력하기

```python
print(input().swapcase())
```



## 문자열 겹쳐쓰기

```python
def solution(my_string, overwrite_string, s):
    return my_string[:s]+overwrite_string+my_string[s+len(overwrite_string):]
```



## 조건 문자열

```python
def solution(ineq, eq, n, m):
    return int(eval(str(n)+ineq+(eq if eq=='=' else '')+str(m)))
```



## 왼쪽 오른쪽

```python
def solution(str_list):
    for i, v in enumerate(str_list):
        if v == 'l':
            return str_list[:i]
        if v == 'r':
            return str_list[i+1:]
    return []
```



## 배열 만들기 6

```python
def solution(arr):
    i, stk = 0, []
    while i<len(arr):
        if not stk:
            stk.append(arr[i])
        elif stk[-1] == arr[i]:
            stk.pop()
        else:
            stk.append(arr[i])
        i += 1
    return stk if stk else [-1]
```



## 문자열 여러 번 뒤집기

```python
def solution(my_string, queries):
    for s, e in queries:
        my_string = my_string[:s]+my_string[s:e+1][::-1]+my_string[e+1:]
    return my_string
```



## 최빈값 구하기

```python
def solution(array):
    from collections import Counter
    cnt = Counter(array).most_common()
    return cnt[0][0] if len(cnt)==1 or cnt[0][1]!=cnt[1][1] else -1
```



## 무작위로 K개의 수 뽑기

```python
def solution(arr, k):
    result = []
    for x in arr:
        if x not in result: result.append(x)
        if len(result)==k: break
    return result + [-1]*(k-len(result))
```



## 정사각형으로 만들기

```python
def solution(arr):
    n, m = len(arr), len(arr[0])
    k = max(n, m)
    answer = [[0]*k for _ in range(k)]

    for i in range(n):
        for j in range(m):
            answer[i][j] = arr[i][j]

    return answer
```



## 문자열 출력하기

```python
print(input())
```



## 다음에 올 숫자

```python
def solution(common):
    a, b, c = common[:3]
    return common[-1]+b-a if a+c==b*2 else common[-1]*b//a
```



## 분수의 덧셈

```python
def solution(numer1, denom1, numer2, denom2):
    from math import gcd
    x = denom1*denom2//gcd(denom1, denom2)
    y = numer1*x//denom1+numer2*x//denom2
    z = gcd(x, y)
    return [y//z, x//z]
```



## 수열과 구간 쿼리 2

```python
def solution(arr, queries):
    result = []
    for s, e, k in queries:
        x = [arr[i] for i in range(s, e+1) if arr[i]>k]
        result.append(min(x) if x else -1)
    return result
```



## 외계어 사전

```python
def solution(spell, dic):
    for d in dic:
        if all(x in d for x in spell): return 1
    return 2
```



## 달리기 경주

```python
def solution(players, callings):
    ranking = {v:i for i, v in enumerate(players)}
    for calling in callings:
        i = ranking[calling]
        ranking[calling], ranking[players[i-1]] = i-1, i
        players[i], players[i-1] = players[i-1], players[i]
        
    return players
```



## 그림 확대

```python
def solution(picture, k):
    answer = []
    for p in picture: answer.extend(["".join(x*k for x in p)]*k)
    return answer
```



## 직사각형 넓이 구하기

```python
def solution(dots):
    l, r = min(dots), max(dots)
    return (r[0]-l[0])*(r[1]-l[1])
```

