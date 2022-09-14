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

