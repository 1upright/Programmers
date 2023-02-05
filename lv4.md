# lv4

## 도둑질

```python
def solution(money):
    n = len(money)
    dp1 = [0]*n
    dp2 = [0]*n
    
    dp1[0] = 0
    for i in range(1, n):
        dp1[i] = max(dp1[i-1], dp1[i-2]+money[i])
    
    dp2[0] = money[0]
    for i in range(1, n-1):
        dp2[i] = max(dp2[i-1], dp2[i-2]+money[i])
    
    return max(dp1[-1], dp2[-2])
```

