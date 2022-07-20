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

