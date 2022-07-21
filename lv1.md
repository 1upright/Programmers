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
```

