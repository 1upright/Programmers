# MySQL

## 상위 n개 레코드

```mysql
SELECT name FROM animal_ins ORDER BY datetime LIMIT 1;
```



## 여러 기준으로 정렬하기

```mysql
SELECT animal_id, name, datetime FROM animal_ins ORDER BY name, datetime DESC;
```



## 역순 정렬하기

```mysql
SELECT name, datetime FROM animal_ins ORDER BY animal_id DESC;
```



## 동물의 아이디와 이름

```mysql
SELECT animal_id, name FROM animal_ins ORDER BY animal_id;
```



## 어린 동물 찾기

```mysql
SELECT animal_id, name FROM animal_ins WHERE intake_condition != "Aged" ORDER BY animal_id;
```



## 아픈 동물 찾기

```mysql
SELECT animal_id, name FROM animal_ins WHERE intake_condition = "Sick" ORDER BY animal_id;
```



## 나이 정보가 없는 회원 수 구하기

```mysql
SELECT COUNT(*) as users FROM user_info WHERE age IS NULL;
```



## 강원도에 위치한 생산공장 목록 출력하기

```mysql
SELECT factory_id, factory_name, address FROM food_factory WHERE address LIKE "강원도%" ORDER BY factory_id;
```



## 이름이 있는 동물의 아이디

```mysql
SELECT animal_id FROM animal_ins WHERE name IS NOT NULL;
```



## 최솟값 구하기

```mysql
SELECT MIN(datetime) FROM animal_ins;
```



## 중복 제거하기

```mysql
-- DISTINCT 절 --
SELECT COUNT(DISTINCT(name)) FROM animal_ins;
```



## 동물 수 구하기

```mysql
SELECT COUNT(*) FROM animal_ins;
```



## 동명 수 구하기

```mysql
-- HAVING 절 --
SELECT name, COUNT(name) FROM animal_ins GROUP BY name HAVING COUNT(name)>=2 ORDER BY name;
```



## 이름에 el이 들어가는 동물 찾기

```mysql
SELECT animal_id, name FROM animal_ins WHERE animal_type = "Dog" AND name LIKE "%el%" ORDER BY name;
```



## 경기도에 위치한 식품창고 목록 출력하기

```mysql
-- ifnull 함수 --
SELECT warehouse_id, warehouse_name, address, IFNULL(freezer_yn, 'N') AS freezer_yn FROM food_warehouse WHERE address LIKE "경기도%";

-- coalesce 함수 --
SELECT warehouse_id, warehouse_name, address, COALESCE(freezer_yn, 'N') AS freezer_yn FROM food_warehouse WHERE address LIKE "경기도%";
```



## NULL 처리하기

```mysql
SELECT animal_type, COALESCE(name, "No name") AS name, sex_upon_intake FROM animal_ins;
```



## DATETIME에서 DATE로 형 변환

```mysql
-- date_format 함수 --
SELECT animal_id, name, date_format(datetime, "%Y-%m-%d") AS 날짜 FROM animal_ins;
```



## 가장 비싼 상품 구하기

```mysql
SELECT MAX(price) AS MAX_PRICE FROM product;
```



## 이름이 없는 동물의 아이디

```mysql
SELECT animal_id FROM animal_ins WHERE name IS NULL;
```



## 조건에 맞는 회원수 구하기

```mysql
SELECT COUNT(*) FROM user_info WHERE date_format(joined, "%Y") = 2021 AND age BETWEEN 20 AND 29;
```



## 가격이 제일 비싼 식품의 정보 출력하기

```mysql
SELECT * FROM food_product ORDER BY price DESC LIMIT 1;
```



## 고양이와 개는 몇 마리 있을까

```mysql
SELECT animal_type, COUNT(animal_type) AS "count" FROM animal_ins GROUP BY animal_type ORDER BY FIELD(animal_type, "Cat", "Dog");
```



## 중성화 여부 확인하기

```mysql
-- case 문, when/then/else/end --
SELECT animal_id, name, (CASE WHEN sex_upon_intake LIKE "%Neutered%" OR sex_upon_intake LIKE "%Spayed%" THEN 'O' ELSE 'X' END) AS 중성화 FROM animal_ins;
```



## 흉부외과 또는 일반외과 의사 목록 출력하기

```mysql
SELECT dr_name, dr_id, mcdp_cd, date_format(hire_ymd, "%Y-%m-%d") FROM doctor WHERE mcdp_cd = "CS" OR mcdp_cd = "GS" ORDER BY hire_ymd DESC, dr_name;
```



## 입양 시각 구하기(1)

```mysql
SELECT hour(datetime) AS hour, COUNT(hour(datetime)) AS "count" FROM animal_outs WHERE hour(datetime) BETWEEN 9 AND 19 GROUP BY hour(datetime) ORDER BY hour(datetime);
```



## 카테고리 별 상품 개수 구하기

```mysql
-- substr 함수 --
SELECT SUBSTR(product_code, 1, 2) AS category, COUNT(*) FROM product GROUP BY SUBSTR(product_code, 1, 2);

-- left 함수 --
SELECT LEFT(product_code, 2) AS category, COUNT(*) FROM product GROUP BY LEFT(product_code, 2);
```



## 12세 이하인 여자 환자 목록 출력하기

```mysql
SELECT pt_name, pt_no, gend_cd, age, COALESCE(tlno, "NONE") FROM patient WHERE AGE<=12 AND gend_cd='W' ORDER BY age DESC, pt_name;
```



## 인기있는 아이스크림

```mysql
SELECT flavor FROM first_half ORDER BY total_order DESC, shipment_id;
```



## 오랜 기간 보호한 동물 (1)

```mysql
-- NOT IN 절 --
SELECT name, datetime FROM animal_ins WHERE animal_id NOT IN (SELECT animal_id FROM animal_outs) ORDER BY datetime LIMIT 3;

-- LEFT JOIN/ON --
SELECT i.name, i.datetime 
FROM animal_ins i 
LEFT JOIN animal_outs o ON i.animal_id = o.animal_id WHERE o.animal_id IS NULL
ORDER BY datetime LIMIT 3;
```



## 진료과별 총 예약 횟수 출력하기

```mysql
SELECT mcdp_cd AS "진료과코드", COUNT(*) AS "5월예약건수" FROM appointment WHERE apnt_ymd LIKE "2022-05%" GROUP BY mcdp_cd ORDER BY 2, 1
```



## 자동차 종류 별 특정 옵션이 포함된 자동차 수 구하기

```mysql
SELECT car_type, COUNT(*) AS cars FROM car_rental_company_car 
WHERE options LIKE "%통풍시트%" 
OR options LIKE "%열선시트%" 
OR options LIKE "%가죽시트%"
GROUP BY car_type ORDER BY car_type
```



## 있었는데요 없었습니다

```mysql
SELECT i.animal_id, i.name FROM animal_ins i
LEFT JOIN animal_outs o ON i.animal_id=o.animal_id WHERE i.datetime>o.datetime
ORDER BY i.datetime
```



## 오랜 기간 보호한 동물 (2)

```mysql
SELECT i.animal_id, i.name FROM animal_ins i
LEFT JOIN animal_outs o ON i.animal_id=o.animal_id ORDER BY o.datetime-i.datetime DESC LIMIT 2;
```



## 상품 별 오프라인 매출 구하기

```mysql
SELECT p.product_code, sum(p.price*s.sales_amount) AS sales FROM product p
LEFT JOIN offline_sale s ON p.product_id=s.product_id GROUP BY 1 ORDER BY 2 DESC, 1;
```



## 보호소에서 중성화한 동물

```mysql
SELECT i.animal_id, i.animal_type, i.name FROM animal_ins i 
LEFT JOIN animal_outs o ON i.animal_id=o.animal_id
WHERE i.sex_upon_intake LIKE 'Intact%' AND (o.sex_upon_outcome LIKE 'Spayed%' OR o.sex_upon_outcome Like 'Neutered%');
```



## 카테고리 별 도서 판매량 집계하기

```mysql
SELECT b.category, SUM(s.sales) AS total_sales FROM book b
LEFT JOIN book_sales s ON b.book_id=s.book_id WHERE s.sales_date LIKE '2022-01%' GROUP BY 1 ORDER BY 1;
```



## 조건에 맞는 도서와 저자 리스트 출력하기

```mysql
SELECT b.book_id, a.author_name, date_format(b.published_date,'%Y-%m-%d') as published_date FROM book b
LEFT JOIN author a ON b.author_id=a.author_id WHERE b.category LIKE '경제' ORDER BY 3;
```



## 루시와 엘라 찾기

```mysql
SELECT animal_id, name, sex_upon_intake from animal_ins
WHERE name IN ('Lucy','Ella','Pickle','Rogan','Sabrina','Mitty') ORDER BY 1;
```



## 조건별로 분류하여 주문상태 출력하기

```mysql
SELECT order_id, product_id, date_format(out_date, "%Y-%m-%d") AS out_date,
    CASE WHEN out_date<="2022-05-01" THEN "출고완료"
    WHEN out_date>"2022-05-01" THEN "출고대기"
    ELSE "출고미정"
    END "출고여부"
FROM food_order;
```



## 성분으로 구분한 아이스크림 총 주문량

```mysql
SELECT i.ingredient_type, sum(f.total_order) AS total_order FROM icecream_info i
LEFT JOIN first_half f ON i.flavor=f.flavor GROUP BY 1;
```



## 조건에 맞는 사용자와 총 거래금액 조회하기

```mysql
SELECT u.user_id, u.nickname, SUM(b.price) AS total_sales FROM used_goods_user u
LEFT JOIN used_goods_board b ON u.user_id=b.writer_id 
WHERE b.status LIKE "DONE" GROUP BY 1 HAVING SUM(b.price)>=700000 ORDER BY 3;
```

