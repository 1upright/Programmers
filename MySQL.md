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



## 조건에 맞는 도서 리스트 출력하기

```mysql
SELECT book_id, date_format(published_date, "%Y-%m-%d") AS published_date FROM book WHERE category LIKE "인문" AND published_date LIKE "2021%" ORDER BY published_date;
```



## 가격대 별 상품 개수 구하기

```mysql
SELECT price-price%10000 AS price_group, COUNT(*) AS products FROM product GROUP BY 1 ORDER BY 1;
```



## 모든 레코드 조회하기

```mysql
SELECT * FROM animal_ins ORDER BY animal_id;
```



## 평균 일일 대여 요금 구하기

```mysql
SELECT ROUND(AVG(daily_fee)) AS average_fee FROM car_rental_company_car WHERE car_type LIKE "SUV";
```



## 3월에 태어난 여성 회원 목록 출력하기

```mysql
SELECT member_id, member_name, gender, date_format(date_of_birth, "%Y-%m-%d") AS date_of_birth from member_profile WHERE date_of_birth LIKE "%-03-%" AND gender='W' AND tlno IS NOT NULL ORDER BY 1;
```



## 즐겨찾기가 가장 많은 식당 정보 출력하기

```mysql
SELECT food_type, rest_id, rest_name, favorites FROM rest_info
WHERE (food_type, favorites) IN (SELECT food_type, MAX(favorites) FROM rest_info GROUP BY food_type) GROUP BY 1 ORDER BY 1 DESC;
```



## 대여 기록이 존재하는 자동차 리스트 구하기

```mysql
SELECT c.car_id FROM car_rental_company_car c
LEFT JOIN car_rental_company_rental_history h ON c.car_id=h.car_id
WHERE h.start_date>='2022-10-01' AND c.car_type='세단'
GROUP BY c.car_id ORDER BY c.car_id DESC;
```



## 식품분류별 가장 비싼 식품의 정보 조회하기

```mysql
SELECT category, price, product_name FROM food_product
WHERE price IN (SELECT max(price) FROM food_product GROUP BY category)
AND category IN ('식용유', '과자', '국', '김치')
GROUP BY 1 ORDER BY 2 DESC;
```



## 없어진 기록 찾기

```mysql
SELECT animal_id, name FROM animal_outs WHERE animal_id not in (SELECT animal_id FROM animal_ins) ORDER BY animal_id;
```



## 5월 식품들의 총매출 조회하기

```mysql
SELECT p.product_id, p.product_name, sum(o.amount*p.price) AS total_sales FROM food_product p
LEFT JOIN food_order o ON p.product_id=o.product_id
WHERE YEAR(o.produce_date)=2022 AND MONTH(o.produce_date)=5 GROUP BY product_id ORDER BY 3 DESC, 1;
```



## 재구매가 일어난 상품과 회원 리스트 구하기

```mysql
SELECT user_id, product_id FROM online_sale GROUP BY user_id, product_id HAVING COUNT(*)>1 ORDER BY 1, 2 DESC;
```



## 과일로 만든 아이스크림 고르기

```mysql
SELECT f.flavor FROM first_half f LEFT JOIN icecream_info i ON f.flavor=i.flavor
WHERE f.total_order>3000 AND i.ingredient_type='fruit_based'
ORDER BY f.total_order DESC;
```



## 최댓값 구하기

```mysql
SELECT max(datetime) FROM animal_ins;
```



## 조건에 맞는 사용자 정보 조회하기

```mysql
SELECT user_id, nickname, CONCAT_WS(' ', u.city, u.street_address1, u.street_address2) AS 전체주소, CONCAT_WS('-', SUBSTR(u.tlno, 1, 3), SUBSTR(u.tlno, 4, LENGTH(u.tlno) - 7), SUBSTR(u.tlno, -4, 4)) AS 전화번호
FROM used_goods_board b LEFT JOIN used_goods_user u ON b.writer_id=u.user_id GROUP BY b.writer_id HAVING COUNT(b.writer_id)>=3 ORDER BY 1 DESC;
```



## 특정 옵션이 포함된 자동차 리스트 구하기

```mysql
SELECT * FROM car_rental_company_car WHERE options LIKE "%네비게이션%" ORDER BY car_id DESC;
```



## 조건에 부합하는 중고거래 상태 조회하기

```mysql
SELECT board_id, writer_id, title, price, CASE
    WHEN STATUS = 'SALE' THEN '판매중'
    WHEN STATUS = 'RESERVED' THEN '예약중'
    WHEN STATUS = 'DONE' THEN '거래완료'
    END AS STATUS
FROM used_goods_board
WHERE created_date='2022-10-05' ORDER BY board_id DESC;
```



## 자동차 대여 기록에서 장기/단기 대여 구분하기

```mysql
SELECT history_id, car_id, DATE_FORMAT(start_date,'%Y-%m-%d') AS start_date, DATE_FORMAT(end_date,'%Y-%m-%d') AS end_date,
(CASE WHEN DATEDIFF(end_date, start_date) < 29
    THEN '단기 대여'
    ELSE '장기 대여'
END) AS rent_type FROM car_rental_company_rental_history
WHERE DATE_FORMAT(start_date,'%Y-%m') = '2022-09'
ORDER BY 1 DESC;
```



## 자동차 대여 기록에서 대여중 / 대여 가능 여부 구분하기

```mysql
SELECT car_id,
(CASE WHEN car_id in (SELECT car_id FROM car_rental_company_rental_history WHERE start_date<='2022-10-16' AND '2022-10-16'<=end_date)
    THEN '대여중'
    ELSE '대여 가능'
    END) AS availability FROM car_rental_company_rental_history
    GROUP BY 1 ORDER BY 1 DESC;
```



## 자동차 평균 대여 기간 구하기

```mysql
SELECT car_id, ROUND(AVG(DATEDIFF(end_date, start_date))+1, 1) AS average_duration FROM car_rental_company_rental_history
    GROUP BY 1
    HAVING average_duration>=7
    ORDER BY 2 DESC, 1 DESC;
```



## 조건에 부합하는 중고거래 댓글 조회하기

```mysql
SELECT b.title, b.board_id, r.reply_id, r.writer_id, r.contents, DATE_FORMAT(r.created_date, '%Y-%m-%d') AS created_date
FROM used_goods_board b JOIN used_goods_reply r ON b.board_id=r.board_id
WHERE b.created_date LIKE '2022-10%'
ORDER BY created_date, b.title;
```



## 취소되지 않은 진료 예약 조회하기

```mysql
SELECT a.apnt_no, p.pt_name, p.pt_no, a.mcdp_cd, d.dr_name, a.apnt_ymd
FROM appointment a JOIN patient p ON a.pt_no = p.pt_no JOIN doctor d ON a.mddr_id = d.dr_id
WHERE a.apnt_ymd LIKE '2022-04-13%' AND a.mcdp_cd = 'CS' AND a.apnt_cncl_yn = 'N'
ORDER BY 6;
```



## 년, 월, 성별 별 상품 구매 회원 수 구하기

```mysql
SELECT YEAR(s.sales_date) AS year, MONTH(s.sales_date) AS month, u.gender, COUNT(DISTINCT u.user_id) AS user
FROM user_info u JOIN online_sale s ON s.user_id=u.user_id
WHERE u.gender IS NOT NULL
GROUP BY 1, 2, 3
ORDER BY 1, 2, 3;
```



## 서울에 위치한 식당 목록 출력하기

```mysql
SELECT i.rest_id, i.rest_name, i.food_type, i.favorites, i.address, ROUND(AVG(r.review_score), 2) AS score
FROM rest_review r JOIN rest_info i ON r.rest_id=i.rest_id
WHERE i.address LIKE '서울%'
GROUP BY 1 ORDER BY 6 DESC, 4 DESC;
```



## 헤비 유저가 소유한 장소

```mysql
SELECT * FROM places
WHERE host_id in (SELECT host_id FROM places GROUP BY host_id HAVING COUNT(*)>1)
ORDER BY id;
```



## 주문량이 많은 아이스크림들 조회하기

```mysql
SELECT f.flavor FROM first_half f JOIN july j ON f.flavor=j.flavor
GROUP BY flavor ORDER BY SUM(f.total_order) + SUM(j.total_order) DESC LIMIT 3;
```



## 대여 횟수가 많은 자동차들의 월별 대여 횟수 구하기

```mysql
SELECT MONTH(start_date) AS month, car_id, COUNT(history_id) AS records FROM car_rental_company_rental_history
WHERE car_id in (SELECT car_id FROM car_rental_company_rental_history
                WHERE start_date BETWEEN '2022-08-01' AND '2022-10-31'
                GROUP BY 1 HAVING COUNT(car_id)>=5)
AND start_date BETWEEN '2022-08-01' AND '2022-10-31'
GROUP BY 1, 2
ORDER BY 1, 2 DESC;
```



## 오프라인/온라인 판매 데이터 통합하기

```mysql
SELECT DATE_FORMAT(sales_date, "%Y-%m-%d") AS sales_date, product_id, user_id, sales_amount FROM online_sale
WHERE sales_date BETWEEN "2022-03-01" AND "2022-03-31"
UNION ALL
SELECT DATE_FORMAT(sales_date, "%Y-%m-%d") AS sales_date, product_id, NULL AS user_id, sales_amount FROM offline_sale
WHERE sales_date BETWEEN "2022-03-01" AND "2022-03-31"
ORDER BY 1, 2, 3;
```



## 그룹별 조건에 맞는 식당 목록 출력하기

```mysql
SELECT p.member_name, r.review_text, DATE_FORMAT(r.review_date, '%Y-%m-%d') AS review_date
FROM member_profile p JOIN rest_review r ON p.member_id=r.member_id
WHERE p.member_id IN (SELECT * FROM (SELECT member_id FROM rest_review GROUP BY member_id ORDER BY COUNT(*) DESC LIMIT 1) AS sub)
ORDER BY 3, 2;
```



## 우유와 요거트가 담긴 장바구니

```mysql
SELECT cart_id FROM cart_products WHERE name='Milk' AND
cart_id IN (SELECT cart_id FROM cart_products WHERE name='Yogurt')
ORDER BY 1;
```



## 조회수가 가장 많은 중고거래 게시판의 첨부파일 조회하기

```mysql
SELECT CONCAT("/home/grep/src/", board_id, "/", file_id, file_name, file_ext) file_path
FROM used_goods_file
WHERE board_id=(SELECT board_id FROM used_goods_board ORDER BY views DESC LIMIT 1)
ORDER BY file_id DESC;
```



## 저자 별 카테고리 별 매출액 집계하기

```mysql
SELECT b.author_id, a.author_name, b.category, SUM(s.sales*b.price) total_sales
FROM book b JOIN author a ON b.author_id=a.author_id
JOIN book_sales s ON b.book_id=s.book_id
WHERE DATE_FORMAT(s.sales_date,'%Y-%m') = '2022-01'
GROUP BY 1, 3
ORDER BY 1, 3 DESC;
```

