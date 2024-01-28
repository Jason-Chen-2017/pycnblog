                 

# 1.背景介绍

Elasticsearch의 데이터 클린징과 처리
=================================

저자: 절 and 계산 프로그램 디자인 아트

## 소개

---

### Elasticsearch란?

- Elasticsearch는 검색과 분석 엔진으로, 확장 가능하고 실시간으로 데이터를 인덱싱하고 검색할 수 있습니다.
- 오픈 소스 기술로, Java 위에서 구동됩니다.
- RESTful API를 통해 다양한 언어와 플랫폼에서 사용이 가능합니다.

### 데이터 클린징과 처리란?

- 데이터 클린징은 잘못된 형식, 누락, 중복 등의 데이터 오류를 제거하는 것을 의미합니다.
- 데이터 처리는 데이터를 가공하여 원하는 결과를 얻도록 하는 것을 의미합니다.

## Elasticsearch의 데이터 클린징과 처리

### 1. 데이터 유효성 검사

#### 1.1. 문자열 길이 검증

- `script` 필드를 사용하여 문자열 길이 검증을 수행할 수 있습니다.

   ```json
   {
     "query": {
       "bool": {
         "must_not": [
           {
             "script": {
               "source": """
                 if (doc['title'].value.length < 5 || doc['title'].value.length > 20) {
                  return false;
                 }
                 return true;
               """
             }
           }
         ]
       }
     }
   }
   ```

#### 1.2. 날짜 유효성 검사

- `range` 필터를 사용하여 날짜 유효성 검사를 수행할 수 있습니다.

   ```json
   {
     "query": {
       "bool": {
         "must_not": [
           {
             "range": {
               "date": {
                 "gte": "2022-01-01",
                 "lte": "2022-12-31",
                 "format": "yyyy-MM-dd"
               }
             }
           }
         ]
       }
     }
   }
   ```

### 2. 데이터 정규화

#### 2.1. 문자열 변환

- `script` 필드를 사용하여 문자열 변환을 수행할 수 있습니다.

   ```json
   {
     "update_by_query": {
       "query": {
         "match_all": {}
       },
       "script": {
         "source": """
           ctx._source.title = ctx._source.title.toLowerCase();
         """
       }
     }
   }
   ```

#### 2.2. 숫자 변환

- `script` 필드를 사용하여 숫자 변환을 수행할 수 있습니다.

   ```json
   {
     "update_by_query": {
       "query": {
         "match_all": {}
       },
       "script": {
         "source": """
           ctx._source.price = (double) Math.round(ctx._source.price * 100);
         """
       }
     }
   }
   ```

### 3. 데이터 삭제

#### 3.1. 특정 값을 가지는 문서 삭제

- `delete_by_query` 플러그인을 사용하여 특정 값을 가지는 문서를 삭제할 수 있습니다.

   ```json
   {
     "delete_by_query": {
       "query": {
         "term": {
           "status": "deleted"
         }
       }
     }
   }
   ```

#### 3.2. 시간 경과에 따른 문서 삭제

- `_update_by_query` API를 사용하여 시간 경과에 따른 문서를 삭제할 수 있습니다.

   ```json
   {
     "update_by_query": {
       "query": {
         "range": {
           "timestamp": {
             "lt": "now-1d"
           }
         }
       },
       "script": {
         "source": """
           ctx._version++;
           ctx.op = "delete";
         """
       }
     }
   }
   ```

### 4. 데이터  Aggregation

#### 4.1. 단순 Aggregation

- `aggs` 필드를 사용하여 단순 Aggregation을 수행할 수 있습니다.

   ```json
   {
     "size": 0,
     "aggs": {
       "genres": {
         "terms": {
           "field": "genre.keyword"
         }
       }
     }
   }
   ```

#### 4.2. Metric Aggregation

- `aggs` 필드를 사용하여 Metric Aggregation을 수행할 수 있습니다.

   ```json
   {
     "size": 0,
     "aggs": {
       "avg_rating": {
         "avg": {
           "field": "rating"
         }
       }
     }
   }
   ```

### 5. 실제 애플리케이션 시나리오

#### 5.1. E-commerce 쇼핑몰

- 상품 검색 및 필터링
- 주문 내역 분석 및 보고
- 재고 관리

#### 5.2. 로그 분석

- 로그 데이터 집계 및 분석
- 장애 진단 및 경보
- 보안 모니터링

### 6. 도구 및 자료 추천

#### 6.1. Elasticsearch 공식 문서

- <https://www.elastic.co/guide/en/elasticsearch/>

#### 6.2. Elasticsearch  Korean Community

- <https:// elasticsearch-korea.github.io/>

#### 6.3. Elasticsearch - The Definitive Guide

- <https://www.elastic.co/guide/en/elasticsearch/guide/current/index.html>

### 7. 요약

---

#### 7.1. 미래의 발전 동향

- Elasticsearch와 Kibana의 통합 및 기능 개선
- 대규모 데이터 처리 기술의 발전

#### 7.2. 과제

- Elasticsearch의 성능 최적화 및 확장
- 보안 및 권한 관리 강화

### 8. 빈칸 채우기

---

#### Q1. Elasticsearch는 어떤 언어로 만들어졌습니까?

- Java입니다.

#### Q2. Elasticsearch에서 날짜 유효성 검사를 위해 어떤 필터를 사용합니까?

- `range` 필터를 사용합니다.

#### Q3. Elasticsearch에서 문자열을 소문자로 변환하려면 어떻게 해야 합니까?

- `script` 필드를 사용하여 `toLowerCase()` 함수를 호출하면 됩니다.