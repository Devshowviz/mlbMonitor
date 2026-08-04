# mlbMonitor

MLB Stats API를 활용해 그날의 MLB 경기들의 승률을 계산하는 프로그램입니다.

## 기능

- 지정한 날짜(기본값: 오늘)의 MLB 경기 일정 조회
- 각 팀의 시즌 전적과 시즌 승률 계산
- 두 팀의 시즌 승률을 바탕으로 한 맞대결 예상 승리 확률 계산 (log5 공식)
- 이미 끝났거나 진행 중인 경기는 점수도 함께 표시

## 요구 사항

- Python 3.9 이상 (표준 라이브러리만 사용, 별도 패키지 설치 불필요)

## 사용법

```bash
# 오늘 경기 승률 보기
python mlb_win_rate.py

# 특정 날짜 경기 보기
python mlb_win_rate.py --date 2026-08-03

# JSON으로 출력 (다른 프로그램에서 활용할 때)
python mlb_win_rate.py --json
```

출력 예시:

```
=== 2026-08-03 MLB 경기 승률 (2경기) ===
원정팀                      전적    승률 | 홈팀                       전적    승률 |  홈팀 승리확률  상태
------------------------------------------------------------------------------------------------
San Francisco Giants       60-50  0.545 | Los Angeles Dodgers        70-40  0.636 |        59.4%  Final (3:5)
Boston Red Sox             55-55  0.500 | New York Yankees           66-44  0.600 |        60.0%  Scheduled
```

## 승리 확률 계산 방식 (log5)

두 팀의 시즌 승률 `pA`(홈), `pB`(원정)를 가지고 Bill James의 log5 공식으로
홈팀이 이길 확률을 추정합니다.

```
P(홈팀 승리) = (pA - pA·pB) / (pA + pB - 2·pA·pB)
```

- 두 팀 승률이 같으면 50%가 나옵니다.
- 아직 경기를 치르지 않은 팀(0승 0패)은 승률 0.5로 간주합니다.
- 홈 어드밴티지는 반영하지 않은 순수 전력 비교입니다.

## 데이터 출처

[MLB Stats API](https://statsapi.mlb.com)의 schedule 엔드포인트를 사용합니다.

```
GET https://statsapi.mlb.com/api/v1/schedule?sportId=1&date=YYYY-MM-DD
```

응답에 각 팀의 시즌 전적(`leagueRecord`)이 포함되어 있어 API 호출 한 번으로
경기 목록과 승률 계산에 필요한 정보를 모두 얻습니다.

## 테스트

```bash
python -m unittest test_mlb_win_rate.py -v
```
