#!/usr/bin/env python3
"""그날의 MLB 경기 목록과 승률을 계산하는 프로그램.

MLB Stats API(https://statsapi.mlb.com)에서 얻을 수 있는 데이터를 총동원해
각 경기의 승리 확률을 앙상블 방식으로 추정한다.

사용하는 신호(컴포넌트):
  1. season      시즌 승률 log5 (표본 크기 보정 포함)
  2. pythagorean 득점/실점 기반 피타고리안 승률 log5 (Pythagenpat 지수)
  3. split       홈팀의 홈 성적 vs 원정팀의 원정 성적 log5
  4. form        최근 10경기 승률 log5
  5. pitcher     예고 선발 투수의 FIP 비교
  6. elo         시즌 경기 결과를 재생해 계산한 Elo 레이팅
  + 홈 어드밴티지 보정

각 컴포넌트가 내놓은 홈팀 승리 확률을 로그오즈(log-odds) 공간에서
가중 평균하고, 데이터가 없는 컴포넌트는 제외한 뒤 가중치를 재정규화한다.
backtest.py로 학습한 가중치(weights.json)가 있으면 그것을 사용한다.

사용 예:
    python mlb_win_rate.py                    # 오늘 날짜 경기
    python mlb_win_rate.py --date 2026-08-03
    python mlb_win_rate.py --detail           # 컴포넌트별 상세 출력
    python mlb_win_rate.py --json             # JSON으로 출력
    python mlb_win_rate.py --weights weights.json  # 학습된 가중치 사용
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from datetime import date, timedelta

API_BASE = "https://statsapi.mlb.com/api/v1"

# ---------------------------------------------------------------------------
# 모델 파라미터
# ---------------------------------------------------------------------------

# 컴포넌트별 기본 가중치. 데이터가 없는 컴포넌트는 빼고 나머지로 재정규화한다.
# backtest.py로 학습한 weights.json이 있으면 그 값으로 대체된다.
WEIGHTS = {
    "season": 0.10,       # 시즌 승률 log5
    "pythagorean": 0.20,  # 득실점 기반 피타고리안 log5
    "split": 0.10,        # 홈/원정 스플릿 log5
    "form": 0.10,         # 최근 10경기 log5
    "pitcher": 0.30,      # 선발 투수 FIP 매치업
    "elo": 0.20,          # Elo 레이팅
}

# 홈 어드밴티지: 로그오즈에 더하는 상수. 0.10 ≈ 승률 +2.5%p.
# (MLB 홈팀 승률은 역사적으로 약 53~54%. split 컴포넌트가 홈/원정
#  성적을 일부 반영하므로 과대계상을 피해 보수적으로 잡는다.)
HOME_ADVANTAGE_LOGIT = 0.10

# 표본 크기 보정(평균 회귀): 실제 전적에 5할 가상 경기를 섞는다.
# 시즌 초반 5승 0패 팀을 승률 1.000으로 취급하지 않기 위한 장치.
PAD_SEASON = 33.0   # 시즌 승률에 섞는 5할 경기 수
PAD_SPLIT = 20.0    # 홈/원정 스플릿용
PAD_FORM = 20.0     # 최근 10경기용 (10경기 표본은 노이즈가 커서 강하게 보정)

# 선발 투수 관련 파라미터
LEAGUE_AVG_FIP = 4.10     # 리그 평균 FIP 근사값
FIP_CONSTANT = 3.15       # FIP 계산 상수
PITCHER_SHRINK_IP = 40.0  # 이닝이 적은 투수는 리그 평균 쪽으로 보정
FIP_LOGIT_SCALE = 0.20    # FIP 1점 차이당 로그오즈 이동량 (≈ 승률 5%p)

# Pythagenpat 지수 파라미터: x = (경기당 총 득실점)^0.287
PYTHAGENPAT_EXPONENT = 0.287

# 로그오즈 변환 시 확률 클램프 (0/1 근처에서 발산 방지)
PROB_CLAMP = 0.01

# 배당 시장(핸디캡/언더오버) 예측용 파라미터
RUN_LINE_DEFAULT = 1.5        # 핸디캡 기준선 (MLB 표준 런라인)
TOTAL_LINE_DEFAULT = 8.5      # 언더오버 기준선
HOME_RUNS_FACTOR = 1.02       # 홈팀 득점 보정 (홈 어드밴티지의 득점 측면)
AWAY_RUNS_FACTOR = 0.98       # 원정팀 득점 보정
STARTER_INNINGS_SHARE = 0.55  # 선발 투수가 책임지는 이닝 비중 (FIP 보정에 사용)
MAX_RUNS_GRID = 30            # 푸아송 점수 격자 상한 (그 이상 득점 확률은 무시 가능)

# Elo 파라미터 (FiveThirtyEight MLB Elo에서 착안)
ELO_INITIAL = 1500.0    # 시즌 시작 레이팅
ELO_K = 4.0             # 경기당 레이팅 이동 폭
ELO_HOME_ADV = 24.0     # 레이팅 업데이트 시 홈팀에 더해주는 점수
ELO_MOV_EXPONENT = 0.7  # 점수차(margin of victory) 반영 지수

# 최종 결합 모델: 가중 평균 로그오즈에 scale을 곱하고 intercept(홈 어드밴티지)를
# 더한다. backtest.py가 학습해서 덮어쓸 수 있는 구조.
DEFAULT_MODEL = {
    "weights": WEIGHTS,
    "scale": 1.0,
    "intercept": HOME_ADVANTAGE_LOGIT,
}


def load_model(path: str) -> dict:
    """backtest.py가 저장한 weights.json을 읽어 결합 모델을 만든다."""
    with open(path, encoding="utf-8") as file:
        data = json.load(file)
    weights = data.get("weights", {})
    if not weights or any(weight < 0 for weight in weights.values()):
        raise ValueError(f"weights.json의 가중치가 올바르지 않습니다: {weights}")
    return {
        "weights": weights,
        "scale": float(data.get("scale", 1.0)),
        "intercept": float(data.get("intercept", HOME_ADVANTAGE_LOGIT)),
        "trained": data.get("trained"),
    }


def describe_model(model: dict) -> str:
    """어떤 모델(가중치)이 쓰이는지 한 줄로 요약한다."""
    weights = ", ".join(
        f"{name}={weight:.2f}" for name, weight in model["weights"].items()
    )
    line = f"모델: scale={model['scale']:.2f}, intercept={model['intercept']:.2f} | {weights}"
    trained = model.get("trained")
    if trained and "holdout_logloss" in trained:
        line += (
            f" | 학습: {trained.get('season')}시즌"
            f" 홀드아웃 로그손실 {trained['holdout_logloss']}"
        )
    return line


# ---------------------------------------------------------------------------
# API 호출
# ---------------------------------------------------------------------------

def _fetch_json(url: str, timeout: float = 15.0) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": "mlbMonitor/2.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def fetch_schedule(game_date: str) -> dict:
    """지정한 날짜(YYYY-MM-DD)의 경기 일정을 예고 선발 포함으로 가져온다."""
    params = urllib.parse.urlencode(
        {"sportId": 1, "date": game_date, "hydrate": "probablePitcher"}
    )
    return _fetch_json(f"{API_BASE}/schedule?{params}")


def fetch_standings(season: int, as_of_date: str | None = None) -> dict:
    """AL/NL 전체 팀의 순위표(득실점, 홈/원정, 최근 10경기 포함)를 가져온다.

    as_of_date(YYYY-MM-DD)를 주면 그 날짜 종료 시점의 순위표를 가져온다.
    과거 경기를 예측할 때 미래 정보가 새지 않도록 "경기 전날"을 넘긴다.
    """
    query = {"leagueId": "103,104", "season": season, "standingsTypes": "regularSeason"}
    if as_of_date:
        query["date"] = as_of_date
    params = urllib.parse.urlencode(query)
    return _fetch_json(f"{API_BASE}/standings?{params}")


def fetch_pitcher_stats(person_ids: list[int], season: int) -> dict:
    """예고 선발 투수들의 시즌 피칭 스탯을 한 번에 가져온다."""
    if not person_ids:
        return {"people": []}
    params = urllib.parse.urlencode(
        {
            "personIds": ",".join(str(pid) for pid in person_ids),
            "hydrate": f"stats(group=[pitching],type=[season],season={season})",
        }
    )
    return _fetch_json(f"{API_BASE}/people?{params}")


def fetch_season_results(season: int, end_date: str) -> list[dict]:
    """시즌 시작부터 end_date(포함)까지 정규시즌 최종 결과를 가져온다."""
    params = urllib.parse.urlencode(
        {
            "sportId": 1,
            "startDate": f"{season}-01-01",
            "endDate": end_date,
            "gameType": "R",
        }
    )
    schedule = _fetch_json(f"{API_BASE}/schedule?{params}")
    return extract_final_results(schedule)


def extract_final_results(schedule: dict) -> list[dict]:
    """schedule 응답에서 끝난 경기의 결과만 시간순으로 뽑아낸다."""
    results = []
    for day in schedule.get("dates", []):
        for game in day.get("games", []):
            if game.get("status", {}).get("abstractGameState") != "Final":
                continue
            away = game["teams"]["away"]
            home = game["teams"]["home"]
            if "score" not in away or "score" not in home:
                continue
            if away["score"] == home["score"]:  # 서스펜디드 등 무승부는 제외
                continue
            results.append(
                {
                    "date": day.get("date", ""),
                    "home_id": home["team"]["id"],
                    "away_id": away["team"]["id"],
                    "home_score": home["score"],
                    "away_score": away["score"],
                }
            )
    results.sort(key=lambda result: result["date"])
    return results


# ---------------------------------------------------------------------------
# 기본 수학 도구
# ---------------------------------------------------------------------------

def logit(p: float) -> float:
    """확률 → 로그오즈. 발산을 막기 위해 [0.01, 0.99]로 클램프한다."""
    p = min(max(p, PROB_CLAMP), 1.0 - PROB_CLAMP)
    return math.log(p / (1.0 - p))


def sigmoid(x: float) -> float:
    """로그오즈 → 확률."""
    return 1.0 / (1.0 + math.exp(-x))


def padded_rate(wins: float, losses: float, padding: float) -> float:
    """실제 전적에 5할짜리 가상 경기 `padding`판을 섞은 보정 승률."""
    games = wins + losses
    return (wins + 0.5 * padding) / (games + padding)


def log5(p_a: float, p_b: float) -> float:
    """log5 공식으로 A팀이 B팀을 이길 확률을 계산한다.

    P(A) = (pA - pA*pB) / (pA + pB - 2*pA*pB)
    분모가 0이 되는 극단적인 경우(둘 다 0.0 또는 1.0)는 0.5로 처리한다.
    """
    denominator = p_a + p_b - 2 * p_a * p_b
    if denominator == 0:
        return 0.5
    return (p_a - p_a * p_b) / denominator


def season_win_rate(league_record: dict) -> float:
    """leagueRecord(wins/losses)에서 시즌 승률을 계산한다. 0경기면 0.5."""
    wins = league_record.get("wins", 0)
    losses = league_record.get("losses", 0)
    if wins + losses == 0:
        return 0.5
    return wins / (wins + losses)


# ---------------------------------------------------------------------------
# 피타고리안 승률 (Pythagenpat)
# ---------------------------------------------------------------------------

def pythagenpat(runs_scored: float, runs_allowed: float, games: float) -> float:
    """득점/실점으로 기대 승률을 계산한다.

    고정 지수 2 대신 득점 환경에 맞춰 지수를 조정하는
    Pythagenpat 방식(x = (경기당 총 득실점)^0.287)을 쓴다.
    """
    if games <= 0 or (runs_scored <= 0 and runs_allowed <= 0):
        return 0.5
    if runs_allowed <= 0:
        return 1.0 - PROB_CLAMP
    if runs_scored <= 0:
        return PROB_CLAMP
    exponent = ((runs_scored + runs_allowed) / games) ** PYTHAGENPAT_EXPONENT
    rs_x = runs_scored ** exponent
    ra_x = runs_allowed ** exponent
    return rs_x / (rs_x + ra_x)


# ---------------------------------------------------------------------------
# 배당 시장 예측 (핸디캡 런라인, 언더오버)
# ---------------------------------------------------------------------------

def poisson_pmf(k: int, mu: float) -> float:
    """평균 mu인 푸아송 분포에서 정확히 k득점할 확률."""
    return math.exp(-mu) * mu**k / math.factorial(k)


def league_runs_per_game(standings_index: dict[int, dict]) -> float:
    """standings에서 리그 전체 팀당 경기당 평균 득점을 계산한다."""
    total_runs = sum(team["runs_scored"] for team in standings_index.values())
    total_games = sum(team["games"] for team in standings_index.values())
    if total_games == 0:
        return 4.5  # 시즌 개막 전이면 역사적 평균 근사값
    return total_runs / total_games


def matchup_expected_runs(
    home_standing: dict,
    away_standing: dict,
    league_rpg: float,
    home_fip: float | None = None,
    away_fip: float | None = None,
) -> tuple[float, float]:
    """두 팀의 이 경기 기대 득점 (홈, 원정)을 추정한다.

    기본 공식(Bill James 매치업): 기대 득점 = 공격력 × 상대 수비력 ÷ 리그 평균.
    여기에 상대 선발 FIP 보정(선발이 STARTER_INNINGS_SHARE 만큼의 이닝을
    책임진다고 보고 리그 평균 FIP 대비 비율로 조정)과 홈/원정 득점 보정을 곱한다.
    """

    def _rate(runs: float, games: float) -> float:
        return runs / games if games > 0 else league_rpg

    home_offense = _rate(home_standing["runs_scored"], home_standing["games"])
    home_defense = _rate(home_standing["runs_allowed"], home_standing["games"])
    away_offense = _rate(away_standing["runs_scored"], away_standing["games"])
    away_defense = _rate(away_standing["runs_allowed"], away_standing["games"])

    def _pitcher_factor(fip: float | None) -> float:
        if fip is None:
            return 1.0
        return STARTER_INNINGS_SHARE * (fip / LEAGUE_AVG_FIP) + (
            1.0 - STARTER_INNINGS_SHARE
        )

    # 홈팀 득점은 원정팀 수비(＋원정 선발), 원정팀 득점은 홈팀 수비(＋홈 선발)에 달려 있다.
    mu_home = (
        home_offense * away_defense / league_rpg
        * _pitcher_factor(away_fip)
        * HOME_RUNS_FACTOR
    )
    mu_away = (
        away_offense * home_defense / league_rpg
        * _pitcher_factor(home_fip)
        * AWAY_RUNS_FACTOR
    )
    return max(mu_home, 0.1), max(mu_away, 0.1)


def market_probs(
    mu_home: float,
    mu_away: float,
    run_line: float = RUN_LINE_DEFAULT,
    total_line: float = TOTAL_LINE_DEFAULT,
) -> dict:
    """기대 득점으로 핸디캡/언더오버/머니라인 확률을 계산한다.

    두 팀 득점을 독립 푸아송으로 두고 점수 조합 격자를 전부 더한다.
    야구는 무승부가 없으므로 동점 질량은 연장전(대부분 1점차 승부)으로
    해소된다고 보고 두 팀 승리 확률 비율로 나눠 배분한다 — 머니라인에만
    영향을 주고, 핸디캡(±1.5) 커버 여부는 동점 해소로 2점차 이상이 되지
    않으므로 격자 값을 그대로 쓴다.
    """
    home_pmf = [poisson_pmf(k, mu_home) for k in range(MAX_RUNS_GRID)]
    away_pmf = [poisson_pmf(k, mu_away) for k in range(MAX_RUNS_GRID)]

    p_home_win = p_away_win = p_tie = 0.0
    p_home_cover = p_away_cover = p_over = 0.0
    for h, ph in enumerate(home_pmf):
        for a, pa in enumerate(away_pmf):
            p = ph * pa
            margin = h - a
            if margin > 0:
                p_home_win += p
            elif margin < 0:
                p_away_win += p
            else:
                p_tie += p
            if margin > run_line:
                p_home_cover += p
            if -margin > run_line:
                p_away_cover += p
            if h + a > total_line:
                p_over += p

    decided = p_home_win + p_away_win
    home_share = p_home_win / decided if decided > 0 else 0.5
    home_ml = p_home_win + p_tie * home_share

    markets = {
        "expected_home_runs": round(mu_home, 2),
        "expected_away_runs": round(mu_away, 2),
        "expected_total": round(mu_home + mu_away, 2),
        "run_line": run_line,
        "total_line": total_line,
        "home_ml_prob": round(home_ml, 3),
        # 홈팀 -run_line 커버 확률과 원정팀 -run_line 커버 확률.
        # (+run_line 쪽 확률은 각각 1 - 반대편 커버 확률)
        "home_runline_prob": round(p_home_cover, 3),
        "away_runline_prob": round(p_away_cover, 3),
        "over_prob": round(p_over, 3),
        "under_prob": round(1.0 - p_over, 3),
    }

    # 핸디캡 픽: 우세팀(-run_line) vs 열세팀(+run_line) 중 확률 높은 쪽.
    favorite = "home" if home_ml >= 0.5 else "away"
    favorite_cover = p_home_cover if favorite == "home" else p_away_cover
    if favorite_cover >= 0.5:
        markets["runline_pick_team"] = favorite
        markets["runline_pick_line"] = -run_line
        markets["runline_pick_prob"] = round(favorite_cover, 3)
    else:
        markets["runline_pick_team"] = "away" if favorite == "home" else "home"
        markets["runline_pick_line"] = run_line
        markets["runline_pick_prob"] = round(1.0 - favorite_cover, 3)

    # 언더오버 픽
    if p_over >= 0.5:
        markets["ou_pick"] = "over"
        markets["ou_pick_prob"] = round(p_over, 3)
    else:
        markets["ou_pick"] = "under"
        markets["ou_pick_prob"] = round(1.0 - p_over, 3)

    return markets


def runline_pick_covered(markets: dict, home_margin: int) -> bool:
    """핸디캡 픽이 실제 점수차로 커버됐는지 판정한다.

    픽 팀 기준 점수차 + 핸디캡 라인이 0보다 크면 커버.
    (예: 홈 -1.5 픽 → 홈 점수차 2 이상, 원정 +1.5 픽 → 홈 점수차 1 이하)
    """
    team_margin = home_margin if markets["runline_pick_team"] == "home" else -home_margin
    return team_margin + markets["runline_pick_line"] > 0


def format_runline_pick(markets: dict) -> str:
    """핸디캡 픽을 '홈 -1.5' 같은 문자열로 만든다."""
    team = "홈" if markets["runline_pick_team"] == "home" else "원정"
    line = markets["runline_pick_line"]
    return f"{team} {line:+.1f}"


def compute_markets(
    game: dict,
    standings_index: dict[int, dict],
    pitcher_index: dict[int, dict],
    run_line: float = RUN_LINE_DEFAULT,
    total_line: float = TOTAL_LINE_DEFAULT,
) -> dict | None:
    """한 경기의 배당 시장 예측을 계산한다. standings가 없으면 None."""
    home = game["teams"]["home"]
    away = game["teams"]["away"]
    home_standing = standings_index.get(home["team"]["id"])
    away_standing = standings_index.get(away["team"]["id"])
    if not home_standing or not away_standing:
        return None
    if home_standing["games"] == 0 or away_standing["games"] == 0:
        return None

    home_pitcher_id = home.get("probablePitcher", {}).get("id")
    away_pitcher_id = away.get("probablePitcher", {}).get("id")
    home_fip = pitcher_index.get(home_pitcher_id, {}).get("fip") if home_pitcher_id else None
    away_fip = pitcher_index.get(away_pitcher_id, {}).get("fip") if away_pitcher_id else None

    mu_home, mu_away = matchup_expected_runs(
        home_standing,
        away_standing,
        league_runs_per_game(standings_index),
        home_fip,
        away_fip,
    )
    return market_probs(mu_home, mu_away, run_line, total_line)


# ---------------------------------------------------------------------------
# Elo 레이팅
# ---------------------------------------------------------------------------

def elo_win_prob(home_elo: float, away_elo: float) -> float:
    """구장 중립 기준으로 홈팀이 이길 확률. (홈 어드밴티지는 결합 단계에서 반영)"""
    return 1.0 / (1.0 + 10.0 ** (-(home_elo - away_elo) / 400.0))


def update_elo(ratings: dict[int, float], result: dict) -> None:
    """한 경기 결과로 두 팀의 Elo를 갱신한다.

    - 업데이트 기대값 계산에는 홈 어드밴티지(ELO_HOME_ADV)를 반영한다.
    - 점수차가 클수록, 그리고 약팀이 이겼을수록 레이팅이 크게 움직인다
      (FiveThirtyEight의 margin-of-victory 배수 방식).
    """
    home = ratings.setdefault(result["home_id"], ELO_INITIAL)
    away = ratings.setdefault(result["away_id"], ELO_INITIAL)

    expected_home = 1.0 / (
        1.0 + 10.0 ** (-((home + ELO_HOME_ADV) - away) / 400.0)
    )
    home_won = result["home_score"] > result["away_score"]
    actual_home = 1.0 if home_won else 0.0

    margin = abs(result["home_score"] - result["away_score"])
    winner_elo_diff = (
        (home + ELO_HOME_ADV) - away if home_won else away - (home + ELO_HOME_ADV)
    )
    mov_multiplier = ((margin + 1) ** ELO_MOV_EXPONENT) / (
        7.5 + 0.006 * winner_elo_diff
    )

    delta = ELO_K * mov_multiplier * (actual_home - expected_home)
    ratings[result["home_id"]] = home + delta
    ratings[result["away_id"]] = away - delta


def compute_elo_ratings(results: list[dict]) -> dict[int, float]:
    """시즌 경기 결과를 시간순으로 재생해 팀별 Elo 레이팅을 계산한다."""
    ratings: dict[int, float] = {}
    for result in sorted(results, key=lambda r: r["date"]):
        update_elo(ratings, result)
    return ratings


# ---------------------------------------------------------------------------
# 선발 투수 (FIP)
# ---------------------------------------------------------------------------

def parse_innings(ip_str: str) -> float:
    """'123.1' 같은 이닝 표기를 실수로 바꾼다. (.1 = 1/3이닝, .2 = 2/3이닝)"""
    try:
        text = str(ip_str)
        if "." in text:
            whole, frac = text.split(".", 1)
            return int(whole) + int(frac) / 3.0
        return float(text)
    except (ValueError, TypeError):
        return 0.0


def fip_from_stat(stat: dict) -> tuple[float, float]:
    """시즌 피칭 스탯에서 (보정된 FIP, 이닝)을 계산한다.

    FIP = (13*HR + 3*BB - 2*K) / IP + 상수
    이닝이 적은 투수는 리그 평균 FIP 쪽으로 회귀시킨다.
    """
    innings = parse_innings(stat.get("inningsPitched", "0"))
    if innings <= 0:
        return LEAGUE_AVG_FIP, 0.0
    home_runs = stat.get("homeRuns", 0)
    walks = stat.get("baseOnBalls", 0) + stat.get("hitByPitch", 0)
    strikeouts = stat.get("strikeOuts", 0)
    raw_fip = (13 * home_runs + 3 * walks - 2 * strikeouts) / innings + FIP_CONSTANT
    # 이닝 기반 회귀: 표본이 작을수록 리그 평균에 가깝게.
    shrunk = (innings * raw_fip + PITCHER_SHRINK_IP * LEAGUE_AVG_FIP) / (
        innings + PITCHER_SHRINK_IP
    )
    return shrunk, innings


def pitcher_matchup_prob(home_fip: float, away_fip: float) -> float:
    """두 선발의 FIP 차이를 홈팀 승리 확률로 변환한다.

    FIP가 낮을수록 좋은 투수이므로 (원정 FIP - 홈 FIP)가 클수록 홈팀에 유리.
    """
    return sigmoid(FIP_LOGIT_SCALE * (away_fip - home_fip))


# ---------------------------------------------------------------------------
# API 응답 인덱싱
# ---------------------------------------------------------------------------

def build_standings_index(standings: dict) -> dict[int, dict]:
    """standings 응답을 팀 ID로 찾아 쓸 수 있게 인덱싱한다."""
    index: dict[int, dict] = {}
    for record_group in standings.get("records", []):
        for team_record in record_group.get("teamRecords", []):
            team_id = team_record.get("team", {}).get("id")
            if team_id is None:
                continue
            splits = {
                split.get("type"): split
                for split in team_record.get("records", {}).get("splitRecords", [])
            }
            index[team_id] = {
                "wins": team_record.get("wins", 0),
                "losses": team_record.get("losses", 0),
                "runs_scored": team_record.get("runsScored", 0),
                "runs_allowed": team_record.get("runsAllowed", 0),
                "games": team_record.get("gamesPlayed", 0),
                "home": splits.get("home"),
                "away": splits.get("away"),
                "last_ten": splits.get("lastTen"),
            }
    return index


def build_pitcher_index(people: dict) -> dict[int, dict]:
    """people 응답에서 투수 ID → {fip, innings} 인덱스를 만든다."""
    index: dict[int, dict] = {}
    for person in people.get("people", []):
        stat = _season_pitching_stat(person)
        if stat is None:
            continue
        fip, innings = fip_from_stat(stat)
        index[person["id"]] = {
            "name": person.get("fullName", ""),
            "fip": fip,
            "innings": innings,
        }
    return index


def _season_pitching_stat(person: dict) -> dict | None:
    for stat_group in person.get("stats", []):
        if stat_group.get("group", {}).get("displayName") != "pitching":
            continue
        for split in stat_group.get("splits", []):
            return split.get("stat", {})
    return None


# ---------------------------------------------------------------------------
# 컴포넌트 계산과 결합
# ---------------------------------------------------------------------------

def compute_components(
    game: dict,
    standings_index: dict[int, dict],
    pitcher_index: dict[int, dict],
    elo_ratings: dict[int, float] | None = None,
    weights: dict[str, float] | None = None,
) -> list[dict]:
    """한 경기에 대해 사용 가능한 모든 컴포넌트의 홈팀 승리 확률을 계산한다."""
    away = game["teams"]["away"]
    home = game["teams"]["home"]
    elo_ratings = elo_ratings or {}
    weights = weights or WEIGHTS
    components: list[dict] = []

    home_standing = standings_index.get(home["team"]["id"])
    away_standing = standings_index.get(away["team"]["id"])

    # 1. 시즌 승률 log5 — standings(경기 전날 기준)가 있으면 그 전적을,
    #    없으면 schedule의 leagueRecord를 쓴다. 과거 경기의 leagueRecord는
    #    경기 후 전적이라 standings 쪽이 누수 없이 더 정확하다.
    if home_standing and away_standing:
        home_record = {"wins": home_standing["wins"], "losses": home_standing["losses"]}
        away_record = {"wins": away_standing["wins"], "losses": away_standing["losses"]}
    else:
        home_record = home.get("leagueRecord", {})
        away_record = away.get("leagueRecord", {})
    home_season = padded_rate(
        home_record.get("wins", 0), home_record.get("losses", 0), PAD_SEASON
    )
    away_season = padded_rate(
        away_record.get("wins", 0), away_record.get("losses", 0), PAD_SEASON
    )
    components.append(
        {
            "name": "season",
            "prob": log5(home_season, away_season),
            "weight": weights.get("season", 0.0),
            "detail": f"시즌 승률 {home_season:.3f} vs {away_season:.3f} (보정)",
        }
    )

    if home_standing and away_standing:
        # 2. 피타고리안 승률 log5
        home_pyth = pythagenpat(
            home_standing["runs_scored"],
            home_standing["runs_allowed"],
            home_standing["games"],
        )
        away_pyth = pythagenpat(
            away_standing["runs_scored"],
            away_standing["runs_allowed"],
            away_standing["games"],
        )
        components.append(
            {
                "name": "pythagorean",
                "prob": log5(home_pyth, away_pyth),
                "weight": weights.get("pythagorean", 0.0),
                "detail": f"피타고리안 {home_pyth:.3f} vs {away_pyth:.3f}",
            }
        )

        # 3. 홈/원정 스플릿 log5 — 홈팀의 홈 성적 vs 원정팀의 원정 성적
        if home_standing["home"] and away_standing["away"]:
            home_split = padded_rate(
                home_standing["home"].get("wins", 0),
                home_standing["home"].get("losses", 0),
                PAD_SPLIT,
            )
            away_split = padded_rate(
                away_standing["away"].get("wins", 0),
                away_standing["away"].get("losses", 0),
                PAD_SPLIT,
            )
            components.append(
                {
                    "name": "split",
                    "prob": log5(home_split, away_split),
                    "weight": weights.get("split", 0.0),
                    "detail": f"홈 성적 {home_split:.3f} vs 원정 성적 {away_split:.3f} (보정)",
                }
            )

        # 4. 최근 10경기 폼 log5
        if home_standing["last_ten"] and away_standing["last_ten"]:
            home_form = padded_rate(
                home_standing["last_ten"].get("wins", 0),
                home_standing["last_ten"].get("losses", 0),
                PAD_FORM,
            )
            away_form = padded_rate(
                away_standing["last_ten"].get("wins", 0),
                away_standing["last_ten"].get("losses", 0),
                PAD_FORM,
            )
            components.append(
                {
                    "name": "form",
                    "prob": log5(home_form, away_form),
                    "weight": weights.get("form", 0.0),
                    "detail": f"최근 10경기 {home_form:.3f} vs {away_form:.3f} (보정)",
                }
            )

    # 5. 선발 투수 FIP 매치업
    home_pitcher_id = home.get("probablePitcher", {}).get("id")
    away_pitcher_id = away.get("probablePitcher", {}).get("id")
    home_pitcher = pitcher_index.get(home_pitcher_id) if home_pitcher_id else None
    away_pitcher = pitcher_index.get(away_pitcher_id) if away_pitcher_id else None
    if home_pitcher and away_pitcher:
        components.append(
            {
                "name": "pitcher",
                "prob": pitcher_matchup_prob(home_pitcher["fip"], away_pitcher["fip"]),
                "weight": weights.get("pitcher", 0.0),
                "detail": (
                    f"{home_pitcher['name']} FIP {home_pitcher['fip']:.2f}"
                    f" vs {away_pitcher['name']} FIP {away_pitcher['fip']:.2f}"
                ),
            }
        )

    # 6. Elo 레이팅
    home_elo = elo_ratings.get(home["team"]["id"])
    away_elo = elo_ratings.get(away["team"]["id"])
    if home_elo is not None and away_elo is not None:
        components.append(
            {
                "name": "elo",
                "prob": elo_win_prob(home_elo, away_elo),
                "weight": weights.get("elo", 0.0),
                "detail": f"Elo {home_elo:.0f} vs {away_elo:.0f}",
            }
        )

    return components


def combine_components(components: list[dict], model: dict | None = None) -> float:
    """컴포넌트들을 로그오즈 가중 평균으로 결합해 최종 확률을 만든다.

    최종 로그오즈 = scale × (가중 평균 로그오즈) + intercept(홈 어드밴티지).
    데이터가 없어 빠진 컴포넌트는 남은 가중치로 재정규화된다.
    """
    model = model or DEFAULT_MODEL
    total_weight = sum(component["weight"] for component in components)
    if total_weight == 0:
        return sigmoid(model["intercept"])
    weighted_logit = (
        sum(component["weight"] * logit(component["prob"]) for component in components)
        / total_weight
    )
    return sigmoid(model["scale"] * weighted_logit + model["intercept"])


# ---------------------------------------------------------------------------
# 경기 파싱과 출력
# ---------------------------------------------------------------------------

def parse_games(
    schedule: dict,
    standings_index: dict[int, dict] | None = None,
    pitcher_index: dict[int, dict] | None = None,
    elo_ratings: dict[int, float] | None = None,
    model: dict | None = None,
    run_line: float = RUN_LINE_DEFAULT,
    total_line: float = TOTAL_LINE_DEFAULT,
) -> list[dict]:
    """schedule 응답에서 경기별 승률 정보를 계산한다."""
    standings_index = standings_index or {}
    pitcher_index = pitcher_index or {}
    model = model or DEFAULT_MODEL
    games = []
    for day in schedule.get("dates", []):
        for game in day.get("games", []):
            # 올스타전(A)·시범경기(E)는 전력 예측 대상이 아니다.
            if game.get("gameType") in ("A", "E"):
                continue
            away = game["teams"]["away"]
            home = game["teams"]["home"]

            components = compute_components(
                game, standings_index, pitcher_index, elo_ratings, model["weights"]
            )
            home_win_prob = combine_components(components, model)
            markets = compute_markets(
                game, standings_index, pitcher_index, run_line, total_line
            )

            row = {
                "away_team": away["team"]["name"],
                "home_team": home["team"]["name"],
                "away_record": _format_record(away.get("leagueRecord", {})),
                "home_record": _format_record(home.get("leagueRecord", {})),
                "away_win_rate": round(season_win_rate(away.get("leagueRecord", {})), 3),
                "home_win_rate": round(season_win_rate(home.get("leagueRecord", {})), 3),
                "home_win_prob": round(home_win_prob, 3),
                "away_win_prob": round(1 - home_win_prob, 3),
                "components": [
                    {
                        "name": component["name"],
                        "home_prob": round(component["prob"], 3),
                        "weight": component["weight"],
                        "detail": component["detail"],
                    }
                    for component in components
                ],
                "status": game.get("status", {}).get("detailedState", ""),
                "venue": game.get("venue", {}).get("name", ""),
                "away_pitcher": away.get("probablePitcher", {}).get("fullName", ""),
                "home_pitcher": home.get("probablePitcher", {}).get("fullName", ""),
            }
            if markets:
                row["markets"] = markets

            # 이미 끝났거나 진행 중인 경기는 점수도 함께 담는다.
            if "score" in away and "score" in home:
                row["away_score"] = away["score"]
                row["home_score"] = home["score"]

            games.append(row)
    return games


def collect_probable_pitcher_ids(schedule: dict) -> list[int]:
    """schedule 응답에서 예고 선발 투수 ID를 전부 모은다."""
    ids: list[int] = []
    for day in schedule.get("dates", []):
        for game in day.get("games", []):
            for side in ("away", "home"):
                pitcher_id = (
                    game["teams"][side].get("probablePitcher", {}).get("id")
                )
                if pitcher_id:
                    ids.append(pitcher_id)
    return sorted(set(ids))


def _format_record(league_record: dict) -> str:
    return f"{league_record.get('wins', 0)}-{league_record.get('losses', 0)}"


COMPONENT_LABELS = {
    "season": "시즌 승률",
    "pythagorean": "피타고리안",
    "split": "홈/원정",
    "form": "최근 10경기",
    "pitcher": "선발 투수",
    "elo": "Elo",
}

STATUS_LABELS = {
    "Final": "종료",
    "Completed Early": "종료",
    "Game Over": "종료",
    "Scheduled": "예정",
    "Pre-Game": "예정",
    "Warmup": "예정",
    "In Progress": "진행중",
    "Postponed": "연기",
    "Suspended": "중단",
    "Cancelled": "취소",
    "Delayed": "지연",
}


def display_width(text: str) -> int:
    """터미널 표시 폭. 한글 등 전각 문자는 2칸으로 계산한다."""
    return sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in text)


def pad(text: str, width: int, align: str = "left") -> str:
    """표시 폭 기준으로 공백을 채워 정렬한다 (한글 섞인 표 정렬용)."""
    gap = max(width - display_width(text), 0)
    if align == "right":
        return " " * gap + text
    return text + " " * gap


def _render_table(headers: list[str], rows: list[list[str]], aligns: list[str]) -> list[str]:
    """헤더와 행들을 표시 폭 기준으로 정렬한 표 문자열 목록으로 만든다."""
    widths = [
        max(display_width(headers[i]), *(display_width(row[i]) for row in rows))
        if rows else display_width(headers[i])
        for i in range(len(headers))
    ]
    lines = [
        "  ".join(pad(header, width) for header, width in zip(headers, widths)),
        "  ".join("─" * width for width in widths),
    ]
    for row in rows:
        lines.append(
            "  ".join(
                pad(cell, width, align)
                for cell, width, align in zip(row, widths, aligns)
            )
        )
    return lines


def _hit_mark(hit: bool) -> str:
    return "✓" if hit else "✗"


def format_table(games: list[dict], game_date: str, detail: bool = False) -> str:
    """경기 목록을 읽기 좋은 표 형태의 문자열로 만든다."""
    if not games:
        return f"{game_date}에는 예정된 MLB 경기가 없습니다."

    headers = ["원정팀 (전적)", "홈팀 (전적)", "홈승", "핸디캡 예상", "언더오버 예상", "상태"]
    aligns = ["left", "left", "right", "left", "left", "left"]
    rows = []
    for game in games:
        finished = (
            "away_score" in game and game["away_score"] != game["home_score"]
        )
        margin = total = None
        if finished:
            margin = game["home_score"] - game["away_score"]
            total = game["home_score"] + game["away_score"]

        win_cell = f"{game['home_win_prob']:.1%}"
        if finished:
            win_cell += _hit_mark((game["home_win_prob"] >= 0.5) == (margin > 0))

        markets = game.get("markets")
        if markets:
            runline_cell = (
                f"{format_runline_pick(markets)} ({markets['runline_pick_prob']:.0%})"
            )
            if finished:
                runline_cell += _hit_mark(runline_pick_covered(markets, margin))
            ou_label = "오버" if markets["ou_pick"] == "over" else "언더"
            ou_cell = f"{ou_label} {markets['total_line']} ({markets['ou_pick_prob']:.0%})"
            if finished and total != markets["total_line"]:
                ou_cell += _hit_mark(
                    (total > markets["total_line"]) == (markets["ou_pick"] == "over")
                )
        else:
            runline_cell = ou_cell = "-"

        status = STATUS_LABELS.get(game["status"], game["status"])
        if "away_score" in game:
            status += f" {game['away_score']}:{game['home_score']}"

        rows.append(
            [
                f"{game['away_team']} ({game['away_record']})",
                f"{game['home_team']} ({game['home_record']})",
                win_cell,
                runline_cell,
                ou_cell,
                status,
            ]
        )

    lines = [f"=== {game_date} MLB 경기 승률 예측 ({len(games)}경기) ==="]
    lines.extend(_render_table(headers, rows, aligns))

    if detail:
        lines.append("")
        for game in games:
            lines.append(f"[{game['away_team']} @ {game['home_team']}]")
            if game["away_pitcher"] or game["home_pitcher"]:
                lines.append(
                    f"  선발: {game['away_pitcher'] or '미정'}"
                    f" vs {game['home_pitcher'] or '미정'}"
                )
            markets = game.get("markets")
            if markets:
                lines.append(
                    f"  마켓: 예상 득점 {markets['expected_away_runs']:.1f}:"
                    f"{markets['expected_home_runs']:.1f}"
                    f" (합 {markets['expected_total']:.1f})"
                    f" | 오버 {markets['total_line']}: {markets['over_prob']:.1%}"
                    f" | 홈 -{markets['run_line']}: {markets['home_runline_prob']:.1%}"
                    f" / 원정 -{markets['run_line']}: {markets['away_runline_prob']:.1%}"
                )
            for component in game["components"]:
                label = COMPONENT_LABELS.get(component["name"], component["name"])
                lines.append(
                    f"  [{pad(label, 12)}] 홈승 {component['home_prob']:.3f}"
                    f" (가중치 {component['weight']:.2f}) — {component['detail']}"
                )
            lines.append("")
    return "\n".join(lines)


def evaluation_stats(games: list[dict]) -> dict | None:
    """끝난 경기들에 대해 예측 성능(적중률/로그손실/브라이어)을 계산한다."""
    finished = [
        game
        for game in games
        if "home_score" in game and game["home_score"] != game["away_score"]
    ]
    if not finished:
        return None

    hits = 0
    home_wins = 0
    pred_home_sum = 0.0
    total_log_loss = 0.0
    total_brier = 0.0
    for game in finished:
        actual = 1 if game["home_score"] > game["away_score"] else 0
        home_wins += actual
        prob = min(max(game["home_win_prob"], 1e-12), 1 - 1e-12)
        pred_home_sum += game["home_win_prob"]
        if (prob >= 0.5) == (actual == 1):
            hits += 1
        total_log_loss += -(
            actual * math.log(prob) + (1 - actual) * math.log(1 - prob)
        )
        total_brier += (prob - actual) ** 2

    count = len(finished)
    stats = {
        "games": count,
        "hits": hits,
        "accuracy": round(hits / count, 3),
        "log_loss": round(total_log_loss / count, 4),
        "brier": round(total_brier / count, 4),
        # 캘리브레이션 진단: 평균 예측 확률과 실제 비율이 다르면 체계적 편향.
        "avg_pred_home": round(pred_home_sum / count, 3),
        "actual_home_rate": round(home_wins / count, 3),
    }

    # 마켓 예측(핸디캡/언더오버)도 실제 점수로 검증한다 (브라이어 점수).
    runline_brier = runline_hits = runline_count = 0
    runline_pred_sum = runline_covers = 0.0
    over_brier = over_hits = over_count = 0
    over_pred_sum = over_actual = 0.0
    expected_total_sum = actual_total_sum = 0.0
    for game in finished:
        markets = game.get("markets")
        if not markets:
            continue
        margin = game["home_score"] - game["away_score"]
        total = game["home_score"] + game["away_score"]

        actual_cover = 1 if margin > markets["run_line"] else 0
        runline_covers += actual_cover
        runline_pred_sum += markets["home_runline_prob"]
        runline_brier += (markets["home_runline_prob"] - actual_cover) ** 2
        # 적중률은 표에 표시되는 실제 픽(우세팀 라인 기준) 기준으로 계산한다.
        if runline_pick_covered(markets, margin):
            runline_hits += 1
        runline_count += 1

        expected_total_sum += markets["expected_total"]
        actual_total_sum += total
        if total != markets["total_line"]:  # 정수 기준선의 푸시(동률)는 제외
            actual_over = 1 if total > markets["total_line"] else 0
            over_actual += actual_over
            over_pred_sum += markets["over_prob"]
            over_brier += (markets["over_prob"] - actual_over) ** 2
            if (markets["over_prob"] >= 0.5) == (actual_over == 1):
                over_hits += 1
            over_count += 1

    if runline_count:
        stats["runline"] = {
            "games": runline_count,
            "hits": runline_hits,
            "accuracy": round(runline_hits / runline_count, 3),
            "brier": round(runline_brier / runline_count, 4),
            "avg_pred_cover": round(runline_pred_sum / runline_count, 3),
            "actual_cover_rate": round(runline_covers / runline_count, 3),
        }
    if over_count:
        stats["over_under"] = {
            "games": over_count,
            "hits": over_hits,
            "accuracy": round(over_hits / over_count, 3),
            "brier": round(over_brier / over_count, 4),
            "avg_pred_over": round(over_pred_sum / over_count, 3),
            "actual_over_rate": round(over_actual / over_count, 3),
            "avg_expected_total": round(expected_total_sum / runline_count, 2),
            "avg_actual_total": round(actual_total_sum / runline_count, 2),
        }
    return stats


def format_evaluation(stats: dict) -> str:
    headers = ["구분", "적중", "적중률", "브라이어", "캘리브레이션 (예측 vs 실제)"]
    aligns = ["left", "right", "right", "right", "left"]
    rows = [
        [
            "승패",
            f"{stats['hits']}/{stats['games']}",
            f"{stats['accuracy']:.1%}",
            f"{stats['brier']:.4f}",
            f"홈승 {stats['avg_pred_home']:.1%} vs {stats['actual_home_rate']:.1%}",
        ]
    ]
    if "runline" in stats:
        runline = stats["runline"]
        rows.append(
            [
                "핸디캡",
                f"{runline['hits']}/{runline['games']}",
                f"{runline['accuracy']:.1%}",
                f"{runline['brier']:.4f}",
                f"홈커버 {runline['avg_pred_cover']:.1%}"
                f" vs {runline['actual_cover_rate']:.1%}",
            ]
        )
    if "over_under" in stats:
        over_under = stats["over_under"]
        rows.append(
            [
                "언더오버",
                f"{over_under['hits']}/{over_under['games']}",
                f"{over_under['accuracy']:.1%}",
                f"{over_under['brier']:.4f}",
                f"오버 {over_under['avg_pred_over']:.1%}"
                f" vs {over_under['actual_over_rate']:.1%}"
                f" · 합계 {over_under['avg_expected_total']:.2f}"
                f" vs {over_under['avg_actual_total']:.2f}",
            ]
        )
    lines = [f"=== 예측 검증 (끝난 경기 {stats['games']}건) ==="]
    lines.extend(_render_table(headers, rows, aligns))
    lines.append(
        f"승패 로그손실 {stats['log_loss']:.4f}"
        " (동전던지기 0.6931, 전적 기반 모델의 현실적 수준 ≈ 0.68)"
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 메인
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="그날의 MLB 경기 승률 예측")
    parser.add_argument(
        "--date",
        default=date.today().isoformat(),
        help="조회할 날짜 (YYYY-MM-DD, 기본값: 오늘)",
    )
    parser.add_argument(
        "--end-date",
        default=None,
        help="여러 날짜를 한 번에 예측할 때의 마지막 날짜 (YYYY-MM-DD)."
        " 끝난 경기가 있으면 예측 검증 요약도 출력한다.",
    )
    parser.add_argument(
        "--detail", action="store_true", help="컴포넌트별 계산 근거도 출력"
    )
    parser.add_argument(
        "--run-line",
        type=float,
        default=RUN_LINE_DEFAULT,
        help=f"핸디캡 기준선 (기본값: {RUN_LINE_DEFAULT})",
    )
    parser.add_argument(
        "--total-line",
        type=float,
        default=TOTAL_LINE_DEFAULT,
        help=f"언더오버 기준선 (기본값: {TOTAL_LINE_DEFAULT})",
    )
    parser.add_argument("--json", action="store_true", help="표 대신 JSON으로 출력")
    parser.add_argument(
        "--weights",
        default=None,
        help="backtest.py로 학습한 가중치 파일 경로"
        " (지정하지 않아도 ./weights.json이 있으면 자동 사용)",
    )
    args = parser.parse_args(argv)

    try:
        start = date.fromisoformat(args.date)
        end = date.fromisoformat(args.end_date) if args.end_date else start
    except ValueError as error:
        print(f"날짜 형식이 잘못됐습니다: {error}", file=sys.stderr)
        return 1
    if end < start:
        print("--end-date는 --date보다 빠를 수 없습니다.", file=sys.stderr)
        return 1

    season = start.year

    # 결합 모델: 학습된 가중치 파일이 있으면 사용, 없으면 기본값.
    model = DEFAULT_MODEL
    weights_path = args.weights or ("weights.json" if os.path.exists("weights.json") else None)
    if weights_path:
        try:
            model = load_model(weights_path)
            print(f"학습된 가중치 사용: {weights_path}", file=sys.stderr)
            if not model.get("trained"):
                print(
                    "경고: 이 가중치 파일에는 학습 메타데이터가 없습니다 —"
                    " 구버전 backtest로 만든 파일일 수 있으니"
                    " `python backtest.py --season 2025 --output weights.json`으로"
                    " 재학습하거나 파일을 지우고 기본 가중치를 쓰세요.",
                    file=sys.stderr,
                )
        except (OSError, ValueError, json.JSONDecodeError) as error:
            print(
                f"경고: 가중치 파일을 읽지 못해 기본 가중치를 사용합니다: {error}",
                file=sys.stderr,
            )
    print(describe_model(model), file=sys.stderr)

    # Elo용 시즌 결과는 범위 전체에 대해 한 번만 가져온다 (마지막 날 전날까지).
    season_results: list[dict] = []
    try:
        day_before_end = (end - timedelta(days=1)).isoformat()
        season_results = fetch_season_results(season, day_before_end)
    except (urllib.error.URLError, TimeoutError, OSError, KeyError) as error:
        print(
            f"경고: 시즌 결과 조회 실패 — Elo 컴포넌트를 건너뜁니다: {error}",
            file=sys.stderr,
        )

    all_games: list[dict] = []
    outputs: list[str] = []
    current = start
    while current <= end:
        game_date = current.isoformat()
        try:
            schedule = fetch_schedule(game_date)
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            print(
                f"MLB Stats API(schedule, {game_date}) 호출에 실패했습니다: {error}",
                file=sys.stderr,
            )
            return 1

        # standings는 미래 정보가 새지 않도록 "경기 전날" 기준으로 가져온다.
        standings_index: dict[int, dict] = {}
        day_before = (current - timedelta(days=1)).isoformat()
        try:
            standings_index = build_standings_index(
                fetch_standings(season, as_of_date=day_before)
            )
        except (urllib.error.URLError, TimeoutError, OSError, KeyError) as error:
            print(
                f"경고: standings({day_before}) 조회 실패 — "
                f"피타고리안/스플릿/폼 컴포넌트를 건너뜁니다: {error}",
                file=sys.stderr,
            )

        pitcher_index: dict[int, dict] = {}
        pitcher_ids = collect_probable_pitcher_ids(schedule)
        if pitcher_ids:
            try:
                pitcher_index = build_pitcher_index(
                    fetch_pitcher_stats(pitcher_ids, season)
                )
            except (urllib.error.URLError, TimeoutError, OSError, KeyError) as error:
                print(
                    f"경고: 투수 스탯 조회 실패 — 선발 투수 컴포넌트를 건너뜁니다: {error}",
                    file=sys.stderr,
                )

        # Elo: 그 날짜 전까지의 결과만 재생한다.
        elo_ratings = compute_elo_ratings(
            [result for result in season_results if result["date"] < game_date]
        )

        games = parse_games(
            schedule,
            standings_index,
            pitcher_index,
            elo_ratings,
            model,
            run_line=args.run_line,
            total_line=args.total_line,
        )
        all_games.extend(games)
        outputs.append(format_table(games, game_date, detail=args.detail))
        current += timedelta(days=1)

    stats = evaluation_stats(all_games)

    if args.json:
        payload = {"date": args.date, "games": all_games}
        if args.end_date:
            payload["end_date"] = args.end_date
        if stats:
            payload["evaluation"] = stats
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print("\n\n".join(outputs))
        if stats:
            print()
            print(format_evaluation(stats))
    return 0


if __name__ == "__main__":
    sys.exit(main())
