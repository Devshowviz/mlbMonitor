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
  + 홈 어드밴티지 보정

각 컴포넌트가 내놓은 홈팀 승리 확률을 로그오즈(log-odds) 공간에서
가중 평균하고, 데이터가 없는 컴포넌트는 제외한 뒤 가중치를 재정규화한다.

사용 예:
    python mlb_win_rate.py                    # 오늘 날짜 경기
    python mlb_win_rate.py --date 2026-08-03
    python mlb_win_rate.py --detail           # 컴포넌트별 상세 출력
    python mlb_win_rate.py --json             # JSON으로 출력
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import urllib.error
import urllib.parse
import urllib.request
from datetime import date

API_BASE = "https://statsapi.mlb.com/api/v1"

# ---------------------------------------------------------------------------
# 모델 파라미터
# ---------------------------------------------------------------------------

# 컴포넌트별 가중치. 데이터가 없는 컴포넌트는 빼고 나머지로 재정규화한다.
WEIGHTS = {
    "season": 0.15,       # 시즌 승률 log5
    "pythagorean": 0.25,  # 득실점 기반 피타고리안 log5
    "split": 0.15,        # 홈/원정 스플릿 log5
    "form": 0.10,         # 최근 10경기 log5
    "pitcher": 0.35,      # 선발 투수 FIP 매치업
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


def fetch_standings(season: int) -> dict:
    """AL/NL 전체 팀의 순위표(득실점, 홈/원정, 최근 10경기 포함)를 가져온다."""
    params = urllib.parse.urlencode(
        {"leagueId": "103,104", "season": season, "standingsTypes": "regularSeason"}
    )
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
) -> list[dict]:
    """한 경기에 대해 사용 가능한 모든 컴포넌트의 홈팀 승리 확률을 계산한다."""
    away = game["teams"]["away"]
    home = game["teams"]["home"]
    components: list[dict] = []

    # 1. 시즌 승률 log5 — schedule 응답의 leagueRecord만으로 항상 계산 가능.
    home_season = padded_rate(
        home.get("leagueRecord", {}).get("wins", 0),
        home.get("leagueRecord", {}).get("losses", 0),
        PAD_SEASON,
    )
    away_season = padded_rate(
        away.get("leagueRecord", {}).get("wins", 0),
        away.get("leagueRecord", {}).get("losses", 0),
        PAD_SEASON,
    )
    components.append(
        {
            "name": "season",
            "prob": log5(home_season, away_season),
            "weight": WEIGHTS["season"],
            "detail": f"시즌 승률 {home_season:.3f} vs {away_season:.3f} (보정)",
        }
    )

    home_standing = standings_index.get(home["team"]["id"])
    away_standing = standings_index.get(away["team"]["id"])

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
                "weight": WEIGHTS["pythagorean"],
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
                    "weight": WEIGHTS["split"],
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
                    "weight": WEIGHTS["form"],
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
                "weight": WEIGHTS["pitcher"],
                "detail": (
                    f"{home_pitcher['name']} FIP {home_pitcher['fip']:.2f}"
                    f" vs {away_pitcher['name']} FIP {away_pitcher['fip']:.2f}"
                ),
            }
        )

    return components


def combine_components(components: list[dict]) -> float:
    """컴포넌트들을 로그오즈 가중 평균으로 결합하고 홈 어드밴티지를 더한다.

    데이터가 없어 빠진 컴포넌트는 남은 가중치로 재정규화된다.
    """
    total_weight = sum(component["weight"] for component in components)
    if total_weight == 0:
        return sigmoid(HOME_ADVANTAGE_LOGIT)
    weighted_logit = (
        sum(component["weight"] * logit(component["prob"]) for component in components)
        / total_weight
    )
    return sigmoid(weighted_logit + HOME_ADVANTAGE_LOGIT)


# ---------------------------------------------------------------------------
# 경기 파싱과 출력
# ---------------------------------------------------------------------------

def parse_games(
    schedule: dict,
    standings_index: dict[int, dict] | None = None,
    pitcher_index: dict[int, dict] | None = None,
) -> list[dict]:
    """schedule 응답에서 경기별 승률 정보를 계산한다."""
    standings_index = standings_index or {}
    pitcher_index = pitcher_index or {}
    games = []
    for day in schedule.get("dates", []):
        for game in day.get("games", []):
            away = game["teams"]["away"]
            home = game["teams"]["home"]

            components = compute_components(game, standings_index, pitcher_index)
            home_win_prob = combine_components(components)

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
}


def format_table(games: list[dict], game_date: str, detail: bool = False) -> str:
    """경기 목록을 읽기 좋은 표 형태의 문자열로 만든다."""
    if not games:
        return f"{game_date}에는 예정된 MLB 경기가 없습니다."

    lines = [f"=== {game_date} MLB 경기 승률 예측 ({len(games)}경기) ==="]
    header = (
        f"{'원정팀':<24} {'전적':>7} | {'홈팀':<24} {'전적':>7} | "
        f"{'홈승확률':>8} {'원정승확률':>8}  상태"
    )
    lines.append(header)
    lines.append("-" * 100)

    for game in games:
        score = ""
        if "away_score" in game:
            score = f" ({game['away_score']}:{game['home_score']})"
        lines.append(
            f"{game['away_team']:<24} {game['away_record']:>7} | "
            f"{game['home_team']:<24} {game['home_record']:>7} | "
            f"{game['home_win_prob']:>9.1%} {game['away_win_prob']:>10.1%}  "
            f"{game['status']}{score}"
        )
        if game["away_pitcher"] or game["home_pitcher"]:
            lines.append(
                f"    선발: {game['away_pitcher'] or '미정'}"
                f" vs {game['home_pitcher'] or '미정'}"
            )
        if detail:
            for component in game["components"]:
                label = COMPONENT_LABELS.get(component["name"], component["name"])
                lines.append(
                    f"    [{label:<7}] 홈승 {component['home_prob']:.3f}"
                    f" (가중치 {component['weight']:.2f}) — {component['detail']}"
                )
            lines.append("")
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
        "--detail", action="store_true", help="컴포넌트별 계산 근거도 출력"
    )
    parser.add_argument("--json", action="store_true", help="표 대신 JSON으로 출력")
    args = parser.parse_args(argv)

    season = int(args.date[:4])

    try:
        schedule = fetch_schedule(args.date)
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        print(f"MLB Stats API(schedule) 호출에 실패했습니다: {error}", file=sys.stderr)
        return 1

    # standings와 투수 스탯은 실패해도 schedule 기반 계산으로 계속 진행한다.
    standings_index: dict[int, dict] = {}
    try:
        standings_index = build_standings_index(fetch_standings(season))
    except (urllib.error.URLError, TimeoutError, OSError, KeyError) as error:
        print(
            f"경고: standings 조회 실패 — 피타고리안/스플릿/폼 컴포넌트를 건너뜁니다: {error}",
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

    games = parse_games(schedule, standings_index, pitcher_index)

    if args.json:
        print(
            json.dumps(
                {"date": args.date, "games": games}, ensure_ascii=False, indent=2
            )
        )
    else:
        print(format_table(games, args.date, detail=args.detail))
    return 0


if __name__ == "__main__":
    sys.exit(main())
