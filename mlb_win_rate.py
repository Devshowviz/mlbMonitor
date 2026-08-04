#!/usr/bin/env python3
"""그날의 MLB 경기 목록과 승률을 계산하는 프로그램.

MLB Stats API(https://statsapi.mlb.com)의 schedule 엔드포인트를 호출해
지정한 날짜의 경기 목록을 가져오고, 각 팀의 시즌 승률과
두 팀 간 예상 승리 확률(log5 공식)을 계산해 표로 출력한다.

사용 예:
    python mlb_win_rate.py                # 오늘 날짜 경기
    python mlb_win_rate.py --date 2026-08-03
    python mlb_win_rate.py --json         # JSON으로 출력
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.parse
import urllib.request
from datetime import date

API_BASE = "https://statsapi.mlb.com/api/v1/schedule"


def fetch_schedule(game_date: str, timeout: float = 15.0) -> dict:
    """지정한 날짜(YYYY-MM-DD)의 MLB 경기 일정을 가져온다."""
    params = urllib.parse.urlencode({"sportId": 1, "date": game_date})
    url = f"{API_BASE}?{params}"
    request = urllib.request.Request(url, headers={"User-Agent": "mlbMonitor/1.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def season_win_rate(league_record: dict) -> float:
    """leagueRecord(wins/losses)에서 시즌 승률을 계산한다.

    아직 경기를 치르지 않았으면(0승 0패) 0.5로 간주한다.
    """
    wins = league_record.get("wins", 0)
    losses = league_record.get("losses", 0)
    games = wins + losses
    if games == 0:
        return 0.5
    return wins / games


def log5(p_a: float, p_b: float) -> float:
    """log5 공식으로 A팀이 B팀을 이길 확률을 계산한다.

    P(A) = (pA - pA*pB) / (pA + pB - 2*pA*pB)

    두 팀의 승률이 같으면 0.5가 나오고, 분모가 0이 되는
    극단적인 경우(둘 다 0.0 또는 둘 다 1.0)도 0.5로 처리한다.
    """
    denominator = p_a + p_b - 2 * p_a * p_b
    if denominator == 0:
        return 0.5
    return (p_a - p_a * p_b) / denominator


def parse_games(schedule: dict) -> list[dict]:
    """schedule API 응답에서 경기별 승률 정보를 뽑아낸다."""
    games = []
    for day in schedule.get("dates", []):
        for game in day.get("games", []):
            away = game["teams"]["away"]
            home = game["teams"]["home"]

            away_rate = season_win_rate(away.get("leagueRecord", {}))
            home_rate = season_win_rate(home.get("leagueRecord", {}))
            home_win_prob = log5(home_rate, away_rate)

            row = {
                "away_team": away["team"]["name"],
                "home_team": home["team"]["name"],
                "away_record": _format_record(away.get("leagueRecord", {})),
                "home_record": _format_record(home.get("leagueRecord", {})),
                "away_win_rate": round(away_rate, 3),
                "home_win_rate": round(home_rate, 3),
                "home_win_prob": round(home_win_prob, 3),
                "away_win_prob": round(1 - home_win_prob, 3),
                "status": game.get("status", {}).get("detailedState", ""),
                "venue": game.get("venue", {}).get("name", ""),
            }

            # 이미 끝났거나 진행 중인 경기는 점수도 함께 담는다.
            if "score" in away and "score" in home:
                row["away_score"] = away["score"]
                row["home_score"] = home["score"]

            games.append(row)
    return games


def _format_record(league_record: dict) -> str:
    return f"{league_record.get('wins', 0)}-{league_record.get('losses', 0)}"


def format_table(games: list[dict], game_date: str) -> str:
    """경기 목록을 읽기 좋은 표 형태의 문자열로 만든다."""
    if not games:
        return f"{game_date}에는 예정된 MLB 경기가 없습니다."

    lines = [f"=== {game_date} MLB 경기 승률 ({len(games)}경기) ==="]
    header = (
        f"{'원정팀':<24} {'전적':>7} {'승률':>6} | "
        f"{'홈팀':<24} {'전적':>7} {'승률':>6} | "
        f"{'홈팀 승리확률':>10}  상태"
    )
    lines.append(header)
    lines.append("-" * len(header))

    for game in games:
        score = ""
        if "away_score" in game:
            score = f" ({game['away_score']}:{game['home_score']})"
        lines.append(
            f"{game['away_team']:<24} {game['away_record']:>7} {game['away_win_rate']:>6.3f} | "
            f"{game['home_team']:<24} {game['home_record']:>7} {game['home_win_rate']:>6.3f} | "
            f"{game['home_win_prob']:>12.1%}  {game['status']}{score}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="그날의 MLB 경기 승률 계산")
    parser.add_argument(
        "--date",
        default=date.today().isoformat(),
        help="조회할 날짜 (YYYY-MM-DD, 기본값: 오늘)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="표 대신 JSON으로 출력",
    )
    args = parser.parse_args(argv)

    try:
        schedule = fetch_schedule(args.date)
    except (urllib.error.URLError, TimeoutError) as error:
        print(f"MLB Stats API 호출에 실패했습니다: {error}", file=sys.stderr)
        return 1

    games = parse_games(schedule)

    if args.json:
        print(json.dumps({"date": args.date, "games": games}, ensure_ascii=False, indent=2))
    else:
        print(format_table(games, args.date))
    return 0


if __name__ == "__main__":
    sys.exit(main())
