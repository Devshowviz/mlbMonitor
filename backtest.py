#!/usr/bin/env python3
"""과거 시즌 데이터로 앙상블 가중치를 학습하고 예측 성능을 평가하는 백테스트.

시즌 전체 경기 결과를 API 한 번으로 받아온 뒤 하루씩 시간순으로 재생한다.
각 경기마다 "그 경기 시작 전까지의 데이터"만으로 5가지 전적 기반 컴포넌트
(season, pythagorean, split, form, elo)의 홈팀 승리 확률을 계산하고,
실제 승패와 비교해 로지스틱 회귀로 최적 가중치를 학습한다.

미래 정보 누수(look-ahead bias)가 없도록 컴포넌트 계산 후에만 팀 상태를
갱신하며, 평가도 시간순 뒤쪽 20%(홀드아웃)로만 한다.

선발 투수 컴포넌트는 "그 시점까지의 투수 스탯"을 API로 복원하기 어려워
학습에서 제외하고, weights.json을 쓸 때 기본 가중치 몫을 그대로 남겨둔다.

사용 예:
    python backtest.py --season 2025
    python backtest.py --season 2025 --output weights.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import urllib.error
from collections import deque

from mlb_win_rate import (
    ELO_INITIAL,
    HOME_ADVANTAGE_LOGIT,
    PAD_FORM,
    PAD_SEASON,
    PAD_SPLIT,
    WEIGHTS,
    elo_win_prob,
    fetch_season_results,
    log5,
    logit,
    padded_rate,
    pythagenpat,
    sigmoid,
    update_elo,
)

# 학습 대상 컴포넌트 (전적 기반 — 경기 결과만으로 시점 복원이 가능한 것들)
FEATURE_NAMES = ["season", "pythagorean", "split", "form", "elo"]

# 학습에서 제외되는 선발 투수 컴포넌트가 weights.json에서 가져갈 몫
PITCHER_WEIGHT_SHARE = WEIGHTS["pitcher"]


# ---------------------------------------------------------------------------
# 시즌 재생 (look-ahead 없는 시점별 상태)
# ---------------------------------------------------------------------------

class TeamState:
    """재생 시점까지의 팀 누적 상태."""

    def __init__(self) -> None:
        self.wins = 0
        self.losses = 0
        self.home_wins = 0
        self.home_losses = 0
        self.away_wins = 0
        self.away_losses = 0
        self.runs_scored = 0
        self.runs_allowed = 0
        self.last_ten: deque[int] = deque(maxlen=10)  # 1=승, 0=패

    @property
    def games(self) -> int:
        return self.wins + self.losses

    def record_game(self, won: bool, at_home: bool, scored: int, allowed: int) -> None:
        if won:
            self.wins += 1
            if at_home:
                self.home_wins += 1
            else:
                self.away_wins += 1
        else:
            self.losses += 1
            if at_home:
                self.home_losses += 1
            else:
                self.away_losses += 1
        self.runs_scored += scored
        self.runs_allowed += allowed
        self.last_ten.append(1 if won else 0)


def game_components(
    home: TeamState, away: TeamState, home_elo: float, away_elo: float
) -> dict[str, float]:
    """예측 모델(mlb_win_rate)과 같은 공식으로 컴포넌트별 홈승 확률을 계산한다."""
    form_home_wins = sum(home.last_ten)
    form_away_wins = sum(away.last_ten)
    return {
        "season": log5(
            padded_rate(home.wins, home.losses, PAD_SEASON),
            padded_rate(away.wins, away.losses, PAD_SEASON),
        ),
        "pythagorean": log5(
            pythagenpat(home.runs_scored, home.runs_allowed, home.games),
            pythagenpat(away.runs_scored, away.runs_allowed, away.games),
        ),
        "split": log5(
            padded_rate(home.home_wins, home.home_losses, PAD_SPLIT),
            padded_rate(away.away_wins, away.away_losses, PAD_SPLIT),
        ),
        "form": log5(
            padded_rate(form_home_wins, len(home.last_ten) - form_home_wins, PAD_FORM),
            padded_rate(form_away_wins, len(away.last_ten) - form_away_wins, PAD_FORM),
        ),
        "elo": elo_win_prob(home_elo, away_elo),
    }


def replay_season(results: list[dict], min_games: int = 15) -> list[dict]:
    """시즌 결과를 시간순으로 재생하며 학습 샘플을 만든다.

    각 샘플: {"components": {이름: 홈승확률}, "home_won": 0/1, "date": ...}
    두 팀 모두 min_games 이상 치른 경기만 샘플로 쓴다(시즌 초반 노이즈 제거).
    """
    states: dict[int, TeamState] = {}
    ratings: dict[int, float] = {}
    samples = []

    for result in sorted(results, key=lambda r: r["date"]):
        home = states.setdefault(result["home_id"], TeamState())
        away = states.setdefault(result["away_id"], TeamState())

        if home.games >= min_games and away.games >= min_games:
            components = game_components(
                home,
                away,
                ratings.get(result["home_id"], ELO_INITIAL),
                ratings.get(result["away_id"], ELO_INITIAL),
            )
            samples.append(
                {
                    "components": components,
                    "home_won": 1 if result["home_score"] > result["away_score"] else 0,
                    "date": result["date"],
                }
            )

        # 상태 갱신은 반드시 컴포넌트 계산 뒤에 (look-ahead 방지)
        home_won = result["home_score"] > result["away_score"]
        home.record_game(home_won, True, result["home_score"], result["away_score"])
        away.record_game(not home_won, False, result["away_score"], result["home_score"])
        update_elo(ratings, result)

    return samples


# ---------------------------------------------------------------------------
# 로지스틱 회귀 (표준 라이브러리만 사용)
# ---------------------------------------------------------------------------

def train_logistic(
    features: list[list[float]],
    labels: list[int],
    learning_rate: float = 0.3,
    epochs: int = 3000,
    l2: float = 0.001,
) -> tuple[list[float], float]:
    """배치 경사하강법으로 로지스틱 회귀를 학습한다.

    모델: P(홈승) = sigmoid(intercept + Σ coef_i · feature_i)
    feature는 각 컴포넌트 확률의 로그오즈.
    반환: (컴포넌트별 계수, 절편)
    """
    n_samples = len(features)
    n_features = len(features[0])
    coefs = [0.0] * n_features
    intercept = 0.0

    for _ in range(epochs):
        grad_coefs = [0.0] * n_features
        grad_intercept = 0.0
        for row, label in zip(features, labels):
            pred = sigmoid(intercept + sum(c * x for c, x in zip(coefs, row)))
            error = pred - label
            for j in range(n_features):
                grad_coefs[j] += error * row[j]
            grad_intercept += error
        for j in range(n_features):
            coefs[j] -= learning_rate * (grad_coefs[j] / n_samples + l2 * coefs[j])
        intercept -= learning_rate * grad_intercept / n_samples

    return coefs, intercept


# ---------------------------------------------------------------------------
# 평가 지표
# ---------------------------------------------------------------------------

def log_loss(probs: list[float], labels: list[int]) -> float:
    total = 0.0
    for p, y in zip(probs, labels):
        p = min(max(p, 1e-12), 1 - 1e-12)
        total += -(y * math.log(p) + (1 - y) * math.log(1 - p))
    return total / len(probs)


def brier_score(probs: list[float], labels: list[int]) -> float:
    return sum((p - y) ** 2 for p, y in zip(probs, labels)) / len(probs)


def accuracy(probs: list[float], labels: list[int]) -> float:
    return sum(1 for p, y in zip(probs, labels) if (p >= 0.5) == (y == 1)) / len(probs)


def evaluate(name: str, probs: list[float], labels: list[int]) -> str:
    return (
        f"{name:<22} 로그손실 {log_loss(probs, labels):.4f}"
        f"  브라이어 {brier_score(probs, labels):.4f}"
        f"  적중률 {accuracy(probs, labels):.1%}"
    )


# ---------------------------------------------------------------------------
# 예측 함수들 (평가용)
# ---------------------------------------------------------------------------

def predict_learned(sample: dict, coefs: list[float], intercept: float) -> float:
    z = intercept + sum(
        coef * logit(sample["components"][name])
        for coef, name in zip(coefs, FEATURE_NAMES)
    )
    return sigmoid(z)


def predict_default(sample: dict) -> float:
    """mlb_win_rate 기본 가중치(전적 기반 컴포넌트만)로 예측."""
    total = sum(WEIGHTS[name] for name in FEATURE_NAMES)
    weighted = (
        sum(WEIGHTS[name] * logit(sample["components"][name]) for name in FEATURE_NAMES)
        / total
    )
    return sigmoid(weighted + HOME_ADVANTAGE_LOGIT)


# ---------------------------------------------------------------------------
# weights.json 생성
# ---------------------------------------------------------------------------

def build_weights_file(coefs: list[float], intercept: float, meta: dict) -> dict:
    """학습된 계수를 mlb_win_rate가 읽는 weights.json 형식으로 변환한다.

    - 음수 계수는 0으로 클램프한다 (해당 컴포넌트는 사실상 제외).
    - 학습된 5개 컴포넌트가 (1 - 선발투수 몫)을 나눠 갖고,
      학습 불가능한 선발 투수 컴포넌트는 기본 몫을 유지한다.
    - scale = 계수 합 (가중 평균 → 회귀 모델의 크기를 복원).
    """
    clamped = [max(coef, 0.0) for coef in coefs]
    total = sum(clamped)
    if total == 0:
        raise ValueError("모든 학습 계수가 0 이하입니다 — 데이터를 확인하세요.")

    weights = {
        name: round((coef / total) * (1.0 - PITCHER_WEIGHT_SHARE), 4)
        for name, coef in zip(FEATURE_NAMES, clamped)
    }
    weights["pitcher"] = PITCHER_WEIGHT_SHARE
    return {
        "weights": weights,
        "scale": round(total, 4),
        "intercept": round(intercept, 4),
        "trained": meta,
    }


# ---------------------------------------------------------------------------
# 메인
# ---------------------------------------------------------------------------

def run_backtest(
    samples: list[dict], holdout: float = 0.2
) -> tuple[list[float], float, list[str]]:
    """샘플을 시간순으로 나눠 학습/평가하고 (계수, 절편, 리포트)를 반환한다."""
    split_at = int(len(samples) * (1.0 - holdout))
    train, test = samples[:split_at], samples[split_at:]

    train_x = [
        [logit(s["components"][name]) for name in FEATURE_NAMES] for s in train
    ]
    train_y = [s["home_won"] for s in train]
    coefs, intercept = train_logistic(train_x, train_y)

    test_y = [s["home_won"] for s in test]
    report = [
        f"학습 샘플 {len(train)}경기 / 평가 샘플 {len(test)}경기 (시간순 홀드아웃)",
        "",
        "--- 홀드아웃 평가 ---",
        evaluate("학습된 앙상블", [predict_learned(s, coefs, intercept) for s in test], test_y),
        evaluate("기본 가중치 앙상블", [predict_default(s) for s in test], test_y),
        evaluate("항상 홈팀 54%", [0.54] * len(test), test_y),
        "",
        "--- 컴포넌트 단독 성능 (홀드아웃) ---",
    ]
    for name in FEATURE_NAMES:
        report.append(
            evaluate(f"  {name}", [s["components"][name] for s in test], test_y)
        )
    report.append("")
    report.append("--- 학습된 계수 (로그오즈 기준) ---")
    for name, coef in zip(FEATURE_NAMES, coefs):
        report.append(f"  {name:<12} {coef:+.4f}")
    report.append(f"  {'intercept':<12} {intercept:+.4f} (홈 어드밴티지)")
    return coefs, intercept, report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="MLB 승률 앙상블 가중치 백테스트")
    parser.add_argument("--season", type=int, required=True, help="백테스트할 시즌 (예: 2025)")
    parser.add_argument(
        "--end-date", default=None, help="이 날짜까지의 경기만 사용 (기본: 시즌 전체)"
    )
    parser.add_argument(
        "--min-games", type=int, default=15, help="샘플로 쓰기 위한 팀당 최소 경기 수"
    )
    parser.add_argument(
        "--holdout", type=float, default=0.2, help="평가용 홀드아웃 비율 (시간순 뒤쪽)"
    )
    parser.add_argument(
        "--output", default=None, help="학습된 가중치를 저장할 경로 (예: weights.json)"
    )
    args = parser.parse_args(argv)

    end_date = args.end_date or f"{args.season}-12-01"
    print(f"{args.season} 시즌 결과를 가져오는 중...", file=sys.stderr)
    try:
        results = fetch_season_results(args.season, end_date)
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        print(f"MLB Stats API 호출에 실패했습니다: {error}", file=sys.stderr)
        return 1

    if not results:
        print("해당 기간에 끝난 경기가 없습니다.", file=sys.stderr)
        return 1

    print(f"완료된 경기 {len(results)}건, 시즌 재생 중...", file=sys.stderr)
    samples = replay_season(results, min_games=args.min_games)
    if len(samples) < 100:
        print(f"학습 샘플이 너무 적습니다 ({len(samples)}건).", file=sys.stderr)
        return 1

    coefs, intercept, report = run_backtest(samples, holdout=args.holdout)
    print("\n".join(report))

    if args.output:
        weights_data = build_weights_file(
            coefs,
            intercept,
            {"season": args.season, "samples": len(samples), "end_date": end_date},
        )
        with open(args.output, "w", encoding="utf-8") as file:
            json.dump(weights_data, file, ensure_ascii=False, indent=2)
        print(f"\n가중치를 저장했습니다: {args.output}")
        print("이제 `python mlb_win_rate.py`가 이 가중치를 자동으로 사용합니다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
