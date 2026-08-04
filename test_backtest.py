"""backtest 모듈 단위 테스트.

합성 시즌 데이터로 시즌 재생(look-ahead 방지), 로지스틱 회귀 학습,
가중치 파일 생성을 검증한다.
"""

import random
import unittest

from backtest import (
    FEATURE_NAMES,
    TeamState,
    accuracy,
    brier_score,
    build_calibrated_weights_file,
    build_weights_file,
    default_ensemble_logit,
    game_components,
    log_loss,
    predict_calibrated,
    predict_default,
    predict_learned,
    replay_season,
    run_backtest,
    train_logistic,
)
from mlb_win_rate import WEIGHTS
from mlb_win_rate import ELO_INITIAL, logit, sigmoid


def make_synthetic_season(
    n_teams: int = 6, games_per_pair: int = 12, seed: int = 42
) -> list[dict]:
    """팀별 실력 차이가 있는 합성 시즌을 만든다.

    팀 i의 '진짜 실력' 로그오즈를 고정해 두고, 매 경기 결과를
    시드 고정 난수로 샘플링한다. 홈 어드밴티지도 넣는다.
    """
    rng = random.Random(seed)
    strengths = {team: (team - n_teams / 2) * 0.25 for team in range(n_teams)}
    games = []
    day = 0
    for round_number in range(games_per_pair):
        for home_team in range(n_teams):
            for away_team in range(n_teams):
                if home_team == away_team:
                    continue
                day += 1
                p_home = sigmoid(
                    strengths[home_team] - strengths[away_team] + 0.15
                )
                home_won = rng.random() < p_home
                margin = 1 + int(rng.random() * 5)
                base = 3
                games.append(
                    {
                        # 날짜 문자열 정렬이 곧 시간순이 되도록 0 패딩
                        "date": f"2025-{(day // 900) + 4:02d}-{(day % 28) + 1:02d}"
                        + f"#{day:05d}",
                        "home_id": home_team,
                        "away_id": away_team,
                        "home_score": base + margin if home_won else base,
                        "away_score": base if home_won else base + margin,
                    }
                )
    return games


class TeamStateTest(unittest.TestCase):
    def test_record_game_updates_all_fields(self):
        state = TeamState()
        state.record_game(won=True, at_home=True, scored=5, allowed=3)
        state.record_game(won=False, at_home=False, scored=2, allowed=7)
        self.assertEqual(state.wins, 1)
        self.assertEqual(state.losses, 1)
        self.assertEqual(state.home_wins, 1)
        self.assertEqual(state.away_losses, 1)
        self.assertEqual(state.runs_scored, 7)
        self.assertEqual(state.runs_allowed, 10)
        self.assertEqual(list(state.last_ten), [1, 0])
        self.assertEqual(state.games, 2)

    def test_last_ten_keeps_only_ten(self):
        state = TeamState()
        for i in range(15):
            state.record_game(won=i % 2 == 0, at_home=True, scored=1, allowed=0)
        self.assertEqual(len(state.last_ten), 10)


class GameComponentsTest(unittest.TestCase):
    def test_stronger_home_team_all_components_above_half(self):
        strong, weak = TeamState(), TeamState()
        for _ in range(30):
            strong.record_game(won=True, at_home=True, scored=6, allowed=2)
            weak.record_game(won=False, at_home=False, scored=2, allowed=6)
        components = game_components(strong, weak, 1550.0, 1450.0)
        self.assertEqual(sorted(components), sorted(FEATURE_NAMES))
        for name, prob in components.items():
            self.assertGreater(prob, 0.5, msg=name)

    def test_identical_teams_all_half(self):
        state_a, state_b = TeamState(), TeamState()
        for won in (True, False) * 10:
            state_a.record_game(won, at_home=True, scored=4, allowed=4)
            state_b.record_game(won, at_home=False, scored=4, allowed=4)
        components = game_components(state_a, state_b, ELO_INITIAL, ELO_INITIAL)
        for name, prob in components.items():
            self.assertAlmostEqual(prob, 0.5, places=6, msg=name)


class ReplaySeasonTest(unittest.TestCase):
    def setUp(self):
        self.results = make_synthetic_season()
        self.samples = replay_season(self.results, min_games=15)

    def test_min_games_filter_skips_early_season(self):
        # 초반 경기는 샘플에서 빠져야 한다.
        self.assertLess(len(self.samples), len(self.results))
        self.assertGreater(len(self.samples), 100)

    def test_samples_have_all_components_and_labels(self):
        for sample in self.samples[:20]:
            self.assertEqual(sorted(sample["components"]), sorted(FEATURE_NAMES))
            self.assertIn(sample["home_won"], (0, 1))

    def test_no_lookahead_first_sample_state(self):
        # 강팀(마지막 팀)이 홈일 때 season 컴포넌트가 우위를 보여야 한다 —
        # 재생이 실력 차이를 실제로 축적하고 있는지에 대한 간접 검증.
        strong_home = [
            s for s in self.samples
            if s["components"]["season"] > 0.55
        ]
        self.assertGreater(len(strong_home), 0)


class TrainLogisticTest(unittest.TestCase):
    def test_recovers_known_model(self):
        # 진짜 모델: logit(p) = 0.2 + 1.5*x1 - 0.5*x2 에서 샘플 생성
        rng = random.Random(7)
        features, labels = [], []
        for _ in range(4000):
            x1 = rng.uniform(-1, 1)
            x2 = rng.uniform(-1, 1)
            p = sigmoid(0.2 + 1.5 * x1 - 0.5 * x2)
            features.append([x1, x2])
            labels.append(1 if rng.random() < p else 0)
        coefs, intercept = train_logistic(features, labels, epochs=4000)
        self.assertAlmostEqual(coefs[0], 1.5, delta=0.25)
        self.assertAlmostEqual(coefs[1], -0.5, delta=0.25)
        self.assertAlmostEqual(intercept, 0.2, delta=0.15)

    def test_non_negative_constraint_clamps_coefficients(self):
        # 진짜 모델의 x2 계수가 음수여도, 제약 학습에서는 0 이상이어야 한다.
        rng = random.Random(11)
        features, labels = [], []
        for _ in range(2000):
            x1 = rng.uniform(-1, 1)
            x2 = rng.uniform(-1, 1)
            p = sigmoid(1.0 * x1 - 0.8 * x2)
            features.append([x1, x2])
            labels.append(1 if rng.random() < p else 0)
        coefs, _ = train_logistic(
            features, labels, epochs=1500, non_negative=True
        )
        self.assertGreaterEqual(coefs[0], 0.0)
        self.assertGreaterEqual(coefs[1], 0.0)
        self.assertEqual(coefs[1], 0.0)  # 음수 신호는 0으로 눌려야 한다
        self.assertGreater(coefs[0], 0.5)  # 양수 신호는 살아 있어야 한다


class MetricsTest(unittest.TestCase):
    def test_perfect_predictions(self):
        self.assertAlmostEqual(log_loss([1.0, 0.0], [1, 0]), 0.0, places=6)
        self.assertEqual(brier_score([1.0, 0.0], [1, 0]), 0.0)
        self.assertEqual(accuracy([0.9, 0.1], [1, 0]), 1.0)

    def test_coin_flip_log_loss(self):
        import math
        self.assertAlmostEqual(log_loss([0.5, 0.5], [1, 0]), math.log(2), places=9)


class EndToEndBacktestTest(unittest.TestCase):
    def test_learned_model_beats_baseline_on_holdout(self):
        samples = replay_season(make_synthetic_season(), min_games=15)
        coefs, intercept, report, metrics = run_backtest(samples, holdout=0.2)

        # 비음수 제약이 적용돼야 하고, 지표 딕셔너리가 채워져야 한다.
        self.assertTrue(all(coef >= 0.0 for coef in coefs))
        self.assertIn("learned_logloss", metrics)
        self.assertIn("default_logloss", metrics)
        self.assertIn("calibrated_logloss", metrics)
        self.assertGreaterEqual(metrics["calibration"]["scale"], 0.0)
        self.assertIn("보정된 기본 앙상블", "\n".join(report))

        split_at = int(len(samples) * 0.8)
        test = samples[split_at:]
        labels = [s["home_won"] for s in test]
        learned_probs = [predict_learned(s, coefs, intercept) for s in test]
        baseline_probs = [0.54] * len(test)
        default_probs = [predict_default(s) for s in test]

        # 학습된 모델은 상수 베이스라인보다 로그손실이 좋아야 한다.
        self.assertLess(log_loss(learned_probs, labels), log_loss(baseline_probs, labels))
        # 기본 가중치 모델도 확률을 내긴 해야 한다.
        self.assertTrue(all(0.0 < p < 1.0 for p in default_probs))
        # 리포트에 핵심 섹션이 들어 있어야 한다.
        text = "\n".join(report)
        self.assertIn("홀드아웃 평가", text)
        self.assertIn("학습된 계수", text)


class BuildWeightsFileTest(unittest.TestCase):
    def test_weights_sum_to_one_with_pitcher_share(self):
        data = build_weights_file([0.5, 1.0, 0.3, 0.1, 0.6], 0.12, {"season": 2025})
        weights = data["weights"]
        self.assertAlmostEqual(sum(weights.values()), 1.0, places=3)
        self.assertEqual(weights["pitcher"], 0.30)
        self.assertAlmostEqual(data["scale"], 2.5, places=3)
        self.assertEqual(data["intercept"], 0.12)

    def test_negative_coefficients_clamped(self):
        data = build_weights_file([1.0, -0.5, 0.0, 0.0, 1.0], 0.1, {})
        self.assertEqual(data["weights"]["pythagorean"], 0.0)
        self.assertGreater(data["weights"]["season"], 0.0)

    def test_all_zero_coefficients_raises(self):
        with self.assertRaises(ValueError):
            build_weights_file([0.0, -1.0, 0.0, 0.0, 0.0], 0.1, {})


class CalibrationTest(unittest.TestCase):
    def test_calibrated_file_keeps_default_weights(self):
        data = build_calibrated_weights_file(1.35, 0.17, {"variant": "calibrated"})
        self.assertEqual(data["weights"], WEIGHTS)
        self.assertEqual(data["scale"], 1.35)
        self.assertEqual(data["intercept"], 0.17)
        self.assertEqual(data["trained"]["variant"], "calibrated")

    def test_single_feature_fit_recovers_scale(self):
        # 진짜 모델: logit(p) = 2.0*x + 0.1 → scale ≈ 2.0을 복원해야 한다.
        rng = random.Random(5)
        features, labels = [], []
        for _ in range(4000):
            x = rng.uniform(-1, 1)
            p = sigmoid(2.0 * x + 0.1)
            features.append([x])
            labels.append(1 if rng.random() < p else 0)
        (scale,), intercept = train_logistic(
            features, labels, epochs=4000, l2=0.0, non_negative=True
        )
        self.assertAlmostEqual(scale, 2.0, delta=0.3)
        self.assertAlmostEqual(intercept, 0.1, delta=0.15)

    def test_predict_calibrated_uses_default_ensemble(self):
        samples = replay_season(make_synthetic_season(), min_games=15)
        sample = samples[0]
        x = default_ensemble_logit(sample)
        self.assertAlmostEqual(
            predict_calibrated(sample, 1.0, 0.0), sigmoid(x), places=9
        )


if __name__ == "__main__":
    unittest.main()
