"""mlb_win_rate 모듈 단위 테스트.

MLB Stats API 실제 응답 구조를 본뜬 샘플 데이터로
각 컴포넌트 계산과 앙상블 결합 로직을 검증한다.
"""

import json
import os
import tempfile
import unittest

from mlb_win_rate import (
    ELO_INITIAL,
    HOME_ADVANTAGE_LOGIT,
    LEAGUE_AVG_FIP,
    WEIGHTS,
    compute_elo_ratings,
    elo_win_prob,
    evaluation_stats,
    extract_final_results,
    format_evaluation,
    load_model,
    update_elo,
    build_pitcher_index,
    build_standings_index,
    collect_probable_pitcher_ids,
    combine_components,
    compute_components,
    fip_from_stat,
    format_table,
    log5,
    logit,
    padded_rate,
    parse_games,
    parse_innings,
    pitcher_matchup_prob,
    pythagenpat,
    season_win_rate,
    sigmoid,
)

# ---------------------------------------------------------------------------
# 샘플 API 응답 (실제 스키마 축약본)
# ---------------------------------------------------------------------------

SAMPLE_SCHEDULE = {
    "dates": [
        {
            "date": "2026-08-03",
            "games": [
                {
                    "gamePk": 1,
                    "status": {"detailedState": "Final"},
                    "venue": {"name": "Dodger Stadium"},
                    "teams": {
                        "away": {
                            "team": {"id": 137, "name": "San Francisco Giants"},
                            "leagueRecord": {"wins": 60, "losses": 50, "pct": ".545"},
                            "probablePitcher": {"id": 1001, "fullName": "Ace Away"},
                            "score": 3,
                        },
                        "home": {
                            "team": {"id": 119, "name": "Los Angeles Dodgers"},
                            "leagueRecord": {"wins": 70, "losses": 40, "pct": ".636"},
                            "probablePitcher": {"id": 1002, "fullName": "Ace Home"},
                            "score": 5,
                        },
                    },
                },
                {
                    "gamePk": 2,
                    "status": {"detailedState": "Scheduled"},
                    "venue": {"name": "Yankee Stadium"},
                    "teams": {
                        "away": {
                            "team": {"id": 111, "name": "Boston Red Sox"},
                            "leagueRecord": {"wins": 55, "losses": 55, "pct": ".500"},
                        },
                        "home": {
                            "team": {"id": 147, "name": "New York Yankees"},
                            "leagueRecord": {"wins": 66, "losses": 44, "pct": ".600"},
                        },
                    },
                },
            ],
        }
    ]
}

SAMPLE_STANDINGS = {
    "records": [
        {
            "league": {"id": 104},
            "teamRecords": [
                {
                    "team": {"id": 119, "name": "Los Angeles Dodgers"},
                    "wins": 70,
                    "losses": 40,
                    "gamesPlayed": 110,
                    "runsScored": 580,
                    "runsAllowed": 450,
                    "records": {
                        "splitRecords": [
                            {"wins": 40, "losses": 15, "type": "home"},
                            {"wins": 30, "losses": 25, "type": "away"},
                            {"wins": 8, "losses": 2, "type": "lastTen"},
                        ]
                    },
                },
                {
                    "team": {"id": 137, "name": "San Francisco Giants"},
                    "wins": 60,
                    "losses": 50,
                    "gamesPlayed": 110,
                    "runsScored": 500,
                    "runsAllowed": 480,
                    "records": {
                        "splitRecords": [
                            {"wins": 33, "losses": 22, "type": "home"},
                            {"wins": 27, "losses": 28, "type": "away"},
                            {"wins": 4, "losses": 6, "type": "lastTen"},
                        ]
                    },
                },
            ],
        }
    ]
}

SAMPLE_PEOPLE = {
    "people": [
        {
            "id": 1001,
            "fullName": "Ace Away",
            "stats": [
                {
                    "group": {"displayName": "pitching"},
                    "type": {"displayName": "season"},
                    "splits": [
                        {
                            "stat": {
                                "inningsPitched": "120.0",
                                "strikeOuts": 100,
                                "baseOnBalls": 40,
                                "hitByPitch": 5,
                                "homeRuns": 15,
                            }
                        }
                    ],
                }
            ],
        },
        {
            "id": 1002,
            "fullName": "Ace Home",
            "stats": [
                {
                    "group": {"displayName": "pitching"},
                    "type": {"displayName": "season"},
                    "splits": [
                        {
                            "stat": {
                                "inningsPitched": "130.2",
                                "strikeOuts": 160,
                                "baseOnBalls": 30,
                                "hitByPitch": 3,
                                "homeRuns": 10,
                            }
                        }
                    ],
                }
            ],
        },
    ]
}


# ---------------------------------------------------------------------------
# 기본 수학 도구
# ---------------------------------------------------------------------------

class MathToolsTest(unittest.TestCase):
    def test_logit_sigmoid_roundtrip(self):
        for p in (0.1, 0.3, 0.5, 0.7, 0.9):
            self.assertAlmostEqual(sigmoid(logit(p)), p, places=9)

    def test_logit_clamps_extremes(self):
        self.assertEqual(logit(0.0), logit(0.01))
        self.assertEqual(logit(1.0), logit(0.99))

    def test_padded_rate_regresses_to_mean(self):
        # 5승 0패는 승률 1.000이 아니라 5할 쪽으로 당겨져야 한다.
        rate = padded_rate(5, 0, 33)
        self.assertGreater(rate, 0.5)
        self.assertLess(rate, 0.6)

    def test_padded_rate_full_season_barely_moves(self):
        # 시즌 후반의 큰 표본은 보정 영향이 작아야 한다.
        self.assertAlmostEqual(padded_rate(90, 60, 33), (90 + 16.5) / 183, places=9)


class SeasonWinRateTest(unittest.TestCase):
    def test_normal_record(self):
        self.assertAlmostEqual(season_win_rate({"wins": 60, "losses": 40}), 0.6)

    def test_no_games_played_defaults_to_half(self):
        self.assertEqual(season_win_rate({"wins": 0, "losses": 0}), 0.5)

    def test_missing_fields_default_to_half(self):
        self.assertEqual(season_win_rate({}), 0.5)


class Log5Test(unittest.TestCase):
    def test_equal_teams_is_coin_flip(self):
        self.assertAlmostEqual(log5(0.5, 0.5), 0.5)
        self.assertAlmostEqual(log5(0.7, 0.7), 0.5)

    def test_stronger_team_favored(self):
        self.assertAlmostEqual(log5(0.6, 0.4), 0.36 / 0.52)

    def test_symmetry(self):
        self.assertAlmostEqual(log5(0.55, 0.45) + log5(0.45, 0.55), 1.0)

    def test_degenerate_cases_return_half(self):
        self.assertEqual(log5(0.0, 0.0), 0.5)
        self.assertEqual(log5(1.0, 1.0), 0.5)


# ---------------------------------------------------------------------------
# 피타고리안
# ---------------------------------------------------------------------------

class PythagenpatTest(unittest.TestCase):
    def test_equal_runs_is_half(self):
        self.assertAlmostEqual(pythagenpat(500, 500, 110), 0.5)

    def test_positive_run_diff_above_half(self):
        self.assertGreater(pythagenpat(580, 450, 110), 0.5)

    def test_negative_run_diff_below_half(self):
        self.assertLess(pythagenpat(450, 580, 110), 0.5)

    def test_no_games_returns_half(self):
        self.assertEqual(pythagenpat(0, 0, 0), 0.5)

    def test_exponent_matches_hand_calc(self):
        # RS=580, RA=450, G=110 → x = (1030/110)^0.287
        x = (1030 / 110) ** 0.287
        expected = 580**x / (580**x + 450**x)
        self.assertAlmostEqual(pythagenpat(580, 450, 110), expected, places=9)


# ---------------------------------------------------------------------------
# 선발 투수
# ---------------------------------------------------------------------------

class PitcherTest(unittest.TestCase):
    def test_parse_innings_thirds(self):
        self.assertAlmostEqual(parse_innings("120.0"), 120.0)
        self.assertAlmostEqual(parse_innings("130.1"), 130 + 1 / 3)
        self.assertAlmostEqual(parse_innings("130.2"), 130 + 2 / 3)

    def test_parse_innings_bad_input(self):
        self.assertEqual(parse_innings(None), 0.0)
        self.assertEqual(parse_innings("abc"), 0.0)

    def test_fip_better_pitcher_is_lower(self):
        good, _ = fip_from_stat(
            {"inningsPitched": "130.2", "strikeOuts": 160, "baseOnBalls": 30,
             "hitByPitch": 3, "homeRuns": 10}
        )
        bad, _ = fip_from_stat(
            {"inningsPitched": "120.0", "strikeOuts": 100, "baseOnBalls": 40,
             "hitByPitch": 5, "homeRuns": 15}
        )
        self.assertLess(good, bad)

    def test_fip_no_innings_returns_league_average(self):
        fip, innings = fip_from_stat({"inningsPitched": "0"})
        self.assertEqual(fip, LEAGUE_AVG_FIP)
        self.assertEqual(innings, 0.0)

    def test_fip_small_sample_shrinks_toward_average(self):
        # 5이닝 9K 무사사구 무홈런 → 원시 FIP는 극단적으로 낮지만
        # 회귀 때문에 리그 평균 근처로 당겨져야 한다.
        fip, _ = fip_from_stat(
            {"inningsPitched": "5.0", "strikeOuts": 9, "baseOnBalls": 0, "homeRuns": 0}
        )
        self.assertGreater(fip, 3.0)

    def test_matchup_equal_fip_is_half(self):
        self.assertAlmostEqual(pitcher_matchup_prob(4.0, 4.0), 0.5)

    def test_matchup_better_home_pitcher_favored(self):
        self.assertGreater(pitcher_matchup_prob(3.0, 4.5), 0.5)
        self.assertLess(pitcher_matchup_prob(4.5, 3.0), 0.5)


# ---------------------------------------------------------------------------
# Elo
# ---------------------------------------------------------------------------

class EloTest(unittest.TestCase):
    def _result(self, home_score, away_score):
        return {
            "date": "2026-08-01",
            "home_id": 1,
            "away_id": 2,
            "home_score": home_score,
            "away_score": away_score,
        }

    def test_equal_ratings_neutral_prob_is_half(self):
        self.assertAlmostEqual(elo_win_prob(1500, 1500), 0.5)

    def test_higher_rating_favored(self):
        self.assertGreater(elo_win_prob(1550, 1450), 0.5)
        self.assertAlmostEqual(
            elo_win_prob(1550, 1450) + elo_win_prob(1450, 1550), 1.0
        )

    def test_home_win_raises_home_rating(self):
        ratings = {}
        update_elo(ratings, self._result(5, 3))
        self.assertGreater(ratings[1], ELO_INITIAL)
        self.assertLess(ratings[2], ELO_INITIAL)

    def test_rating_change_is_zero_sum(self):
        ratings = {1: 1520.0, 2: 1480.0}
        update_elo(ratings, self._result(2, 9))
        self.assertAlmostEqual(ratings[1] + ratings[2], 3000.0, places=9)

    def test_blowout_moves_more_than_close_game(self):
        close, blowout = {}, {}
        update_elo(close, self._result(4, 3))
        update_elo(blowout, self._result(10, 0))
        self.assertGreater(blowout[1] - ELO_INITIAL, close[1] - ELO_INITIAL)

    def test_upset_moves_more_than_expected_win(self):
        # 약팀(1400)이 강팀(1600)을 이기면 크게, 강팀이 이기면 조금 움직인다.
        upset = {1: 1400.0, 2: 1600.0}
        update_elo(upset, self._result(5, 3))
        expected = {1: 1600.0, 2: 1400.0}
        update_elo(expected, self._result(5, 3))
        self.assertGreater(upset[1] - 1400.0, expected[1] - 1600.0)

    def test_compute_elo_ratings_orders_by_date(self):
        results = [
            {"date": "2026-05-02", "home_id": 1, "away_id": 2,
             "home_score": 1, "away_score": 2},
            {"date": "2026-05-01", "home_id": 1, "away_id": 2,
             "home_score": 5, "away_score": 0},
        ]
        ratings = compute_elo_ratings(results)
        self.assertEqual(set(ratings), {1, 2})


class ExtractResultsTest(unittest.TestCase):
    def test_extracts_only_final_games_with_scores(self):
        schedule = {
            "dates": [
                {
                    "date": "2026-08-01",
                    "games": [
                        {
                            "status": {"abstractGameState": "Final"},
                            "teams": {
                                "away": {"team": {"id": 2}, "score": 3},
                                "home": {"team": {"id": 1}, "score": 5},
                            },
                        },
                        {
                            "status": {"abstractGameState": "Preview"},
                            "teams": {
                                "away": {"team": {"id": 4}},
                                "home": {"team": {"id": 3}},
                            },
                        },
                        {
                            # 동점(서스펜디드)은 제외돼야 한다
                            "status": {"abstractGameState": "Final"},
                            "teams": {
                                "away": {"team": {"id": 6}, "score": 2},
                                "home": {"team": {"id": 5}, "score": 2},
                            },
                        },
                    ],
                }
            ]
        }
        results = extract_final_results(schedule)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["home_id"], 1)
        self.assertEqual(results[0]["home_score"], 5)


# ---------------------------------------------------------------------------
# 가중치 파일 로딩
# ---------------------------------------------------------------------------

class LoadModelTest(unittest.TestCase):
    def _write(self, data):
        handle = tempfile.NamedTemporaryFile(
            "w", suffix=".json", delete=False, encoding="utf-8"
        )
        json.dump(data, handle)
        handle.close()
        self.addCleanup(os.unlink, handle.name)
        return handle.name

    def test_loads_valid_model(self):
        path = self._write(
            {"weights": {"season": 0.5, "elo": 0.5}, "scale": 2.0, "intercept": 0.12}
        )
        model = load_model(path)
        self.assertEqual(model["weights"]["elo"], 0.5)
        self.assertEqual(model["scale"], 2.0)
        self.assertEqual(model["intercept"], 0.12)

    def test_missing_optional_fields_get_defaults(self):
        path = self._write({"weights": {"season": 1.0}})
        model = load_model(path)
        self.assertEqual(model["scale"], 1.0)
        self.assertEqual(model["intercept"], HOME_ADVANTAGE_LOGIT)

    def test_negative_weight_rejected(self):
        path = self._write({"weights": {"season": -0.2}})
        with self.assertRaises(ValueError):
            load_model(path)

    def test_empty_weights_rejected(self):
        path = self._write({"weights": {}})
        with self.assertRaises(ValueError):
            load_model(path)


# ---------------------------------------------------------------------------
# 인덱싱
# ---------------------------------------------------------------------------

class IndexingTest(unittest.TestCase):
    def test_standings_index(self):
        index = build_standings_index(SAMPLE_STANDINGS)
        self.assertIn(119, index)
        self.assertEqual(index[119]["runs_scored"], 580)
        self.assertEqual(index[119]["home"]["wins"], 40)
        self.assertEqual(index[119]["last_ten"]["wins"], 8)

    def test_pitcher_index(self):
        index = build_pitcher_index(SAMPLE_PEOPLE)
        self.assertIn(1001, index)
        self.assertIn(1002, index)
        self.assertLess(index[1002]["fip"], index[1001]["fip"])

    def test_collect_pitcher_ids(self):
        self.assertEqual(collect_probable_pitcher_ids(SAMPLE_SCHEDULE), [1001, 1002])


# ---------------------------------------------------------------------------
# 컴포넌트 계산과 결합
# ---------------------------------------------------------------------------

class ComponentsTest(unittest.TestCase):
    def setUp(self):
        self.standings = build_standings_index(SAMPLE_STANDINGS)
        self.pitchers = build_pitcher_index(SAMPLE_PEOPLE)
        self.game1 = SAMPLE_SCHEDULE["dates"][0]["games"][0]
        self.game2 = SAMPLE_SCHEDULE["dates"][0]["games"][1]

    def test_full_data_yields_all_five_components(self):
        components = compute_components(self.game1, self.standings, self.pitchers)
        names = [component["name"] for component in components]
        self.assertEqual(
            names, ["season", "pythagorean", "split", "form", "pitcher"]
        )

    def test_missing_standings_and_pitchers_only_season(self):
        components = compute_components(self.game2, self.standings, self.pitchers)
        self.assertEqual([c["name"] for c in components], ["season"])

    def test_elo_ratings_add_sixth_component(self):
        elo = {119: 1560.0, 137: 1490.0}
        components = compute_components(
            self.game1, self.standings, self.pitchers, elo_ratings=elo
        )
        names = [component["name"] for component in components]
        self.assertEqual(
            names, ["season", "pythagorean", "split", "form", "pitcher", "elo"]
        )
        elo_component = components[-1]
        self.assertGreater(elo_component["prob"], 0.5)  # 다저스 레이팅 우위

    def test_custom_weights_are_applied(self):
        custom = dict(WEIGHTS, season=0.99)
        components = compute_components(
            self.game2, self.standings, self.pitchers, weights=custom
        )
        self.assertEqual(components[0]["weight"], 0.99)

    def test_learned_model_scale_and_intercept_used(self):
        component = {"name": "season", "prob": 0.6, "weight": 1.0, "detail": ""}
        model = {"weights": {"season": 1.0}, "scale": 2.0, "intercept": 0.0}
        expected = sigmoid(2.0 * logit(0.6))
        self.assertAlmostEqual(
            combine_components([component], model), expected, places=9
        )

    def test_all_components_favor_dodgers(self):
        # 샘플 데이터에서 다저스가 전 지표 우위 → 모든 컴포넌트 > 0.5
        components = compute_components(self.game1, self.standings, self.pitchers)
        for component in components:
            self.assertGreater(
                component["prob"], 0.5, msg=f"{component['name']} 컴포넌트"
            )

    def test_combine_equal_components_is_home_advantage_only(self):
        components = [
            {"name": name, "prob": 0.5, "weight": weight, "detail": ""}
            for name, weight in WEIGHTS.items()
        ]
        expected = sigmoid(HOME_ADVANTAGE_LOGIT)
        self.assertAlmostEqual(combine_components(components), expected, places=9)

    def test_combine_empty_falls_back_to_home_advantage(self):
        self.assertAlmostEqual(
            combine_components([]), sigmoid(HOME_ADVANTAGE_LOGIT), places=9
        )

    def test_combine_renormalizes_missing_weights(self):
        # season 하나만 있을 때 그 확률이 (홈보정 전) 그대로 반영돼야 한다.
        component = {"name": "season", "prob": 0.6, "weight": WEIGHTS["season"], "detail": ""}
        expected = sigmoid(logit(0.6) + HOME_ADVANTAGE_LOGIT)
        self.assertAlmostEqual(combine_components([component]), expected, places=9)


# ---------------------------------------------------------------------------
# 경기 파싱과 출력
# ---------------------------------------------------------------------------

class ParseGamesTest(unittest.TestCase):
    def setUp(self):
        self.games = parse_games(
            SAMPLE_SCHEDULE,
            build_standings_index(SAMPLE_STANDINGS),
            build_pitcher_index(SAMPLE_PEOPLE),
        )

    def test_parses_all_games(self):
        self.assertEqual(len(self.games), 2)

    def test_team_names_and_records(self):
        first = self.games[0]
        self.assertEqual(first["away_team"], "San Francisco Giants")
        self.assertEqual(first["home_team"], "Los Angeles Dodgers")
        self.assertEqual(first["away_record"], "60-50")
        self.assertEqual(first["home_record"], "70-40")

    def test_probabilities_sum_to_one(self):
        for game in self.games:
            self.assertAlmostEqual(
                game["home_win_prob"] + game["away_win_prob"], 1.0, places=3
            )

    def test_full_data_game_favors_dodgers_strongly(self):
        first = self.games[0]
        self.assertGreater(first["home_win_prob"], 0.55)
        self.assertEqual(len(first["components"]), 5)

    def test_partial_data_game_still_produces_probability(self):
        second = self.games[1]
        self.assertEqual(len(second["components"]), 1)
        self.assertGreater(second["home_win_prob"], 0.5)  # 양키스 전적 우위 + 홈

    def test_pitcher_names_included(self):
        first = self.games[0]
        self.assertEqual(first["away_pitcher"], "Ace Away")
        self.assertEqual(first["home_pitcher"], "Ace Home")

    def test_final_game_includes_score(self):
        self.assertEqual(self.games[0]["away_score"], 3)
        self.assertEqual(self.games[0]["home_score"], 5)

    def test_no_indices_still_works(self):
        games = parse_games(SAMPLE_SCHEDULE)
        self.assertEqual(len(games), 2)
        for game in games:
            self.assertEqual([c["name"] for c in game["components"]], ["season"])

    def test_empty_schedule(self):
        self.assertEqual(parse_games({"dates": []}), [])


class FormatTableTest(unittest.TestCase):
    def setUp(self):
        self.games = parse_games(
            SAMPLE_SCHEDULE,
            build_standings_index(SAMPLE_STANDINGS),
            build_pitcher_index(SAMPLE_PEOPLE),
        )

    def test_no_games_message(self):
        output = format_table([], "2026-12-25")
        self.assertIn("예정된 MLB 경기가 없습니다", output)

    def test_table_contains_teams_probability_and_score(self):
        output = format_table(self.games, "2026-08-03")
        self.assertIn("Los Angeles Dodgers", output)
        self.assertIn("San Francisco Giants", output)
        self.assertIn("2경기", output)
        self.assertIn("(3:5)", output)
        self.assertIn("Ace Home", output)

    def test_detail_mode_shows_components(self):
        output = format_table(self.games, "2026-08-03", detail=True)
        self.assertIn("피타고리안", output)
        self.assertIn("선발 투수", output)
        self.assertIn("가중치", output)

    def test_finished_game_marked_hit_or_miss(self):
        # 다저스 홈승(5:3) 예측이 홈승이므로 '적중' 표시가 나와야 한다.
        output = format_table(self.games, "2026-08-03")
        self.assertIn("→ 적중", output)


class EvaluationStatsTest(unittest.TestCase):
    def setUp(self):
        self.games = parse_games(
            SAMPLE_SCHEDULE,
            build_standings_index(SAMPLE_STANDINGS),
            build_pitcher_index(SAMPLE_PEOPLE),
        )

    def test_counts_only_finished_games(self):
        stats = evaluation_stats(self.games)
        self.assertEqual(stats["games"], 1)  # game1만 Final
        self.assertEqual(stats["hits"], 1)   # 홈승 예측, 실제 홈승
        self.assertEqual(stats["accuracy"], 1.0)
        self.assertGreater(stats["log_loss"], 0.0)
        self.assertLess(stats["brier"], 0.25)

    def test_no_finished_games_returns_none(self):
        scheduled_only = [g for g in self.games if "home_score" not in g]
        self.assertIsNone(evaluation_stats(scheduled_only))

    def test_tie_games_excluded(self):
        tied = dict(self.games[0], home_score=4, away_score=4)
        self.assertIsNone(evaluation_stats([tied]))

    def test_miss_counted(self):
        upset = dict(self.games[0], home_score=1, away_score=9)
        stats = evaluation_stats([upset])
        self.assertEqual(stats["hits"], 0)
        self.assertEqual(stats["accuracy"], 0.0)

    def test_format_evaluation_output(self):
        text = format_evaluation(evaluation_stats(self.games))
        self.assertIn("예측 검증", text)
        self.assertIn("적중 1/1", text)


class SeasonFromStandingsTest(unittest.TestCase):
    def test_season_component_prefers_standings_record(self):
        # standings 전적(전날 기준)이 leagueRecord(경기 후)와 다를 때
        # standings 쪽을 써야 한다 — 과거 경기 예측의 미래 누수 방지.
        standings = {
            "records": [
                {
                    "teamRecords": [
                        {
                            "team": {"id": 119},
                            "wins": 69, "losses": 40, "gamesPlayed": 109,
                            "runsScored": 575, "runsAllowed": 448,
                            "records": {"splitRecords": []},
                        },
                        {
                            "team": {"id": 137},
                            "wins": 60, "losses": 49, "gamesPlayed": 109,
                            "runsScored": 497, "runsAllowed": 477,
                            "records": {"splitRecords": []},
                        },
                    ]
                }
            ]
        }
        game = SAMPLE_SCHEDULE["dates"][0]["games"][0]
        components = compute_components(game, build_standings_index(standings), {})
        season = components[0]
        self.assertEqual(season["name"], "season")
        # 69/(109+33) 기반 보정 승률이 detail에 나와야 한다 (70승이 아니라 69승)
        expected_home = (69 + 16.5) / (109 + 33)
        self.assertIn(f"{expected_home:.3f}", season["detail"])

    def test_falls_back_to_league_record_without_standings(self):
        game = SAMPLE_SCHEDULE["dates"][0]["games"][0]
        components = compute_components(game, {}, {})
        expected_home = (70 + 16.5) / (110 + 33)
        self.assertIn(f"{expected_home:.3f}", components[0]["detail"])


if __name__ == "__main__":
    unittest.main()
