"""mlb_win_rate 모듈 단위 테스트.

MLB Stats API 실제 응답 구조를 본뜬 샘플 데이터로
승률 계산과 파싱 로직을 검증한다.
"""

import unittest

from mlb_win_rate import format_table, log5, parse_games, season_win_rate

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
                            "score": 3,
                        },
                        "home": {
                            "team": {"id": 119, "name": "Los Angeles Dodgers"},
                            "leagueRecord": {"wins": 70, "losses": 40, "pct": ".636"},
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
        # 승률 0.6 팀 vs 0.4 팀 → log5 = 0.36/0.52 ≈ 0.6923
        self.assertAlmostEqual(log5(0.6, 0.4), 0.36 / 0.52)

    def test_symmetry(self):
        self.assertAlmostEqual(log5(0.55, 0.45) + log5(0.45, 0.55), 1.0)

    def test_degenerate_cases_return_half(self):
        self.assertEqual(log5(0.0, 0.0), 0.5)
        self.assertEqual(log5(1.0, 1.0), 0.5)


class ParseGamesTest(unittest.TestCase):
    def setUp(self):
        self.games = parse_games(SAMPLE_SCHEDULE)

    def test_parses_all_games(self):
        self.assertEqual(len(self.games), 2)

    def test_team_names_and_records(self):
        first = self.games[0]
        self.assertEqual(first["away_team"], "San Francisco Giants")
        self.assertEqual(first["home_team"], "Los Angeles Dodgers")
        self.assertEqual(first["away_record"], "60-50")
        self.assertEqual(first["home_record"], "70-40")

    def test_win_rates(self):
        first = self.games[0]
        self.assertAlmostEqual(first["away_win_rate"], round(60 / 110, 3))
        self.assertAlmostEqual(first["home_win_rate"], round(70 / 110, 3))

    def test_home_win_probability_favors_better_team(self):
        first = self.games[0]
        self.assertGreater(first["home_win_prob"], 0.5)
        self.assertAlmostEqual(
            first["home_win_prob"] + first["away_win_prob"], 1.0, places=3
        )

    def test_final_game_includes_score(self):
        self.assertEqual(self.games[0]["away_score"], 3)
        self.assertEqual(self.games[0]["home_score"], 5)

    def test_scheduled_game_has_no_score(self):
        self.assertNotIn("away_score", self.games[1])

    def test_empty_schedule(self):
        self.assertEqual(parse_games({"dates": []}), [])


class FormatTableTest(unittest.TestCase):
    def test_no_games_message(self):
        output = format_table([], "2026-12-25")
        self.assertIn("예정된 MLB 경기가 없습니다", output)

    def test_table_contains_teams_and_probability(self):
        output = format_table(parse_games(SAMPLE_SCHEDULE), "2026-08-03")
        self.assertIn("Los Angeles Dodgers", output)
        self.assertIn("San Francisco Giants", output)
        self.assertIn("2경기", output)
        self.assertIn("(3:5)", output)


if __name__ == "__main__":
    unittest.main()
