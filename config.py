# --------------------------
# CONFIGURATION FOR BIAS MODEL
# --------------------------

# Tiered Defensive Factor Scores (used for stat normalization)
defensive_rushing_factors = {
    "top_5": 90,
    "top_10": 85,
    "top_15": 70,
    "top_20": 60,
    "top_25": 50,
    "bottom_32": 40
}

defensive_passing_factors = {
    "top_5": 90,
    "top_10": 85,
    "top_15": 70,
    "top_20": 60,
    "top_25": 50,
    "bottom_32": 40
}

defensive_points_factors = {
    "top_5": 90,
    "top_10": 85,
    "top_15": 70,
    "top_20": 60,
    "top_25": 50,
    "bottom_32": 40
}

# Position-based factor weight model
# Positive weights push toward OVER; negative weights push toward UNDER
factors_by_position_stat = {
    "RB": {
        "rushing_yards": {
            "rush_defense": -1.2,
            "oline_ranking": 0.9,
            "rush_rate": 0.7,
            "yards_per_carry": 1.1,
            "carries": 0.8,
            "past_games": 0.85,
            "h2h": 1.1
        },
        "rushing_tds": {
            "rush_defense": -0.8,
            "oline_ranking": 0.8,
            "red_zone_usage": 1.2,
            "points_allowed": 0.9,
            "past_games": 0.85,
            "h2h": 1.1
        },
        "receiving_yards": {
            "defensive_rankings": -0.8,
            "targets": 0.75,
            "past_games": 0.85,
            "h2h": 1.1
        }
        # },
        # "receiving_yards": {
        #     "defensive_rankings" : -0.8,
        #     "targets" : 0.75,
        #     "sum else here" : 0.1
        # }
    },
    "QB": {
        "passing_yards": {
            "pass_defense": -1.1,
            "oline_ranking": 0.8,
            "pass_attempts": 0.9,
            "past_games": 0.85,
            "h2h": 1.1
        },
        "passing_tds": {
            "pass_defense": -0.9,
            "red_zone_usage": 1.1,
            "weapons_grade": 1.2,
            "points_allowed": 0.8,
            "air_yards": 0.7,
            "past_games": 0.85,
            "h2h": 1.1
        },
        "interceptions": {
            "pass_defense": 0.8,
            "pressure_rate": 1.0,
            "td_int:ratio": 0.75,
            "blitz_rate": 0.9,
            "past_games": 0.85,
            "h2h": 1.1
        },
        "rushing_yards": {
            "rush_defense": -0.7,
            "oline_ranking": 0.7,
            "rush_attempts": 0.9,
            "past_games": 0.85,
            "h2h": 1.1
        },
        "rushing_tds": {
            "qb_size": 0.6,
            "rush_attempts": 0.9,
            "rush_defense": -0.5,
            "past_games": 0.85,
            "h2h": 1.1
        },
    },
    "WR": {
        "receiving_yards": {
            "targets": 1.1,
            "yac_avg": 0.85,
            "pass_defense": -0.8,
            "past_games": 0.85,
            "h2h": 1.1
        },
        "receiving_tds": {
            "targets": 1.1,
            "past_games": 0.85,
            "h2h": 1.1
        },
        "rushing_yards": {
            "rush_defense": -0.7,
            "rush_attempts": 0.9,
            "past_games": 0.85,
            "h2h": 1.1
        },
        "receptions": {
            "targets": 1.4,
            "past_games": 0.85,
            "h2h": 1.1,
            "pass_defense": -0.9
        },
    },
    "TE": {
        "receiving_yards": {
            "targets": 1.1,
            "yac_avg": 0.85,
            "pass_defense": -0.8,
            "past_games": 0.85,
            "h2h": 1.1
        },
        "receiving_tds": {
            "targets": 1.1,
            "past_games": 0.85,
            "h2h": 1.1
        },
        "rushing_yards": {
            "rush_defense": -0.7,
            "rush_attempts": 0.9,
            "past_games": 0.85,
            "h2h": 1.1
        },
        "receptions": {
            "targets": 1.4,
            "past_games": 0.85,
            "h2h": 1.1,
            "pass_defense": -0.9
        },
    },
}
