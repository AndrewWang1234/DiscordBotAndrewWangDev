import nfl_data_py as nfl
from config import defensive_rushing_factors, defensive_passing_factors, defensive_points_factors, factors_by_position_stat
from predictionHelpers import playerAverage, calculate_gamescript, get_defensive_stat_rank, olineRanking, passRushRate, pointsAllowed, playerRZUsage, playerNameAbrev, calculate_rb_rating, get_player_position, get_player_carries, get_player_yards_per_carry
from nfl_api import calculate_defensive_stats, calculate_offensive_line_metrics

### Prediction Code #####
def get_defense_score(rank, factor_type="rushing"):
    # Maps numeric rank to defensive factor score using your config
    # factor_type can be "rushing", "passing", or "points"
    factors_map = {
        "rushing": defensive_rushing_factors,
        "passing": defensive_passing_factors,
        "points": defensive_points_factors
    }
    factors = factors_map.get(factor_type, defensive_rushing_factors)
    
    if rank <= 5:
        return factors["top_5"]
    elif rank <= 10:
        return factors["top_10"]
    elif rank <= 15:
        return factors["top_15"]
    elif rank <= 20:
        return factors["top_20"]
    elif rank <= 25:
        return factors["top_25"]
    else:
        return factors["bottom_32"]

def normalize(value, min_val, max_val):
    # Normalize a numeric value to [0,1]
    return max(0, min(1, (value - min_val) / (max_val - min_val)))

def bias_adjust_stat(stat_type, base_stat, pos, factor_values):
    if pos not in factors_by_position_stat:
        print(f"No factors for position {pos}")
        return base_stat

    if stat_type not in factors_by_position_stat[pos]:
        print(f"No factor weights for {pos} stat '{stat_type}'")
        return base_stat

    # Pull the configured factor weights directly
    weights = factors_by_position_stat[pos][stat_type]
    adjustment = 0.0
    contributions = {}

    for factor_name, weight in weights.items():
        if factor_name not in factor_values:
            continue

        raw_val = factor_values[factor_name]

        # Convert defense ranks into normalized values
        if factor_name == "rush_defense":
            raw_val = get_defense_score(raw_val, "rushing")
            norm_val = normalize(raw_val, 40, 90)
        elif factor_name == "pass_defense":
            raw_val = get_defense_score(raw_val, "passing")
            norm_val = normalize(raw_val, 40, 90)
        elif factor_name == "points_allowed":
            raw_val = get_defense_score(raw_val, "points")
            norm_val = normalize(raw_val, 40, 90)
        elif factor_name in ["oline_ranking"]:
            norm_val = 1 - normalize(raw_val, 1, 32)  # lower rank = better
        elif factor_name == "player_rating":
            norm_val = normalize(raw_val, 50, 100)
        else:
            # Already assumed to be 0–1 scale
            norm_val = max(0, min(1, raw_val))

        # Multiply directly by the weight (positive or negative impact)
        adj = weight * norm_val
        adjustment += adj
        contributions[factor_name] = adj

    adjusted_stat = base_stat + adjustment
    print(f"Base stat: {base_stat:.2f}")
    print(f"Total adjustment: {adjustment:.2f}")
    print("Factor contributions:")
    for factor, contrib in contributions.items():
        sign = "+" if contrib >= 0 else "-"
        print(f"  {factor}: {sign}{abs(contrib):.3f}")
    print(f"Adjusted stat: {adjusted_stat:.2f}")
    
    return max(0, adjusted_stat)



#### Needed Information ####
# playerName = "Bijan Robinson"
# statLine = "rushing_yards"
# opp = "KC"
# player_team = "ATL"

pbp = nfl.import_pbp_data([2024])
defensive_df = calculate_defensive_stats(pbp)
off_line_df = calculate_offensive_line_metrics(pbp)
rosters = nfl.import_weekly_rosters([2024])

def predict_stat(player_name, stat_type, opp_team, player_team):
    pos = get_player_position(pbp, player_name)
    pos = pos.upper()

    if pos not in factors_by_position_stat or stat_type not in factors_by_position_stat[pos]:
        raise ValueError(f"No config for position {pos} and stat {stat_type}")
    
    name_abbr = playerNameAbrev(pbp, player_name, ['rusher_player_name', 'passer_player_name'])

    needed_factors = factors_by_position_stat[pos][stat_type].keys()
    factor_values = {}

    for factor in needed_factors:
        if factor == "rush_defense":
            factor_values[factor] = get_defensive_stat_rank(pbp, opp_team, "rush_rank")
        elif factor == "pass_defense":
            factor_values[factor] = get_defensive_stat_rank(pbp, opp_team, "pass_rank")
        elif factor == "points_allowed":
            factor_values[factor] = pointsAllowed(pbp, opp_team)
        elif factor == "oline_ranking":
            factor_values[factor] = olineRanking(pbp, player_team)
        elif factor == "rush_rate":
            factor_values[factor] = passRushRate(pbp, player_team, "rush rate")
        elif factor == "pass_attempts":
            factor_values[factor] = passRushRate(pbp, player_team, "pass attempts rate")
        elif factor == "game_script":
            factor_values[factor] = calculate_gamescript(pbp, player_team)
        elif factor == "red_zone_usage":
            factor_values[factor] = playerRZUsage(pbp, name_abbr)
        elif factor == "player_rating":
            factor_values[factor] = calculate_rb_rating(player_name, pbp, player_team, off_line_df, defensive_df)
        elif factor == "carries":
            factor_values[factor] = get_player_carries(pbp, rosters, player_name, player_team)
        elif factor == "yards_per_carry":
            factor_values[factor] = get_player_yards_per_carry(pbp, rosters, player_name, player_team)
        # QB-specific ones you may have to add here:
        elif factor == "weapons_grade":
            factor_values[factor] = 0.8  # placeholder until you implement
        elif factor == "air_yards":
            factor_values[factor] = 0.7  # placeholder
        elif factor == "pressure_rate":
            factor_values[factor] = 0.6  # placeholder
        elif factor == "turnover_prone":
            factor_values[factor] = 0.4  # placeholder
        elif factor == "blitz_rate":
            factor_values[factor] = 0.5  # placeholder
        elif factor == "wind_conditions":
            factor_values[factor] = 0.5  # placeholder
        elif factor == "qb_mobility":
            factor_values[factor] = 0.7  # placeholder
        elif factor == "design_rush":
            factor_values[factor] = 0.6  # placeholder
        elif factor == "red_zone_mobility":
            factor_values[factor] = 0.6  # placeholder
        elif factor == "qb_size":
            factor_values[factor] = 0.75  # placeholder

    base_proj = playerAverage(player_name, stat_type)
    adjusted_proj = bias_adjust_stat(stat_type, base_proj, pos, factor_values)

    print(f"Original projection for {player_name} ({stat_type}): {base_proj:.2f}")
    print(f"Adjusted projection after factors: {adjusted_proj:.2f}")

    return {
        "player": player_name,
        "position": pos,
        "stat": stat_type,
        "base_projection": base_proj,
        "adjusted_projection": adjusted_proj,
    }    