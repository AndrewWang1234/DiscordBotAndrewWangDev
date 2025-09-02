# from src import config
# from src.config import weight_by_pos

import config
from config import defensive_rushing_factors


import nfl_data_py as nfl
import pandas as pd

def load_data(season):
    print("Loading play-by-play data...")
    return nfl.import_pbp_data([season])

def check_team_exists(pbp, team):
    available_offense_teams = pbp['posteam'].dropna().unique()
    available_defense_teams = pbp['defteam'].dropna().unique()
    all_teams = set(available_offense_teams) | set(available_defense_teams)
    return team in all_teams, sorted(all_teams)

def calculate_offensive_stats(pbp, team):
    offense_plays = pbp[pbp['posteam'] == team]
    total_plays = len(offense_plays)

    pass_plays = offense_plays[offense_plays['pass_attempt'] == 1]
    rush_plays = offense_plays[offense_plays['rush_attempt'] == 1]

    pass_rate = len(pass_plays) / total_plays if total_plays > 0 else 0
    rush_rate = len(rush_plays) / total_plays if total_plays > 0 else 0

    return pass_rate, rush_rate

def calculate_offensive_line_metrics(pbp):
    sacks_allowed = pbp[(pbp['posteam'].notnull()) & (pbp['sack'] == 1)]
    sacks_by_team = sacks_allowed.groupby('posteam').size()

    tfl_plays = pbp[(pbp['posteam'].notnull()) & (pbp['rush_attempt'] == 1) & (pbp['yards_gained'] < 0)]
    tfl_by_team = tfl_plays.groupby('posteam').size()

    rush_yards_by_team = pbp[pbp['rush_attempt'] == 1].groupby('posteam')['yards_gained'].sum()

    off_line_df = pd.DataFrame({
        'sacks_allowed': sacks_by_team,
        'tfl_allowed': tfl_by_team,
        'rush_yards': rush_yards_by_team
    }).fillna(0)

    epsilon = 0.0001 #avoid divison by 0
    off_line_df['off_line_metric'] = (off_line_df['sacks_allowed'] + off_line_df['tfl_allowed']) / (off_line_df['rush_yards'] + epsilon)
    off_line_df['off_line_rank'] = off_line_df['off_line_metric'].rank(method='min')

    return off_line_df

def calculate_defensive_stats(pbp):
    team_stats = []
    teams = pbp['defteam'].dropna().unique()

    for team in teams:
        defense_plays = pbp[pbp['defteam'] == team]

        rush_def = defense_plays[defense_plays['rush_attempt'] == 1]
        pass_def = defense_plays[defense_plays['pass_attempt'] == 1]

        rush_yards = rush_def['yards_gained'].sum()
        pass_yards = pass_def['yards_gained'].sum()
        total_yards = rush_yards + pass_yards

        touchdowns = defense_plays[defense_plays['touchdown'] == 1]
        touchdown_points = touchdowns['td_team'].apply(lambda x: 6 if pd.notna(x) else 0).sum()

        field_goals = defense_plays[defense_plays['field_goal_result'] == 'made']
        field_goal_points = 3 * len(field_goals)

        extra_points = defense_plays[defense_plays['extra_point_result'] == 'good']
        extra_point_points = len(extra_points)

        two_pt_plays = defense_plays[defense_plays['two_point_conv_result'] == 'success']
        two_pt_points = 2 * len(two_pt_plays)

        total_points_allowed = touchdown_points + field_goal_points + extra_point_points + two_pt_points

        team_stats.append({
            'team': team,
            'rush_yards_allowed': rush_yards,
            'pass_yards_allowed': pass_yards,
            'total_yards_allowed': total_yards,
            'points_allowed': total_points_allowed
        })

    df = pd.DataFrame(team_stats)
    df['rush_rank'] = df['rush_yards_allowed'].rank(method='min')
    df['pass_rank'] = df['pass_yards_allowed'].rank(method='min')
    df['total_rank'] = df['total_yards_allowed'].rank(method='min')
    df['points_allowed_rank'] = df['points_allowed'].rank(method='min')

    return df

def print_stats(team, season, pass_rate, rush_rate, off_line_df, user_def_row):
    print(f"\n--- {team} Stats for {season} Season ---")
    print(f"Pass Rate:            {pass_rate:.2%}")
    print(f"Rush Rate:            {rush_rate:.2%}")

    if team in off_line_df.index:
        user_off_line_metric = off_line_df.loc[team, 'off_line_metric']
        user_off_line_rank = off_line_df.loc[team, 'off_line_rank']
        print(f"Offensive Line Metric (Sacks+TFL per Rush Yard): {user_off_line_metric:.4f}")
        print(f"Offensive Line Rank:  {int(user_off_line_rank)}")
    else:
        print("Offensive Line Metric: Data not available.")

    print(f"Rush Yards Allowed:   {int(user_def_row['rush_yards_allowed'].values[0])}")
    print(f"Pass Yards Allowed:   {int(user_def_row['pass_yards_allowed'].values[0])}")
    print(f"Points Allowed:       {int(user_def_row['points_allowed'].values[0])}\n")

    print(f"Defensive Rankings for {team}:")
    print(f"Total Yards Allowed Rank: {int(user_def_row['total_rank'].values[0])}")
    print(f"Rush Defense Rank:        {int(user_def_row['rush_rank'].values[0])}")
    print(f"Pass Defense Rank:        {int(user_def_row['pass_rank'].values[0])}")
    print(f"Points Allowed Rank:      {int(user_def_row['points_allowed_rank'].values[0])}")

# Helper to get bias category based on rush defense rank
def get_rush_defense_category(rank):
    if rank <= 5:
        return "top_5"
    elif rank <= 10:
        return "top_10"
    elif rank <= 15:
        return "top_15"
    elif rank <= 20:
        return "top_20"
    elif rank <= 25:
        return "top_25"
    else:
        return "bottom_32"

def adjusted_rush_defense_metric(team, defensive_df):
    team_row = defensive_df[defensive_df['team'] == team].iloc[0]
    rush_yards_allowed = team_row['rush_yards_allowed']
    rush_rank = team_row['rush_rank']

    # Use your bias model dictionary (make sure to import it or define it)
    category = get_rush_defense_category(rush_rank)
    defensive_rush_factor = defensive_rushing_factors[category]

    # Calculate an adjusted metric (example formula, tweak as needed)
    adjusted_metric = defensive_rush_factor - rush_yards_allowed * 0.01

    return adjusted_metric, defensive_rush_factor


def main():
    season = 2024
    user_team = input("Enter NFL team abbreviation (e.g., NE, DAL, ARI): ").upper()

    pbp = load_data(season)

    exists, all_teams = check_team_exists(pbp, user_team)
    if not exists:
        print(f"Team '{user_team}' not found in data. Available teams are:\n{all_teams}")
        return

    pass_rate, rush_rate = calculate_offensive_stats(pbp, user_team)
    off_line_df = calculate_offensive_line_metrics(pbp)
    defensive_df = calculate_defensive_stats(pbp)

    user_def_row = defensive_df[defensive_df['team'] == user_team]
    if user_def_row.empty:
        print(f"\nTeam '{user_team}' not found in defensive stats.")
        return

    print_stats(user_team, season, pass_rate, rush_rate, off_line_df, user_def_row)

    # Call adjusted metric function here
    adjusted_metric, factor_used = adjusted_rush_defense_metric(user_team, defensive_df)
    print(f"Adjusted Rush Defense Metric: {adjusted_metric:.2f}")
    print(f"Bias factor used: {factor_used}")

def rb_nfl_algorithm():
    over, under = 50, 50

    defensive_rush_factor = config.defensive_rushing_factors["top_10"]
    print(defensive_rush_factor)
    # defensive_rush_factor_scale = defensive_rush_factor * (weight_by_pos["RB"]["rush_defense"])
    # defensive_rush_factor_scale = defensive_rush_factor * weight_by_pos["RB"]["rush_defense"]["max_impact"]
    # stat_type = "rushing_yards"  # or "rushing_tds" or whichever stat you’re calculating here
    # defensive_rush_factor_scale = defensive_rush_factor * weight_by_pos["RB"][stat_type]["rush_defense"]["max_impact"]


    # print(defensive_rush_factor_scale)
rb_nfl_algorithm()


if __name__ == "__main__":
    main()


