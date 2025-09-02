import nfl_data_py as nfl
from config import defensive_rushing_factors, defensive_passing_factors, defensive_points_factors
from nfl_api import calculate_defensive_stats, calculate_offensive_line_metrics, calculate_offensive_stats
import pandas as pd

playerName = "Bijan Robinson" 
statLine = "rushing_yards"  
opponentTeam = "KC"
playerTeam = "ATL"
pbp = nfl.import_pbp_data([2024])


#### BaseLine Stat Calculation ####
def playerAverage(name, stat):
    statLine = stat.lower().replace(' ', '_')
    schedule = nfl.import_weekly_data([2024])
    player = schedule[schedule['player_display_name'] == name] 


    if player.empty: 
        return f"No data found for player {name} in 2024 season."
    
    stat_values = player[statLine].tolist()

    if not stat_values:
        return f"No data found for stat {stat} for player {name} in 2024 season."
    
    avg_stat = sum(stat_values) / len(stat_values)

    return avg_stat

## Name Change ##
def playerNameAbrev(pbp, full_name, player_name_cols):
    def full_name_to_abbrev(name):
        parts = name.split()
        if len(parts) >= 2:
            return parts[0][0] + "." + parts[-1]
        else:
            return name

    abbrev_name = full_name_to_abbrev(full_name)

    # Gather unique player names from all relevant columns
    unique_names = set()
    for col in player_name_cols:
        if col in pbp.columns:
            unique_names.update(pbp[col].dropna().unique())
    
    # Count how many players share this abbreviation
    count = sum(full_name_to_abbrev(name) == abbrev_name for name in unique_names)
    is_ambiguous = count > 1

    if is_ambiguous:
        print(f"Abbreviation ambiguous for {full_name}, use full name '{abbrev_name}' for filtering.")
        return abbrev_name
    else:
        print(f"Using abbreviation '{abbrev_name}' for filtering.")
        return abbrev_name

player_name_columns = ['rusher_player_name', 'passer_player_name']
abbrevName = playerNameAbrev(pbp, playerName, player_name_columns)


############################## Prediction Methods for RB ##############################

# rush_defense: nfl_api -> calculate_defensive_stats(pbp)
def get_defensive_stat_rank(pbp, opponentTeam, statColumn):
    defensive_df = calculate_defensive_stats(pbp)
    opp_def_row = defensive_df[defensive_df["team"] == opponentTeam]
    if opp_def_row.empty:
        raise ValueError(f"Opponent team '{opponentTeam}' not found in defensive stats.")

    def_rank = int(opp_def_row[statColumn].values[0]) #rush_rank can be replaced with pass_rank, points_allowed_rank, total_rank, points_allowed, total_yards_allowed, pass_yards_allowed, rush_yards_allowed
    print("defensive rank:", def_rank)
    return def_rank

# red_zone_usage: custom     
def playerRZUsage(pbp, abbrevName):
    player_plays = pbp[pbp['rusher_player_name'] == abbrevName]  # or passer_player_name depending on stat
    red_zone_plays = player_plays[player_plays['yardline_100'] <= 20]

    red_zone_usage_rate_player = len(red_zone_plays) / len(player_plays) if len(player_plays) > 0 else 0
    # print("Player Red Zone Usage Rate:", red_zone_usage_rate_player)
    return red_zone_usage_rate_player


# player_rating: custom
def calculate_rb_rating(player_name, pbp, player_team, offensive_line_df, defensive_df):
    abbrevName = playerNameAbrev(pbp, player_name, ['rusher_player_name', 'receiver_player_name'])

    player_rushes = pbp[(pbp['rusher_player_name'] == abbrevName) & (pbp['posteam'] == player_team)]
    rush_attempts = len(player_rushes)
    rush_yards = player_rushes['yards_gained'].sum()
    rush_ypc = (rush_yards / rush_attempts) if rush_attempts > 0 else 0
    rush_tds = player_rushes['touchdown'].sum()

    player_targets = pbp[(pbp['receiver_player_name'] == abbrevName) & (pbp['posteam'] == player_team)]
    receptions = len(player_targets[player_targets['complete_pass'] == 1])
    receiving_yards = player_targets[player_targets['complete_pass'] == 1]['yards_gained'].sum()
    receiving_tds = player_targets[player_targets['touchdown'] == 1].shape[0]

    # red_zone_plays = pbp[(pbp['yardline_100'] <= 20) & (pbp['posteam'] == player_team)]
    # red_zone_player_plays = red_zone_plays[
    #     (red_zone_plays['rusher_player_name'] == abbrevName) |
    #     (red_zone_plays['receiver_player_name'] == abbrevName)
    # ]
    # total_off_plays = len(pbp[pbp['posteam'] == player_team])
    red_zone_usage = playerRZUsage(pbp, abbrevName)
    # print(f"Red Zone Usage for {player_name}: {red_zone_usage:.2f}")

    off_line_metric = offensive_line_df.loc[player_team, 'off_line_metric'] if player_team in offensive_line_df.index else 0.5
    off_line_metric = min(max(off_line_metric, 0), 1)

    # opp_def_row = defensive_df[defensive_df['team'] == opponent_team]
    # rush_def_rank = opp_def_row['rush_rank'].values[0] if not opp_def_row.empty else 32

    # -- Normalized Components --
    rush_efficiency_score = min(rush_ypc / 6.0, 1.0) * 30  # max 25
    receiving_yards_score = min(receiving_yards / 80.0, 1.0) * 15  # max 15
    red_zone_score = min(red_zone_usage / 0.5, 1.0) * 10  # max 10

    # TDs capped
    rush_td_score = min(rush_tds, 2) * 10  # max 15
    rec_td_score = min(receiving_tds, 2) * 5  # max 10

    # Receptions with diminishing returns
    if receptions <= 5:
        reception_score = receptions * 3
    else:
        reception_score = 5 * 2 + (receptions - 5) * 0.5  # capped around ~15

    reception_score = min(reception_score, 15) # max 15

    # Final Score
    rating = (
        rush_efficiency_score +
        rush_td_score +
        rec_td_score +
        reception_score +
        receiving_yards_score +
        red_zone_score
    )

    rating = max(0, min(100, rating))

    # print(f"RB Rating for {player_name}: {rating:.2f}")
    # print(f"Rush YPC score: {rush_efficiency_score:.2f}")
    # print(f"Rush TD score: {rush_td_score:.2f}")
    # print(f"Rec TD score: {rec_td_score:.2f}")
    # print(f"Receptions score: {reception_score:.2f}")
    # print(f"Receiving yards score: {receiving_yards_score:.2f}")
    # print(f"Red zone usage score: {red_zone_score:.2f}")
    return rating



defensive_df = calculate_defensive_stats(pbp)
off_line_df = calculate_offensive_line_metrics(pbp)
calculate_rb_rating(playerName, pbp, playerTeam, off_line_df, defensive_df)



# oline_ranking: nfl_api -> calculate_offensive_line_metrics(pbp)
def olineRanking(pbp, playerTeam):
    off_line_df = calculate_offensive_line_metrics(pbp)
    if playerTeam not in off_line_df.index:
        raise ValueError(f"Offensive line data not found for team '{playerTeam}'")

    oline_ranking = int(off_line_df.loc[playerTeam, "off_line_rank"])
    print("Offensive line ranking:", oline_ranking)
    return oline_ranking

# rush_rate: nfl_api -> calculate_offensive_stats(pbp, team)    
def passRushRate(pbp, playerTeam, rp):
    passRate, rushRate = calculate_offensive_stats(pbp, playerTeam)
    if rp.lower() == "rush rate":
        print("Rush Rate:", rushRate)
        return rushRate
    else:
        print("Pass Rate:", passRate)
        return passRate
      
# game_script: custom

def calculate_gamescript(pbp, team):
    # Filter plays where the team is on offense
    team_plays = pbp[pbp['posteam'] == team]

    # Calculate score differential from offense perspective at each play
    # Usually, score differential = (team's score) - (opponent's score)
    score_diff = team_plays['score_differential']

    # Normalize the score differential into a 0-1 scale
    # For example, clip at -20 to +20 and scale
    clipped_diff = score_diff.clip(lower=-20, upper=20)
    normalized_diff = (clipped_diff + 20) / 40  # Maps -20->0, 0->0.5, +20->1

    # Average normalized difference is a rough gamescript proxy
    gamescript = normalized_diff.mean()

    return gamescript

# weather: N/A           

# points_allowed: derived from nfl_api -> calculate_defensive_stats(pbp)
def pointsAllowed(pbp, opponentTeam):
    defensive_df = calculate_defensive_stats(pbp)
    opp_def_row = defensive_df[defensive_df["team"] == opponentTeam]
    if opp_def_row.empty:
        raise ValueError(f"Opponent team '{opponentTeam}' not found in defensive stats.")

    # # To get points allowed (total points given up by the defense)
    # points_allowed = int(opp_def_row["points_allowed"].values[0])
    # print("Points Allowed:", points_allowed)

    # To get points allowed rank (where the defense ranks compared to others)
    points_allowed_rank = int(opp_def_row["points_allowed_rank"].values[0])
    # print("Points Allowed Rank:", points_allowed_rank)
    return points_allowed_rank

# injury_risk: N/A

#Get player ID 
def get_player_id(player_name, team):
    # Import player data (automatically loads data for available seasons)
    df = nfl.import_players()

    # Search for the player based on name and team
    player_data = df[(df['display_name'] == player_name) & (df['latest_team'] == team)]

    if player_data.empty:
        raise ValueError(f"Player {player_name} not found in team {team}")

    # Return the player's ID (or any other identifier)
    player_id = player_data.iloc[0]['gsis_id']  # Using 'gsis_id' as the player ID

    return player_id

def get_player_position(pbp, player_name):
    # Search for player in pbp data
    # This example assumes you have a roster DataFrame or can extract position info from pbp

    # Simplified example: pbp usually doesn't have player position directly, so you might have to load roster
    # Here is a placeholder approach
    roster = roster = nfl.import_seasonal_rosters([2024])  # requires nfl_data_py roster import
    roster = roster[roster['player_name'].str.lower() == player_name.lower()]

    if not roster.empty:
        return roster.iloc[0]['position']
    else:
        raise ValueError(f"Player '{player_name}' not found in roster data.")
    
# def get_player_id(rosters_df, full_name, team_abbr):
#     first_name, last_name = full_name.split(" ", 1)  # splits into first and last name
    
#     filtered = rosters_df[
#        (rosters_df['team'] == team_abbr) &
#         (rosters_df['first_name'].str.lower() == first_name.lower()) &
#         (rosters_df['last_name'].str.lower() == last_name.lower())
#     ]
    
#     if filtered.empty:
#         raise ValueError(f"Player '{full_name}' not found on team '{team_abbr}'.")
    
#     # Return the first matching player's ID
#     return filtered.iloc[0]['player_id'] 

def get_player_carries(pbp, rosters_df, player_name, player_team):
    player_id = get_player_id(rosters_df, player_name, player_team)
    # Filter pbp where rusher_player_id == player_id
    carries_df = pbp[pbp['rusher_player_id'] == player_id]
    carries = carries_df['rush_attempt'].sum()
    return int(carries)

def get_player_yards_per_carry(pbp, player_name, player_team):
    player_id = get_player_id(player_name, player_team)
    # Filter pbp where rusher_player_id == player_id and rush_attempt == 1
    player_plays = pbp[(pbp['rusher_player_id'] == player_id) & (pbp['rush_attempt'] == 1)]
    total_yards = player_plays['rushing_yards'].sum()

    carries = player_plays['rush_attempt'].sum()
    if carries == 0:
        return 0
    
    yards_per_carry = total_yards / carries
    return yards_per_carry


########## red zone usage #################
def get_red_zone_usage(pbp, player_name, player_team):
    # Step 1: Get player ID and position
    player_id = get_player_id(player_name, player_team)
    position = get_player_position(pbp, player_name)

    # Step 2: Load 2024 play-by-play and filter for red zone
    red_zone_plays = pbp[pbp['yardline_100'] <= 20]

    # Step 3: Initialize usage tracking
    rz_plays = 0
    team_plays = 0
    usage_type = ""

    if position in ['WR', 'TE', 'RB']:
        rz_plays = len(red_zone_plays[
            (red_zone_plays['play_type'] == 'pass') &
            (red_zone_plays['receiver_player_id'] == player_id)
        ])
        team_plays = len(red_zone_plays[
            (red_zone_plays['play_type'] == 'pass') &
            (red_zone_plays['posteam'] == player_team)
        ])
        usage_type = "targets"
    elif position in ['RB']:
        rz_plays = len(red_zone_plays[
            (red_zone_plays['play_type'] == 'run') &
            (red_zone_plays['rusher_player_id'] == player_id)
        ])
        team_plays = len(red_zone_plays[
            (red_zone_plays['play_type'] == 'run') &
            (red_zone_plays['posteam'] == player_team)
        ])
        usage_type = "rushes"
    elif position in ['QB']:
        rz_plays = len(red_zone_plays[
            (red_zone_plays['play_type'] == 'pass') &
            (red_zone_plays['passer_player_id'] == player_id)
        ])
        team_plays = len(red_zone_plays[
            (red_zone_plays['posteam'] == player_team)
        ])
        usage_type = "pass attempts"
    else:
        raise ValueError(f"Unsupported position '{position}' for red zone usage.")

    # Step 4: Calculate usage
    usage = rz_plays / team_plays if team_plays > 0 else 0

    # Step 5: Return result
    return usage

def calculate_weapons_grade(pbp: pd.DataFrame, player_team: str) -> float:
    if pbp is None or pbp.empty or not player_team:
        return 0.5
    team_passes = pbp[pbp["posteam"] == player_team]
    if team_passes.empty:
        return 0.5
    
    if "receiver" in team_passes.columns and "yards_gained" in team_passes.columns:
        rec_stats = (
            team_passes.groupby("receiver")["yards_gained"]
            .sum()
            .sort_values(ascending=False)
        )
        top3_yards = rec_stats.head(3).sum()
        total_yards = rec_stats.sum()
        top3_share = top3_yards / max(total_yards, 1)  # fraction of yards from top 3
    else:
        top3_share = 0.5

    # 2) Average air yards
    if "air_yards" in team_passes.columns:
        avg_air_yards = team_passes["air_yards"].mean()
        air_score = min(avg_air_yards / 20.0, 1.0)  # cap at 20 yds
    else:
        air_score = 0.5

    # 3) Red-zone targets
    if "yardline_100" in team_passes.columns:  # distance from endzone
        rz_passes = team_passes[team_passes["yardline_100"] <= 20]
        rz_share = len(rz_passes) / max(len(team_passes), 1)
    else:
        rz_share = 0.5

    # Combine factors (weights can be tuned)
    grade = 0.5 * top3_share + 0.3 * air_score + 0.2 * rz_share
    grade = min(max(grade, 0.0), 1.0)  # clip to [0,1]
    return grade

