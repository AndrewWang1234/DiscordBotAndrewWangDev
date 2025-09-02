import nfl_data_py as nfl
import pandas as pd
from collections import defaultdict

def custom_stats(player, statLine):
    #Completion Percentage (QB)
    if statLine == 'completion_percentage':
        player['completion_percentage'] = player.apply(
            lambda row: (row['completions'] / row['attempts'] * 100) if row['attempts'] > 0 else 0,
            axis=1
        )
        display_stat = 'completion_percentage'

    #Yards / Carry (RB)
    elif statLine == 'yards_per_carry':
        player['yards_per_carry'] = player.apply(
            lambda row: (row['rushing_yards'] / row['carries']) if row['carries'] > 0 else 0,
            axis=1
        )
        display_stat = 'yards_per_carry'

    #Yards / Reception (WR)
    elif statLine == 'yards_per_reception':
        player['yards_per_reception'] = player.apply(
            lambda row: (row['receiving_yards'] / row['receptions']) if row['receptions'] > 0 else 0,
            axis=1
        )
        display_stat = 'yards_per_reception'

    #All normal stats
    else:
        display_stat = statLine

    return display_stat



#############    Season Summary Stats           #############
def season_stats(playerName,  statLine, year):
    statLine = statLine.lower().replace(' ', '_')  # Normalize stat line name
    schedule = nfl.import_weekly_data([year])   # gets the season table
    player = schedule[schedule['player_display_name'] == playerName] # Filter by player name

    display_stat = custom_stats(player, statLine)  # Get the display stat name

    #ditionary the stats
    stats_dictionary = defaultdict(list)
    for _, row in player.iterrows():
        team = row['opponent_team']
        stat_value = row[display_stat]
        stats_dictionary[team].append(stat_value)

    return dict(stats_dictionary) #opponent team : [stat values] (duplicates listed by team)


#############    Player vs Team Stats Average           #############
def player_vs_team_average(oppTeam, playerName, statLine):
    statLine = statLine.lower().replace(' ', '_')  # Normalize stat line name
    all_data = nfl.import_weekly_data(list(range(2000, 2025)))
    player_games = all_data[(all_data['player_display_name'] == playerName) & (all_data['opponent_team'] == oppTeam)].copy()

    display_stat = custom_stats(player_games, statLine)  # Get the display stat name

    stats_dictionary = defaultdict(list)
    for _, row in player_games.iterrows():
        team = row['opponent_team']
        stat_value = row[display_stat]
        stats_dictionary[team].append(stat_value)

    avg_stats = {team: sum(values) / len(values) if values else 0 for team, values in stats_dictionary.items()}

    return avg_stats





#############    Player vs Team History           #############
def player_vs_team(oppTeam, playerName, statLine):
    statLine = statLine.lower().replace(' ', '_')  # Normalize stat line name
    all_data = nfl.import_weekly_data(list(range(2000, 2025)))
    player_games = all_data[(all_data['player_display_name'] == playerName) & (all_data['opponent_team'] == oppTeam)]

    display_stat = custom_stats(player_games, statLine)  # Get the display stat name

    stats_dictionary = defaultdict(list)
    for _, row in player_games.iterrows():
        team = row['opponent_team']
        stat_value = row[display_stat]
        stats_dictionary[team].append(stat_value)

    # print(player_games[['player_display_name', 'opponent_team', 'week', 'season']]) uncomment to see games

    return dict(stats_dictionary) #opponent team : [stat values] (duplicates listed by team)





#############    Last 10 Games                  #############
def L10_Average(playerName, statLine, year):
    statLine = statLine.lower().replace(' ', '_')  # Normalize stat name
    schedule = nfl.import_weekly_data([year])
    player = schedule[schedule['player_display_name'] == playerName]

    if player.empty:
        return 0

    player_sorted = player.sort_values(by='week', ascending=False).head(10)

    # Apply custom stats before averaging
    display_stat = custom_stats(player_sorted, statLine)

    if display_stat not in player_sorted.columns:
        return 0

    avg = player_sorted[display_stat].mean()
    return avg




#QB
    #Pass Attempts -> 'attempts'
    #Completions -> 'completions'
    #Passing Yards -> 'passing_yards'
    #Completion Percentage -> 'completion_percentage'
    #Passing TDs -> 'passing_tds'
    #Interceptions -> 'interceptions'
    #Sacks Taken -> 'sacks'
    #Carries -> 'carries'
    #Rushing Yards -> 'rushing_yards'
    #Rushing TDs -> 'rushing_tds'

#RB
    #Carries -> 'carries'
    #Rushing Yards -> 'rushing_yards'
    #Yards / Carry -> 'yards_per_carry'
    #Rushing TDs -> 'rushing_tds'
    #Rushing Fumbles -> 'rushing_fumbles'

    #Targets -> 'targets'
    #Receptions -> 'receptions'
    #Receiving Yards -> 'receiving_yards'
    #Yards / Reception -> 'yards_per_reception'
    #Receiving TDs -> 'receiving_tds'

#WR
    #Targets -> 'targets'
    #Receptions -> 'receptions'
    #Receiving Yards -> 'receiving_yards'
    #Yards / Reception -> 'yards_per_reception'
    #Receiving TDs -> 'receiving_tds'
    #Air Yards -> 'receiving_air_yards'
    #Yards After Catch -> 'receiving_yards_after_catch'

    #Carries -> 'carries'
    #Rushing Yards -> 'rushing_yards'
    #Yards / Carry -> 'yards_per_carry'
    #Rushing TDs -> 'rushing_tds'
    #Rushing Fumbles -> 'rushing_fumbles'

#dictionary return {Den : 96, TeamAgainst : statline}
