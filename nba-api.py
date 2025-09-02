from nba_api.stats.static import players
from nba_api.stats.endpoints import playercareerstats
from nba_api.stats.endpoints import playergamelog
import time

def getPlayerName(name):
    player = players.find_players_by_full_name(name)
    if (players):
        return player[0]['id']
    return None


def getPlayersStats(player_ID):
    time.sleep(1.5)
    careerStats = playercareerstats.PlayerCareerStats(player_id=player_ID)
    dataFrames = careerStats.get_data_frames()[0]

    seasonStats = dataFrames[dataFrames['SEASON_ID'] == "2024-25"]

    if (seasonStats.empty):
        print("Player data could not be found for this season")
        return None
    return seasonStats.iloc[0]


def getUsersStats(stats, statCategory):
    games_Played = stats['GP']
    statCat = statCategory.lower().strip()

    if (statCat == "points"):
        return stats['PTS'] / games_Played

    elif (statCat == "rebounds"):
        return stats['REB'] / games_Played

    elif (statCat == "assists"):
        return stats['AST'] / games_Played

    elif (statCat == "points + rebounds"):
        return (stats['PTS'] + stats['REB']) / games_Played

    elif (statCat == "points + assists"):
        return (stats['PTS'] + stats['AST']) / games_Played

    elif (statCat == "points + rebounds + assists"):
        return (stats['PTS'] + stats['REB'] + stats['AST']) / games_Played

    elif (statCat == "3PM"):
        return stats['FG3M'] / games_Played


def getPast5Games(playerID, statCategory):
    time.sleep(1.5)
    gamelog = playergamelog.PlayerGameLog(player_id=playerID, season='2024-25', season_type_all_star='Regular Season')

    data = gamelog.get_data_frames()[0]
    stateCategory = statCategory.lower().strip()
    pastFive = data.head(5)

    print("Last five games:")
    for _, row in pastFive.iterrows():
        date = row['GAME_DATE']
        matchup = row['MATCHUP']
        if (statCategory == "points"):
            statLine = row['PTS']

        elif (statCategory == "rebounds"):
            statLine = row['REB']

        elif (statCategory == "assists"):
            statLine = row['AST']

        elif (statCategory == "points + rebounds"):
            statLine = row['PTS'] + row['REB']

        elif (statCategory == "points + assists"):
            statLine = row['PTS'] + row['AST']

        elif (statCategory == "rebounds + assists"):
            statLine = row['REB'] + row['AST']

        elif (statCategory == "3pm"):
            statLine = row['FG3M']

        elif (statCategory == "points + rebounds + assists"):
            statLine = row['PTS'] + row['REB'] + row['AST']
        else:
            return
        print(f"{date} {matchup} ---- {statCategory.title()} : {statLine}")


def getPast10Games(playerID, statCategory):
    time.sleep(1.5)
    gamelog = playergamelog.PlayerGameLog(player_id=playerID, season='2024-25', season_type_all_star='Regular Season')

    data = gamelog.get_data_frames()[0]
    stateCategory = statCategory.lower().strip()
    pastFive = data.head(10)
    print("Last ten games:")
    for _, row in pastFive.iterrows():
        date = row['GAME_DATE']
        matchup = row['MATCHUP']
        if (statCategory == "points"):
            statLine = row['PTS']

        elif (statCategory == "rebounds"):
            statLine = row['REB']

        elif (statCategory == "assists"):
            statLine = row['AST']

        elif (statCategory == "points + rebounds"):
            statLine = row['PTS'] + row['REB']

        elif (statCategory == "points + assists"):
            statLine = row['PTS'] + row['AST']

        elif (statCategory == "rebounds + assists"):
            statLine = row['REB'] + row['AST']

        elif (statCategory == "3pm"):
            statLine = row['FG3M']

        elif (statCategory == "points + rebounds + assists"):
            statLine = row['PTS'] + row['REB'] + row['AST']
        else:
            return
        print(f"{date} {matchup} ---- {statCategory.title()} : {statLine}")


def getPast20Games(playerID, statCategory):
    time.sleep(1.5)
    gamelog = playergamelog.PlayerGameLog(player_id=playerID, season='2024-25', season_type_all_star='Regular Season')

    data = gamelog.get_data_frames()[0]
    stateCategory = statCategory.lower().strip()
    pastFive = data.head(20)
    print("Last twenty games:")
    for _, row in pastFive.iterrows():
        date = row['GAME_DATE']
        matchup = row['MATCHUP']
        if (statCategory == "points"):
            statLine = row['PTS']

        elif (statCategory == "rebounds"):
            statLine = row['REB']

        elif (statCategory == "assists"):
            statLine = row['AST']

        elif (statCategory == "points + rebounds"):
            statLine = row['PTS'] + row['REB']

        elif (statCategory == "points + assists"):
            statLine = row['PTS'] + row['AST']

        elif (statCategory == "rebounds + assists"):
            statLine = row['REB'] + row['AST']

        elif (statCategory == "3pm"):
            statLine = row['FG3M']

        elif (statCategory == "points + rebounds + assists"):
            statLine = row['PTS'] + row['REB'] + row['AST']
        else:
            return
        print(f"{date} {matchup} ---- {statCategory.title()} : {statLine}")



playerName = input("Enter player name: ")
playerID = getPlayerName(playerName)
userWantedStats = input("Enter statLine You want: ")
if (playerID):
    stats = getPlayersStats(playerID)
    games_Played_this_Season = stats['GP']
    if (stats is not None):
        # print(f"{playerName}'s stats this Season are : ")
        # print(f"PTS: {stats['PTS'] / games_Played_this_Season:.1f}")
        # print(f"REB: {stats['REB'] / games_Played_this_Season:.1f}")
        # print(f"AST: {stats['AST'] / games_Played_this_Season:.1f}")
        # print(f"MIN: {stats['MIN'] / games_Played_this_Season:.1f}")
        print(f"{playerName} averages {getUsersStats(stats, userWantedStats):.1f} {userWantedStats}")
        getPast10Games(playerID, userWantedStats)
    else:
        print("Stats not available for this player yet.")
else:
    print("Player could not be found")


