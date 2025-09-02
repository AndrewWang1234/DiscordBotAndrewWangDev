import nfl_data_py as nfl
import pandas as pd
from collections import defaultdict
from nfl_player_stats_v2 import custom_stats  # Assuming you have this already
import matplotlib.pyplot as plt
import numpy as np
import io
import seaborn as sns

# def L10(playerName, statLine, lineNumber, OA):
#     statLine = statLine.lower().replace(' ', '_')  # Normalize stat line name
#     schedule = nfl.import_weekly_data([2024])      # get the season table (hardcoded 2024, can be parameterized)
#     player = schedule[schedule['player_display_name'] == playerName]  # Filter by player name

#     player_sorted = player.sort_values(by='week', ascending=True).tail(10)

#     display_stat = custom_stats(player_sorted, statLine)  # Get the correct stat column name

#     # Gather last 10 game stat values with opponent teams
#     results = []
#     for _, row in player_sorted.iterrows():
#         stat_value = row[display_stat]
#         opponent = row['opponent_team']
#         results.append((opponent, stat_value))
    
#     # Display last 10 games results
#     print(f"Last 10 games for {playerName} - Stat: {display_stat}")
#     for i, (opp, val) in enumerate(results, 1):
#         print(f"Game {i}: vs {opp} - {display_stat} = {val}")
    
#     # Calculate over/under percentage
#     if OA.lower() == 'over':
#         count = sum(1 for _, val in results if val > lineNumber)
#     elif OA.lower() == 'under':
#         count = sum(1 for _, val in results if val < lineNumber)
#     else:
#         raise ValueError("OA must be 'over' or 'under'")
    
#     percentage = (count / len(results)) * 100 if results else 0

#     print(f"\n{playerName} went {OA} {lineNumber} {percentage:.1f}% of the last {len(results)} games.")

#     # Return both the detailed results and the percentage
#     return {
#         'results': results,
#         'percentage': percentage,
#         'line': lineNumber,
#         'OA': OA.lower()
#     }




def L10(playerName, statLine, lineNumber, OA):
    statLine = statLine.lower().replace(' ', '_')  # Normalize stat line name

    current_season = 2024
    collected_games = pd.DataFrame()

    # Try loading current season first
    schedule = nfl.import_weekly_data([current_season])
    player_games = schedule[schedule['player_display_name'] == playerName]
    player_games = player_games.sort_values(by='week', ascending=False)

    collected_games = player_games.head(10)

    # If less than 10 games, load previous seasons one by one
    season_to_load = current_season - 1
    while len(collected_games) < 10 and season_to_load >= 2005:  # choose earliest season limit
        schedule_prev = nfl.import_weekly_data([season_to_load])
        player_prev = schedule_prev[schedule_prev['player_display_name'] == playerName]
        player_prev = player_prev.sort_values(by='week', ascending=False)

        # Add games from previous season up to remaining slots needed
        needed = 10 - len(collected_games)
        collected_games = pd.concat([collected_games, player_prev.head(needed)], ignore_index=True)

        season_to_load -= 1

    # Sort combined games by season and week descending to keep most recent first
    collected_games = collected_games.sort_values(by=['season', 'week'], ascending=[False, False]).head(10)

    display_stat = custom_stats(collected_games, statLine)  # Get the correct stat column name

    # Gather last 10 game stat values with opponent teams
    results = []
    for _, row in collected_games.iterrows():
        stat_value = row[display_stat]
        opponent = row['opponent_team']
        results.append((opponent, stat_value))

    results.reverse()  # Reverse to show most recent first
    # Display last 10 games results
    print(f"Last 10 games for {playerName} - Stat: {display_stat}")
    for i, (opp, val) in enumerate(results, 1):
        print(f"Game {i}: vs {opp} - {display_stat} = {val}")

    # Calculate over/under percentage
    if OA.lower() == 'over':
        count = sum(1 for _, val in results if val > lineNumber)
    elif OA.lower() == 'under':
        count = sum(1 for _, val in results if val < lineNumber)
    else:
        raise ValueError("OA must be 'over' or 'under'")

    percentage = (count / len(results)) * 100 if results else 0

    print(f"\n{playerName} went {OA} {lineNumber} {percentage:.1f}% of the last {len(results)} games.")

    # Return both the detailed results and the percentage
    return {
        'results': results,
        'percentage': percentage,
        'line': lineNumber,
        'OA': OA.lower()
    }



# stats = L10("Patrick Mahomes", "passing yards", 100, "over")
# print(stats)

######################## Old Graph Design ########################
# def plot_last_10_results(results, line_number, oa, player_name, stat_name):
#     opponents = [opp for opp, _ in results]
#     values = [val for _, val in results]
#     indices = np.arange(len(results))

#     # Determine bar colors based on Over/Under
#     if oa.lower() == 'over':
#         colors = ['green' if val > line_number else 'red' for val in values]
#     else:
#         colors = ['green' if val < line_number else 'red' for val in values]

#     # Create the plot
#     fig, ax = plt.subplots(figsize=(10, 6))
#     ax.bar(indices, values, color=colors)
#     ax.axhline(line_number, color='black', linestyle='dashed', linewidth=1)
#     ax.text(len(results) - 0.5, line_number + 0.5, f'Line: {line_number}', ha='right')

#     # Labels and styling
#     ax.set_title(f"{player_name} - Last 10 Games ({stat_name.title()})")
#     ax.set_xlabel("Opponent")
#     ax.set_ylabel(stat_name.title())
#     ax.set_xticks(indices)
#     ax.set_xticklabels(opponents, rotation=45, ha='right')
#     plt.tight_layout()

#     # Save the plot to a BytesIO stream
#     buf = io.BytesIO()
#     fig.savefig(buf, format='png')
#     plt.close(fig)
#     buf.seek(0)
#     return buf


def plot_last_10_results(results, line_number, oa, player_name, stat_name):
    # Extract opponent names and performance values
    opponents = [opp for opp, _ in results]
    values = [val for _, val in results]
    indices = np.arange(len(results))

    # Determine bar colors based on Over/Under, using muted colors for a sleek look
    if oa.lower() == 'over':
        colors = ['#4CAF50' if val > line_number else '#F44336' for val in values]  # Green for over, Red for under
    else:
        colors = ['#4CAF50' if val < line_number else '#F44336' for val in values]  # Green for under, Red for over

    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 6))

    # Bar plot with clean color scheme
    bars = ax.bar(indices, values, color=colors, width=0.6, edgecolor='black', linewidth=1.2)

    # Add horizontal line to indicate the threshold
    ax.axhline(line_number, color='black', linestyle='--', linewidth=2)

    # Add the line number text near the dashed line
    ax.text(len(results) - 0.5, line_number + 0.5, f'Line: {line_number}', ha='right', va='bottom', fontsize=12, color='black')

    # Add data labels on top of the bars for exact value display
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + 0.15, f'{height:.1f}', ha='center', va='bottom', fontsize=10)

    # Title and labels with more modern typography
    ax.set_title(f"{player_name} - Last 10 Games ({stat_name.title()})", fontsize=16, weight='bold', color='black')
    ax.set_xlabel("Opponent", fontsize=14, color='black')
    ax.set_ylabel(stat_name.title(), fontsize=14, color='black')

    # Improve x-tick labels (opponent names), smaller and more spaced out
    ax.set_xticks(indices)
    ax.set_xticklabels(opponents, rotation=45, ha='right', fontsize=12, color='black')

    # Add a subtle grid for better readability
    ax.grid(True, axis='y', linestyle='--', alpha=0.4)

    # Adjust layout for better spacing and fitting
    plt.tight_layout()

    # Save the plot to a BytesIO stream
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=300)
    plt.close(fig)
    buf.seek(0)
    return buf



######################## This version is less efficient for shorter carreers, but better for longer careers ########################
# def h2h(playerName, statLine, lineNumber, OA, opp):
#     statLine = statLine.lower().replace(' ', '_')
#     all_data = nfl.import_weekly_data(list(range(2000, 2025)))  # Load all seasons
#     player_games = all_data[
#         (all_data['player_display_name'] == playerName) &
#         (all_data['opponent_team'] == opp)
#     ]

#     if player_games.empty:
#         return None  # No data for that matchup

#     display_stat = custom_stats(player_games, statLine)

#     results = []
#     for _, row in player_games.iterrows():
#         stat_value = row[display_stat]
#         # opponent = row['opponent_team']
#         # results.append((opponent, stat_value))
#         game_label = f"Wk {row['week']} {row['season']}"  # Example: "Wk 5 2021"
#         results.append((game_label, stat_value))

#     # Over/Under calc
#     if OA.lower() == 'over':
#         count = sum(1 for _, val in results if val > lineNumber)
#     elif OA.lower() == 'under':
#         count = sum(1 for _, val in results if val < lineNumber)
#     else:
#         raise ValueError("OA must be 'over' or 'under'")

#     percentage = (count / len(results)) * 100 if results else 0

#     return {
#         'results': results,
#         'percentage': percentage,
#         'line': lineNumber,
#         'OA': OA.lower()
#     }

def get_player_career_span(playerName, start_year=2000, end_year=2024, chunk_size=5):
    # Search in chunks of years, moving backward, until no more data for player found
    seasons_found = []
    for year_start in range(end_year - chunk_size + 1, start_year - 1, -chunk_size):
        year_end = year_start + chunk_size - 1
        if year_end > end_year:
            year_end = end_year
        
        years = list(range(year_start, year_end + 1))
        data_chunk = nfl.import_weekly_data(years)
        player_data = data_chunk[data_chunk['player_display_name'] == playerName]
        
        if not player_data.empty:
            seasons_found.extend(player_data['season'].unique().tolist())
        else:
            # If no data in this chunk, assume no earlier data — can stop early
            if seasons_found:
                break

    if seasons_found:
        return min(seasons_found), max(seasons_found)
    else:
        return None, None

def h2h(playerName, statLine, lineNumber, OA, opp):
    statLine = statLine.lower().replace(' ', '_')
    
    # Get player's career span
    min_season, max_season = get_player_career_span(playerName)
    
    if min_season is None or max_season is None:
        return None  # Player not found at all
    
    # Load all seasons in career span
    all_data = nfl.import_weekly_data(list(range(min_season, max_season + 1)))
    
    player_games = all_data[
        (all_data['player_display_name'] == playerName) &
        (all_data['opponent_team'] == opp)
    ]

    if player_games.empty:
        return None  # No data for that matchup

    display_stat = custom_stats(player_games, statLine)

    results = []
    for _, row in player_games.iterrows():
        stat_value = row[display_stat]
        game_label = f"Wk {row['week']} {row['season']}"  # e.g., "Wk 5 2021"
        results.append((game_label, stat_value))

    # Over/Under calculation
    if OA.lower() == 'over':
        count = sum(1 for _, val in results if val > lineNumber)
    elif OA.lower() == 'under':
        count = sum(1 for _, val in results if val < lineNumber)
    else:
        raise ValueError("OA must be 'over' or 'under'")

    percentage = (count / len(results)) * 100 if results else 0

    return {
        'results': results,
        'percentage': percentage,
        'line': lineNumber,
        'OA': OA.lower()
    }

########################## Old Graph Design ##########################
# def plot_vs_team_results(results, line_number, oa, player_name, stat_name, opp_team):
#     labels = [label for label, _ in results]
#     values = [val for _, val in results]
#     indices = np.arange(len(results))

#     # Determine bar colors
#     if oa.lower() == 'over':
#         colors = ['green' if val > line_number else 'red' for val in values]
#     else:
#         colors = ['green' if val < line_number else 'red' for val in values]

#     # Create the plot
#     fig, ax = plt.subplots(figsize=(12, 6))
#     ax.bar(indices, values, color=colors)
#     ax.axhline(line_number, color='black', linestyle='dashed', linewidth=1)
#     ax.text(len(results) - 1, line_number + 0.5, f'Line: {line_number}', ha='right')

#     # Labels and styling
#     ax.set_title(f"{player_name} vs {opp_team} – {stat_name.title()} by Game")
#     ax.set_xlabel("Game Date (Week & Season)")
#     ax.set_ylabel(stat_name.title())
#     ax.set_xticks(indices)
#     ax.set_xticklabels(labels, rotation=45, ha='right')
#     plt.tight_layout()

#     # Save to buffer
#     buf = io.BytesIO()
#     fig.savefig(buf, format='png')
#     plt.close(fig)
#     buf.seek(0)
#     return buf

def h2h_last_10_vs_team(playerName, statLine, lineNumber, OA, opp):
    statLine = statLine.lower().replace(' ', '_')
    
    # Get player's career span
    min_season, max_season = get_player_career_span(playerName)
    if min_season is None or max_season is None:
        return None  # Player not found

    # Load all seasons in career span
    all_data = nfl.import_weekly_data(list(range(min_season, max_season + 1)))

    # Filter player's games vs opponent
    player_games = all_data[
        (all_data['player_display_name'] == playerName) & 
        (all_data['opponent_team'] == opp)
    ]

    if player_games.empty:
        return None

    # Sort by season and week ascending (oldest to newest)
    player_games = player_games.sort_values(by=['season', 'week'], ascending=True)

    # Keep only the last 10 games vs the opponent
    last_10_games = player_games.tail(10)

    display_stat = custom_stats(last_10_games, statLine)

    results = []
    for _, row in last_10_games.iterrows():
        stat_value = row[display_stat]
        game_label = f"Wk {row['week']} {row['season']}"  # e.g. "Wk 5 2021"
        results.append((game_label, stat_value))

    if OA.lower() == 'over':
        count = sum(1 for _, val in results if val > lineNumber)
    elif OA.lower() == 'under':
        count = sum(1 for _, val in results if val < lineNumber)
    else:
        raise ValueError("OA must be 'over' or 'under'")

    percentage = (count / len(results)) * 100 if results else 0

    return {
        'results': results,  # Reverse to show most recent first
        'percentage': percentage,
        'line': lineNumber,
        'OA': OA.lower()
    }

def plot_vs_team_results(results, line_number, oa, player_name, stat_name, opp_team):
    labels = [label for label, _ in results]
    values = [val for _, val in results]
    indices = np.arange(len(results))

    # Determine bar colors based on Over/Under, using a muted color palette
    if oa.lower() == 'over':
        colors = ['#4CAF50' if val > line_number else '#F44336' for val in values]  # Green for over, Red for under
    else:
        colors = ['#4CAF50' if val < line_number else '#F44336' for val in values]  # Green for under, Red for over

    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 6))

    # Bar plot with clean color scheme
    bars = ax.bar(indices, values, color=colors, width=0.6, edgecolor='black', linewidth=1.2)

    # Add horizontal line to indicate the threshold
    ax.axhline(line_number, color='black', linestyle='--', linewidth=2)

    # Add the line number text near the dashed line
    ax.text(len(results) - 1, line_number + 0.5, f'Line: {line_number}', ha='right', va='bottom', fontsize=12, color='black')

    # Add data labels on top of the bars for exact value display
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + 0.15, f'{height:.1f}', ha='center', va='bottom', fontsize=10)

    # Title and labels with modern typography
    ax.set_title(f"{player_name} vs {opp_team} – {stat_name.title()} by Game", fontsize=16, weight='bold', color='black')
    ax.set_xlabel("Game Date (Week & Season)", fontsize=14, color='black')
    ax.set_ylabel(stat_name.title(), fontsize=14, color='black')

    # Improve x-tick labels (game dates), rotate for readability
    ax.set_xticks(indices)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=12, color='black')

    # Add a subtle grid for better readability
    ax.grid(True, axis='y', linestyle='--', alpha=0.4)

    # Adjust layout for better spacing and fitting
    plt.tight_layout()

    # Save the plot to a BytesIO stream
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=300)
    plt.close(fig)
    buf.seek(0)
    return buf
