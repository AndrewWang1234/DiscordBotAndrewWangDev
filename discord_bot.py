import os
from dotenv import load_dotenv
load_dotenv()

import discord
from discord.ext import commands
import nfl_data_py as nfl
import pandas as pd
import config  # your config file with weights/factors
from config import defensive_rushing_factors
from nfl_api import (
    load_data,
    check_team_exists,
    calculate_offensive_stats,
    calculate_offensive_line_metrics,
    calculate_defensive_stats,
    adjusted_rush_defense_metric
)
from bot_mehtods import L10, plot_last_10_results, h2h, plot_vs_team_results, h2h_last_10_vs_team
import io
import matplotlib.pyplot as plt
import time
from prediction import predict_stat
from OverUnderPrediction import predict_over_under

intents = discord.Intents.default()
intents.message_content = True  # required for reading messages in new discord.py versions

bot = commands.Bot(command_prefix="!", intents=intents)

# Reuse your existing functions here (load_data, check_team_exists, calculate_offensive_stats, etc.)
# Or import them if you put them in a separate module

@bot.event
async def on_ready():
    print(f'Logged in as {bot.user}')

@bot.command(name="predict_over_under")
async def predict_over_under_command(ctx, *args):
    start_time = time.time()

    try:
        raw_input = " ".join(args)
        params = [param.strip() for param in raw_input.split(';')]

        if len(params) != 4:
            await ctx.send(
                "❌ Invalid format.\n"
                "Use: `Player Name; Stat Line; Line Number; Opponent Team; Season`\n"
                "Example: `Aaron Rodgers; passing yards; 250; MIN; 2024`"
            )
            return
        
        playerName = params[0]
        statLine = params[1]
        lineNumber = float(params[2])
        opponentTeam = params[3]
        season = 2024

        # Call the correct predict_over_under function here
        result = predict_over_under(
            player_name=playerName,
            stat_line=statLine,
            line_value=lineNumber,
            opponent_team=opponentTeam,
            season=season
        )

        # Create an embed message
        embed = discord.Embed(
            title=f"Prediction for {result.player} ({result.stat_line}) vs {result.opponent}",
            description=f"Season: {result.season}",
            color=discord.Color.blue()  # You can customize the color here
        )

        embed.add_field(
            name="⬆️ **Over Probability**",
            value=f"{result.over_probability:.2%}",
            inline=False
        )
        embed.add_field(
            name="⬇️ **Under Probability**: ",
            value=f"{result.under_probability:.2%}",
            inline=False
        )
        embed.add_field(
            name="**Decision**",
            value=result.decision,
            inline=False
        )

        # Optionally, you can add more fields for contributions or notes
        # embed.add_field(name="Contributions (pp)", value=format_contributions(result.contributions), inline=False)

        # Send the embed back to the Discord channel
        await ctx.send(embed=embed)

        elapsed = time.time() - start_time
        #await ctx.send(f"⏱️ Command completed in {elapsed:.2f} seconds.")

    except Exception as e:
        await ctx.send(f"❌ Error: {e}")


@bot.command(name="predict_stat")
async def predict(ctx, playerName: str, statLine: str, opponentTeam: str, playerTeam: str):
    """
    Example usage:
    !predict RB "Bijan Robinson" rushing_yards KC ATL
    !predict QB "Patrick Mahomes" passing_tds BUF KC
    """

    start_time = time.time()

    try:
        result = predict_stat(
            player_name=playerName,
            stat_type=statLine,
            opp_team=opponentTeam,
            player_team=playerTeam
        )

        elapsed = time.time() - start_time
        await ctx.send(
            f"📊 **{result['player']}** ({result['position']}) — {result['stat']} projection:\n"
            f"Base: {result['base_projection']:.2f}\n"
            f"Adjusted: {result['adjusted_projection']:.2f}"
        )
        #await ctx.send(f"⏱️ Command completed in {elapsed:.2f} seconds.")

    except ValueError as e:
        await ctx.send(f"❌ Error: {e}")

@bot.command(name="h2hl10")
async def h2hl10(ctx, *, args):
    start_time = time.time()
    try:
        playerName, statLine, lineNumber, OA, oppTeam = [x.strip() for x in args.split(';')]
        lineNumber = float(lineNumber)
    except Exception:
        await ctx.send(
            "❌ Invalid format.\n"
            "Use: `Player Name; Stat Line; Line Number; Over/Under; Opponent Team`\n"
            "Example: `Aaron Rodgers; passing yards; 250; over; MIN`"
        )
        return

    try:
        stats = h2h_last_10_vs_team(playerName, statLine, lineNumber, OA, oppTeam)
        if stats is None:
            await ctx.send(f"⚠️ No data found for {playerName} against {oppTeam}.")
            return
    except Exception as e:
        await ctx.send(f"⚠️ Error processing data: {e}")
        return

    try:
        image_buf = plot_vs_team_results(
            results=stats['results'],
            line_number=stats['line'],
            oa=stats['OA'],
            player_name=playerName,
            stat_name=statLine,
            opp_team=oppTeam
        )
        file = discord.File(fp=image_buf, filename="h2hl10.png")
        await ctx.send(
            content=(
                f"📊 **Last 10 games for `{playerName}` vs `{oppTeam}` – `{statLine.title()}`**\n"
                f"{playerName} went **{stats['OA']} {stats['line']}** "
                f"{stats['percentage']:.1f}% of these games."
            ),
            file=file
        )
    except Exception as e:
        await ctx.send(f"📉 Could not generate graph: {e}")
    
    elapsed = time.time() - start_time
    #await ctx.send(f"⏱️ Command completed in {elapsed:.2f} seconds.")


@bot.command(name="h2h")
async def vs_team(ctx, *, args):
    start_time = time.time()
    try:
        playerName, statLine, lineNumber, OA, oppTeam = [x.strip() for x in args.split(';')]
        lineNumber = float(lineNumber)
    except Exception:
        await ctx.send(
            "❌ Invalid format.\n"
            "Use: `Player Name; Stat Line; Line Number; Over/Under; Opponent Team`\n"
            "Example: `Aaron Rodgers; passing yards; 250; over; CHI`"
        )
        return

    try:
        all_data = nfl.import_weekly_data(list(range(2005, 2025)))
    except Exception as e:
        await ctx.send(f"❌ Failed to load NFL data: {e}")
        return

    if playerName not in all_data['player_display_name'].unique():
        await ctx.send(f"❌ Player `{playerName}` not found in historical data.")
        return

    valid_stats = all_data.columns.tolist()
    stat_clean = statLine.lower().replace(' ', '_')
    if stat_clean not in valid_stats:
        await ctx.send(f"❌ `{statLine}` is not a valid stat.")
        return

    if OA.lower() not in ['over', 'under']:
        await ctx.send("❌ Please specify `over` or `under` as the fourth argument.")
        return

    # Process the stat history
    try:
        stats = h2h(playerName, statLine, lineNumber, OA, oppTeam)
    except Exception as e:
        await ctx.send(f"⚠️ Error during analysis: `{e}`")
        return

    if not stats:
        await ctx.send(f"ℹ️ No games found for `{playerName}` vs `{oppTeam}`.")
        return

    # Generate plot
    try:
        image_buf = plot_vs_team_results(
            results=stats['results'],
            line_number=stats['line'],
            oa=stats['OA'],
            player_name=playerName,
            stat_name=statLine,
            opp_team=oppTeam
        )

        file = discord.File(fp=image_buf, filename="vsteam.png")
        await ctx.send(
            content=(
                f"📊 **{playerName} vs {oppTeam} – `{statLine.title()}`**\n"
                f"{playerName} went **{stats['OA']} {stats['line']}** "
                f"{stats['percentage']:.1f}% of the {len(stats['results'])} games."
            ),
            file=file
        )

    except Exception as e:
        await ctx.send(f"📉 Could not generate graph: {e}")

    elapsed = time.time() - start_time
    #await ctx.send(f"⏱️ Command completed in {elapsed:.2f} seconds.")


@bot.command(name="last10")
async def last10(ctx, *, args):
    try:
        playerName, statLine, lineNumber, OA = [x.strip() for x in args.split(';')]
        lineNumber = float(lineNumber)
    except Exception:
        await ctx.send(
            "❌ Invalid format.\n"
            "Use: `Player Name; Stat Line; Line Number; Over/Under`\n"
            "Example: `Patrick Mahomes; passing yards; 300; over`"
        )
        return

    # Load 2024 data once here to use for player check
    try:
        schedule = nfl.import_weekly_data([2024])
    except Exception as e:
        await ctx.send(f"❌ Failed to load NFL data: {e}")
        return

    # Player existence check
    if playerName not in schedule['player_display_name'].unique():
        all_players = sorted(schedule['player_display_name'].unique().tolist())
        suggestions = [p for p in all_players if playerName.lower() in p.lower()]
        suggestion_msg = "\nMaybe you meant:\n" + "\n".join(suggestions[:5]) if suggestions else ""
        await ctx.send(f"❌ Player `{playerName}` not found.{suggestion_msg}")
        return

    # Stat check – get valid columns
    valid_stats = schedule.columns.tolist()
    stat_clean = statLine.lower().replace(' ', '_')
    if stat_clean not in valid_stats:
        # Filter useful player stat fields
        exclude = ['player_id', 'player_name', 'player_display_name', 'season', 'week',
                   'team', 'opponent_team', 'headshot_url', 'recent_team', 'season_type']
        allowed_stats = [s for s in valid_stats if s not in exclude and not s.startswith('team')]

        await ctx.send(
            f"❌ `{statLine}` is not a valid stat.\n"
            f"📊 Available stat lines include:\n" +
            ", ".join(sorted(allowed_stats[:30])) + "..."
        )
        return

    # Over/Under validation
    if OA.lower() not in ['over', 'under']:
        await ctx.send("❌ Please specify `over` or `under` as the last argument.")
        return

    # All checks passed – try to run the function
    try:
        stats = L10(playerName, statLine, lineNumber, OA)
    except Exception as e:
        await ctx.send(f"⚠️ Unexpected error while processing: `{e}`")
        return
    try:
        image_buf = plot_last_10_results(
        results=stats['results'],
        line_number=stats['line'],
        oa=stats['OA'],
        player_name=playerName,
        stat_name=statLine
    )


        # Create discord File and send it
        file = discord.File(fp=image_buf, filename="last10.png")
        await ctx.send(
            content=(
                f"📊 **Last 10 games for `{playerName}` – `{statLine.title()}`**\n"
                f"{playerName} went **{stats['OA']} {stats['line']}** "
                f"{stats['percentage']:.1f}% of the last 10 games."
            ),
            file=file
        )

    except Exception as e:
        await ctx.send(f"📉 Could not generate graph: {e}")

    # Format response
    response = f"**Last 10 games for {playerName} - `{statLine.title()}`**\n"
    
    if 'results' in stats:
        for i, (opp, val) in enumerate(stats['results'], 1):
            response += f"Game {i}: vs {opp} - `{statLine}` = {val}\n"

    response += (
        f"\n➡️ `{playerName}` went **{stats['OA']} {stats['line']}** "
        f"{stats['percentage']:.1f}% of the last 10 games."
    )

    await ctx.send(response)



@bot.command(name="nflstats")
async def nfl_stats(ctx, team_abbr: str):
    season = 2024
    team = team_abbr.upper()

    pbp = load_data(season)
    exists, all_teams = check_team_exists(pbp, team)
    if not exists:
        await ctx.send(f"Team '{team}' not found. Available teams: {', '.join(all_teams)}")
        return

    pass_rate, rush_rate = calculate_offensive_stats(pbp, team)
    off_line_df = calculate_offensive_line_metrics(pbp)
    defensive_df = calculate_defensive_stats(pbp)
    user_def_row = defensive_df[defensive_df['team'] == team]
    if user_def_row.empty:
        await ctx.send(f"Defensive stats for team '{team}' not found.")
        return

    # Calculate adjusted rush defense metric similarly if you want
    adjusted_metric, bias_factor = adjusted_rush_defense_metric(team, defensive_df)

    # Format your output nicely
    msg = (
        f"**{team} Stats for {season} Season:**\n"
        f"Pass Rate: {pass_rate:.2%}\n"
        f"Rush Rate: {rush_rate:.2%}\n"
        f"Offensive Line Metric: {off_line_df.loc[team, 'off_line_metric']:.4f}\n"
        f"Rush Yards Allowed: {int(user_def_row['rush_yards_allowed'].values[0])}\n"
        f"Adjusted Rush Defense Metric: {adjusted_metric:.2f}\n"
    )
    await ctx.send(msg)

# Run your bot
bot.run(os.getenv("DISCORD_BOT_TOKEN"))
