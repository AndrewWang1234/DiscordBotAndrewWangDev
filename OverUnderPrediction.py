# OverUnderPrediction.py

import math
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import pandas as pd
import nfl_data_py as nfl

# Local modules
import nfl_api                       # uses config.defensive_* and line/defense calcs
import nfl_player_stats_v2 as nps    # L10, L10_Average, player_vs_team_average, etc.
import config                        # factors_by_position_stat + defensive_*_factors
import traceback
from predictionHelpers import get_red_zone_usage, pointsAllowed, get_player_position, calculate_weapons_grade, get_player_id

# ---------- Utils

def _sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))

def _clip(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))

def _safe_div(a: float, b: float, default: float = 0.0) -> float:
    return a / b if b else default

def _normalize_rank_positive_is_good(rank: float, total_teams: int = 32) -> float:
    """Lower rank is better. Return in [-1, 1] where +1 is best possible rank."""
    return _clip(1 - 2 * ((rank - 1) / (total_teams - 1)), -1.0, 1.0)

def _has_cols(df: Optional[pd.DataFrame], cols: Tuple[str, ...]) -> bool:
    return isinstance(df, pd.DataFrame) and not df.empty and all(c in df.columns for c in cols)


# ---------- Dataclass for output

@dataclass
class PredictionResult:
    player: str
    stat_line: str
    line_value: float
    opponent: str
    season: int
    over_probability: float
    under_probability: float
    decision: str
    contributions: Dict[str, float]   # percentage point deltas
    notes: Dict[str, str]             # context per factor + fallback messages


# ---------- Mapping helpers

def _stat_context(stat_line: str) -> str:
    s = stat_line.lower().replace(" ", "_")
    if any(k in s for k in ["rushing_", "rush_", "yards_per_carry", "carries"]):
        return "rush"
    if any(k in s for k in ["passing_", "pass_", "air_yards", "completions", "attempts", "interceptions"]):
        return "pass"
    if any(k in s for k in ["receiving_", "targets", "receptions", "yards_per_reception"]):
        return "receive"
    return "rush"


def _position_of_player(player_name: str, season: int) -> Optional[str]:
    """
    Infer the player's position from weekly data, then roster. Safe when offline.
    """
    try:
        weekly = nfl.import_weekly_data([season])
        rows = weekly[weekly["player_display_name"] == player_name]
        if not rows.empty:
            pos = rows["position"].dropna()
            if not pos.empty:
                return str(pos.mode().iloc[0]).upper()
    except Exception:
        pass

    try:
        players = nfl.import_players()
        prows = players[players["display_name"] == player_name]
        if not prows.empty:
            pos = prows["position"].dropna()
            if not pos.empty:
                return str(pos.mode().iloc[0]).upper()
    except Exception:
        pass
    return None


def _team_of_player(player_name: str, season: int) -> Optional[str]:
    """
    Infer player's (most frequent) team in a given season from weekly data. Safe when offline.
    """
    try:
        weekly = nfl.import_weekly_data([season])
        rows = weekly[weekly["player_display_name"] == player_name]
        if rows.empty:
            return None
        if "recent_team" in rows.columns:
            return str(rows["recent_team"].mode().iloc[0])
        if "posteam" in rows.columns:
            p = rows["posteam"].dropna()
            if not p.empty:
                return str(p.mode().iloc[0])
    except Exception:
        pass
    return None


# ---------- Factor calculations (robust to missing data)

def _defense_adjustment(
    opp_team: str,
    defensive_df: Optional[pd.DataFrame],
    stat_ctx: str
) -> Tuple[float, str]:
    if defensive_df is None or defensive_df.empty:
        return 0.0, "Skipped: no defensive data (offline or unavailable)"

    row = defensive_df[defensive_df["team"] == opp_team]
    if row.empty:
        return 0.0, f"Skipped: defensive row missing for {opp_team}"

    default_rank = 16

    if stat_ctx == "rush":
        rank = float(row["rush_rank"].values[0])
        dict_used = "defensive_rushing_factors"
        score_map = config.defensive_rushing_factors
    elif stat_ctx in ("pass", "receive"):
        rank = float(row["pass_rank"].values[0])
        dict_used = "defensive_passing_factors"
        score_map = config.defensive_passing_factors
    else:
        rank = float(row["total_rank"].values[0])
        dict_used = "defensive_points_factors"
        score_map = config.defensive_points_factors

    if pd.isna(rank) or not isinstance(rank, (int, float)):
        rank = default_rank

    def _tier(r: float) -> str:
        if r <= 5:  return "top_5"
        if r <= 10: return "top_10"
        if r <= 15: return "top_15"
        if r <= 20: return "top_20"
        if r <= 25: return "top_25"
        return "bottom_32"

    cat = _tier(rank)
    score = score_map[cat]   # 90..40
    norm = -_clip((score - 65.0) / 25.0, -1.0, 1.0)
    note = f"{dict_used}:{cat} (rank={int(rank)}, score={score}, norm={norm:.2f})"
    return norm, note


def _oline_adjustment(player_team: Optional[str], off_line_df: Optional[pd.DataFrame]) -> Tuple[float, str]:
    if not player_team:
        return 0.0, "Skipped: unknown player team for O-line"
    if off_line_df is None or off_line_df.empty or player_team not in off_line_df.index:
        return 0.0, "Skipped: no offensive line metrics (offline or unavailable)"
    rank = float(off_line_df.loc[player_team, "off_line_rank"])
    norm = _normalize_rank_positive_is_good(rank)
    return norm, f"oline_rank={int(rank)}, norm={norm:.2f}"


def _usage_rate_adjustment(
    pbp: Optional[pd.DataFrame],
    player_team: Optional[str],
    stat_ctx: str
) -> Tuple[float, str]:
    if pbp is None or pbp.empty or not player_team or not _has_cols(pbp, ("posteam", "pass_attempt", "rush_attempt")):
        return 0.0, "Skipped: no PBP usage rates (offline or unavailable)"
    pass_rate, rush_rate = nfl_api.calculate_offensive_stats(pbp, player_team)
    if stat_ctx == "rush":
        norm = _clip((rush_rate - 0.45) / 0.25, -1.0, 1.0)
        return norm, f"rush_rate={rush_rate:.2%}, norm={norm:.2f}"
    else:
        norm = _clip((pass_rate - 0.55) / 0.25, -1.0, 1.0)
        return norm, f"pass_rate={pass_rate:.2%}, norm={norm:.2f}"


def _recent_form_adjustment(player_name: str, stat_line: str, season: int, line_value: float) -> Tuple[float, str]:
    try:
        recent_avg = nps.L10_Average(player_name, stat_line, season)
        rel = _clip(_safe_div(recent_avg - line_value, max(line_value, 1e-6)), -1.0, 1.0)
        return rel, f"L10_avg={recent_avg:.2f}, line={line_value}, rel={rel:.2f}"
    except Exception as e:
        return 0.0, f"Skipped: recent form unavailable ({e.__class__.__name__})"


def _vs_team_adjustment(player_name: str, stat_line: str, opp_team: str, line_value: float) -> Tuple[float, str]:
    try:
        vs_avg_dict = nps.player_vs_team_average(opp_team, player_name, stat_line)
        vs_avg = list(vs_avg_dict.values())[0] if vs_avg_dict else 0.0
        rel = _clip(_safe_div(vs_avg - line_value, max(line_value, 1e-6)), -1.0, 1.0)
        return rel, f"vs_{opp_team}_avg={vs_avg:.2f}, line={line_value}, rel={rel:.2f}"
    except Exception as e:
        return 0.0, f"Skipped: vs-team history unavailable ({e.__class__.__name__})"
    
def _yards_per_carry_adjustment(player_name: str, season: int, line_value: float) -> Tuple[float, str]:
    try:
        recent_yards = nps.L10_Average(player_name, "rushing_yards", season)
        recent_carries = nps.L10_Average(player_name, "carries", season)

        if recent_carries == 0: 
            return 0.0, "Skipped: no recent carries"

        recent_ypc = recent_yards / recent_carries
        
        line_ypc = line_value / recent_carries

        rel = _clip(_safe_div(recent_ypc - line_ypc, max(line_ypc, 1e-6)), -1.0, 1.0)
        return rel, (
            f"YPC_recent={recent_ypc:.2f}, "
            f"line_YPC={line_ypc:.2f}, rel={rel:.2f}"
        )
    except Exception as e:
        return 0.0, f"Skipped: yards_per_carry unavailable ({e.__class__.__name__})"

def _carries_adjustment(player_name: str, season: int, stat_line: str, line_value: float) -> Tuple[float,str]:
    try:
        recent_carries = nps.L10_Average(player_name, "carries", season)
        if recent_carries == 0:
            return 0.0, "Skipped: no recent carries"
        
        if "rushing" in stat_line:
            recent_yards = nps.L10_Average(player_name, "rushing_yards", season)
            recent_ypc = _safe_div(recent_yards, recent_carries, 0.0)

            if recent_ypc == 0: 
                return 0.0, "Skipped: no recent rushing efficiency"
            
            implied_carries = line_value / recent_ypc
            rel = _clip(_safe_div(recent_carries - implied_carries, max(implied_carries, 1e-6)), -1.0, 1.0)
            return rel, (
                f"L10_carries={recent_carries:.2f}, implied_line_carries={implied_carries:.2f}, "
                f"YPC={recent_ypc:.2f}, rel={rel:.2f}"
            )
        return 0.0, f"Skipped: carries not relevent for {stat_line}"
    except Exception as e:
        return 0.0, f"Skipped: carries adjustment unavailable ({e.__class__.__name__})"
    
def _red_zone_adjustment(pbp, player_name: str, player_team: Optional[str]) -> Tuple[float, str]:
    try:
        if not player_team:
            return 0.0, "No team provided"
        
        rate = get_red_zone_usage(pbp, player_name, player_team)
        norm = _clip((rate - 0.20) / 0.20, -1.0, 1.0)
        # norm = rate
        note = f"red_zone_rate={rate:.2%}, norm={norm:.2f}"
        return norm, note
    except Exception as e:
        return 0.0, f"Error: {str(e)}"
    
def _points_allowed_adjustment(opp_team: str, points_rank: int) -> tuple[float, str]:
    if points_rank is None:
        return 0.0, "Skipped: no defensive data (offline or unavailable)"
    
    def _teir(r: float) -> str:
        if r <= 5:  return "top_5"
        if r <= 10: return "top_10"
        if r <= 15: return "top_15"
        if r <= 20: return "top_20"
        if r <= 25: return "top_25"
        return "bottom_32"

    cat = _teir(points_rank)
    score = config.defensive_passing_factors[cat]
    norm = -_clip((score - 65.0) / 25.0, -1.0, 1.0)
    note = f"defensive_points_factors:{cat} (rank={int(points_rank)}, score={score}, norm={norm:.2f})"
    return norm, note

def _weapons_grade_adjustment(pbp: pd.DataFrame, player_team: str, player_name: str) -> tuple[float, str]:
    pos = get_player_position(pbp, player_name)
    if pos != "QB":
        return 0.0, "Skipped: not applicable"
    
    try:
        grade = calculate_weapons_grade(pbp, player_team)
        return grade, f"weapons_grade{grade:.2f}"
    except Exception as e:
        return 0.0, f"Skipped: weapons grade calculation failed ({e})"
    
def _air_yards_adjustment(pbp: pd.DataFrame, player_team: str, player_name: str) -> Tuple[float, str]:
    try:

        player_id = get_player_id(player_name, player_team)
        if player_id is None:
            return 0.0, "Skipped: could not resolve player ID"

        qb_passes = pbp[(pbp["passer_player_id"] == player_id)]

        if qb_passes.empty or "air_yards" not in qb_passes.columns:
            return 0.0, "Skipped: no air yards data avaliable"
        
        avg_air_yards = qb_passes["air_yards"].mean()

        norm = _clip((avg_air_yards - 7.0) / 7.0, -1.0, 1.0)

        note = f"avg_air_yards={avg_air_yards:.2f}, norm={norm:.2f}"
        return norm, note

    except Exception as e:
        return 0.0, f"Skipped: air yards calculation failed ({e})"
    
def _pressure_rate_adjustment(pbp: pd.DataFrame, player_team: str, player_name: str) -> tuple[float, str]:
    try:
        player_id = get_player_id(player_name, player_team)
        if player_id is None:
            return 0.0, "Skipped: could not resolve player ID"
        
        qb_passes = pbp[pbp["passer_player_id"] == player_id]

        if qb_passes.empty:
            return 0.0, "Skipped: no pass attempts cound"
        
        pressure_cols = ["sack", "qb_hit", "hurry"]
        available_cols = [c for c in pressure_cols if c in qb_passes.columns]
        if not available_cols:
            return 0.0, "Skipped: no pressure data available"
        
        pressured = qb_passes[available_cols].sum(axis=1)
        pressure_count = (pressured > 0).sum()
        total_dropbacks = len(qb_passes)
        pressure_rate = pressure_count / total_dropbacks if total_dropbacks > 0 else 0.0

        norm = _clip((pressure_rate - 0.25) / 0.25, -1.0, 1.0)
        note = f"pressure_rate={pressure_rate:.2f}, norm={norm:.2f}"

        return norm, note
    except Exception as e:
        return 0.0, "Skipped: pressure rate calculation failed ({e})"
    
def _td_int_ratio_adjustment(pbp, player_team, player_name):
    player_id = get_player_id(player_name, player_team)

    qb_passes = pbp[pbp["passer_player_id"] == player_id]
    tds = qb_passes["touchdown"].sum()
    ints = qb_passes["interception"].sum()
    
    ratio = tds / max(1, ints)
    norm = _clip((ratio - 1.5) / 1.5, -1.0, 1.0)
    note = f"TDs={tds}, INTs={ints}, ration={ratio:.2f}, norm={norm:.2f}"
    return norm, note

def _blitz_rate_adjustment(pbp: pd.DataFrame, player_team: Optional[str]) -> tuple[float, str]:
    """
    Estimate blitz rate for a team as a normalized signal in [-1, 1].
    Uses sacks and QB hits per pass attempt as a proxy since num_rushers isn't available.
    """
    if pbp is None or pbp.empty:
        return 0.0, "Skipped: PBP data unavailable"

    if player_team not in pbp['defteam'].unique():
        return 0.0, f"Skipped: {player_team} not in PBP data"

    # Filter passing plays against the team
    team_pass_plays = pbp[(pbp["defteam"] == player_team) & (pbp.get("pass_attempt", 0) == 1)]
    if team_pass_plays.empty:
        return 0.0, "Skipped: no passing plays against team"

    # Count pressures: sacks + qb_hits (you can add hurries if available)
    sack_count = team_pass_plays["sack"].sum() if "sack" in team_pass_plays.columns else 0
    qb_hit_count = team_pass_plays["qb_hit"].sum() if "qb_hit" in team_pass_plays.columns else 0

    pressure_count = sack_count + qb_hit_count
    total_pass_plays = len(team_pass_plays)

    raw_rate = pressure_count / total_pass_plays if total_pass_plays > 0 else 0.0

    # Normalize: assume average pressure rate ~0.25, scale ±0.25 → [-1,1]
    norm = _clip((raw_rate - 0.25) / 0.25, -1.0, 1.0)

    note = (f"team={player_team}, pressures={pressure_count}, "
            f"pass_plays={total_pass_plays}, raw_rate={raw_rate:.2f}, norm={norm:.2f}")
    return norm, note

def _rush_attempts_adjustment(player_name: str, season: int, stat_line: str, line_value: float) -> tuple[float, str]:
    """
    Estimate the rush attempt signal for a player in [-1,1].
    Compares recent L10 carries vs. the implied line.
    """
    try:
        recent_carries = nps.L10_Average(player_name, "carries", season)
        if recent_carries == 0:
            return 0.0, "Skipped: no recent carries"

        # If the stat line is rushing-related, compute implied attempts
        if "rush" in stat_line:
            recent_yards = nps.L10_Average(player_name, "rushing_yards", season)
            ypc = recent_yards / max(recent_carries, 1e-6)

            implied_attempts = line_value / max(ypc, 1e-6)
            rel = _clip((recent_carries - implied_attempts) / max(implied_attempts, 1e-6), -1.0, 1.0)
            
            note = (f"L10_carries={recent_carries:.2f}, implied_line_attempts={implied_attempts:.2f}, "
                    f"YPC={ypc:.2f}, rel={rel:.2f}")
            return rel, note

        return 0.0, f"Skipped: stat_line not relevant for rushing attempts ({stat_line})"

    except Exception as e:
        return 0.0, f"Skipped: rush attempts unavailable ({e.__class__.__name__})"

def _yac_avg_adjustment(pbp: pd.DataFrame, player_name: str, player_team: str) -> tuple[float, str]:
    """
    Calculate average Yards After Catch (YAC) for a player.
    Returns a normalized signal in [-1, 1] and a descriptive note.
    """
    try:
        if pbp is None or pbp.empty or "yards_after_catch" not in pbp.columns:
            return 0.0, "Skipped: PBP data or 'yards_after_catch' not available"

        # Resolve player ID safely
        player_id = get_player_id(player_name, player_team)
        if not player_id:
            return 0.0, "Skipped: could not resolve player ID"

        # Ensure 'receiver_player_id' exists
        if "receiver_player_id" not in pbp.columns:
            return 0.0, "Skipped: 'receiver_player_id' column not found"

        # Filter all plays for this player
        player_plays = pbp.loc[pbp["receiver_player_id"] == player_id, "yards_after_catch"].dropna()

        if player_plays.empty:
            return 0.0, "Skipped: no receptions found for player"
        
        avg_yac = player_plays.mean()

        # Normalize around a typical YAC (7 yards)
        norm = max(-1.0, min((avg_yac - 7.0) / 7.0, 1.0))

        note = f"avg_yac={avg_yac:.2f}, norm={norm:.2f}"
        return norm, note

    except Exception as e:
        traceback.print_exc()
        return 0.0, f"Skipped: YAC calculation failed ({e.__class__.__name__})"

def _qb_size_adjustment(player_name: str) -> tuple[float, str]:
    try:
        players = nfl.import_players()
        qb = players[players['display_name'] == player_name]
        if qb.empty or qb.iloc[0]['position'] != 'QB':
            return 0.0, "Skipped: not a QB or player not found"

        height = qb.iloc[0]['height']  # inches
        weight = qb.iloc[0]['weight']  # lbs

        size_metric = (height - 72)/6 + (weight - 210)/40
        size_norm = max(-1, min(size_metric / 2, 1))
        note = f"height={height}, weight={weight}, size_norm={size_norm:.2f}"
        return size_norm, note

    except Exception as e:
        return 0.0, f"Skipped: QB size calculation failed ({e.__class__.__name__})"









# ---------- Safe PBP loader with fallback

def _load_pbp_with_fallback(season: int) -> Tuple[Optional[pd.DataFrame], int, str]:

    """
    Try requested season, then season-1. Returns (pbp_df_or_none, actual_season_used, message).
    """
    # 1) Try requested season
    try:
        pbp = nfl_api.load_data(season)
        if isinstance(pbp, pd.DataFrame) and not pbp.empty and "defteam" in pbp.columns:
            return pbp, season, f"Loaded PBP for {season}"
    except Exception as e:
        pass  # fall through

    # 2) Fallback to previous season
    prev = season - 1
    try:
        pbp_prev = nfl_api.load_data(prev)
        if isinstance(pbp_prev, pd.DataFrame) and not pbp_prev.empty and "defteam" in pbp_prev.columns:
            return pbp_prev, prev, f"No PBP for {season}; fell back to {prev}"
    except Exception as e:
        pass

    # 3) No data
    return None, season, f"No PBP available for {season} (or {prev}). Offline or data unavailable."


# ---------- Core predictor

def predict_over_under(
    player_name: str,
    stat_line: str,
    line_value: float,
    opponent_team: str,
    season: int = 2024
) -> PredictionResult:
    """
    Returns a probability-based OVER/UNDER prediction with factor breakdown.
    Robust to offline/no-data situations.
    """
    # Normalize
    stat_line = stat_line.lower().replace(" ", "_")
    stat_ctx = _stat_context(stat_line)

    # Data pulls (robust)
    pbp, pbp_season_used, pbp_msg = _load_pbp_with_fallback(season)

    if pbp is not None:
        try:
            defensive_df = nfl_api.calculate_defensive_stats(pbp)
        except Exception:
            defensive_df = None
        try:
            off_line_df = nfl_api.calculate_offensive_line_metrics(pbp)
        except Exception:
            off_line_df = None
    else:
        defensive_df = None
        off_line_df = None

    player_team = _team_of_player(player_name, season)
    player_pos = (_position_of_player(player_name, season) or "").upper()

    # Choose weights for this player's position+stat
    factors_cfg = None
    if player_pos in config.factors_by_position_stat:
        factors_cfg = config.factors_by_position_stat[player_pos].get(stat_line)
    if not factors_cfg:
        if stat_ctx == "rush":
            factors_cfg = {
                "rush_defense": -1.0,
                "oline_ranking": 0.8,
                "rush_rate": 0.7,
                "yards_per_carry": 0.6,
                "carries": 0.6
            }
        elif stat_ctx in ("pass", "receive"):
            factors_cfg = {
                "pass_defense": -1.0,
                "oline_ranking": 0.7,
                "pass_attempts": 0.6,
                "targets": 0.6
            }
        else:
            factors_cfg = {"defensive_rankings": -0.8}

    # ----- Build normalized signals in [-1,1] -----

    # 1) Defense difficulty (rush/pass)
    def_sig, def_note = _defense_adjustment(opponent_team, defensive_df, stat_ctx)

    # 2) Offensive line strength of player's team
    oline_sig, oline_note = _oline_adjustment(player_team, off_line_df)

    # 3) Team usage tendency (pass_rate / rush_rate) for player's team
    usage_sig, usage_note = _usage_rate_adjustment(pbp, player_team, stat_ctx)

    # 4) Recent form (L10 vs line)
    recent_sig, recent_note = _recent_form_adjustment(player_name, stat_line, season, line_value)

    # 5) Opponent history (player vs opponent)
    vs_sig, vs_note = _vs_team_adjustment(player_name, stat_line, opponent_team, line_value)

    ypc_sig, ypc_note = _yards_per_carry_adjustment(player_name, season, line_value)
    carries_sig, carries_note = _carries_adjustment(player_name, season, stat_line, line_value)
    rz_sig, rz_note = _red_zone_adjustment(pbp, player_name, player_team)
    points_sig, points_note = _points_allowed_adjustment(opponent_team, pointsAllowed(pbp, opponent_team))
    weapons_grade_sig, _weapons_grade_note = _weapons_grade_adjustment(pbp, player_team, player_name)
    air_yards_sig, air_yards_note = _air_yards_adjustment(pbp, player_team, player_name)
    pressure_sig, pressure_note = _pressure_rate_adjustment(pbp, player_team, player_name)
    td_int_sig, td_int_note = _td_int_ratio_adjustment(pbp, player_team, player_name)
    blitz_sig, blitz_note = _blitz_rate_adjustment(pbp, player_team)
    rush_attempts_sig, rush_attempts_note = _rush_attempts_adjustment(player_name, season, stat_line, line_value)
    yac_sig, yac_note = _yac_avg_adjustment(pbp, player_name, player_team)
    qb_size_sig, qb_size_note = _qb_size_adjustment(player_name)

    # Map signals to weight keys
    signals: Dict[str, Tuple[float, str]] = {
        "rush_defense": (def_sig if stat_ctx == "rush" else 0.0, def_note),
        "pass_defense": (def_sig if stat_ctx in ("pass", "receive") else 0.0, def_note),
        "defensive_rankings": (def_sig, def_note),

        "oline_ranking": (oline_sig, oline_note),

        "rush_rate": (usage_sig if stat_ctx == "rush" else 0.0, usage_note),
        "pass_attempts": (usage_sig if stat_ctx in ("pass", "receive") else 0.0, usage_note),
        "targets": (usage_sig if stat_ctx == "receive" else 0.0, usage_note),

        "yards_per_carry": (ypc_sig if stat_ctx == "rush" else 0.0, ypc_note),
        "carries": (carries_sig if stat_ctx == "rush" else 0.0, carries_note),
        "red_zone_usage": (rz_sig, rz_note),
        "points_allowed": (points_sig, points_note),
        "weapons_grade": (weapons_grade_sig if stat_ctx == "pass" else 0.0, _weapons_grade_note),
        "air_yards": (air_yards_sig if stat_ctx == "pass" else 0.0, air_yards_note),
        "pressure_rate": (pressure_sig, pressure_note),
        "td_int:ratio": (td_int_sig if stat_ctx in ("pass", "passing_tds", "interceptions") else 0.0, td_int_note),
        "blitz_rate": (blitz_sig, blitz_note),
        "rush_attempts": (rush_attempts_sig, rush_attempts_note),
        "yac_avg": (yac_sig, yac_note),
        "qb_size": (qb_size_sig, qb_size_note),

        # Always available (best-effort)
        "_recent": (recent_sig, recent_note),
        "_vs_team": (vs_sig, vs_note),
    }

    # ----- Convert to probability shift -----
    SCALE = 0.35  # conservative; tune as needed

    logit = 0.0
    contributions_pp: Dict[str, float] = {}
    notes: Dict[str, str] = {}

    # Add global PBP message (so you see the fallback path taken)
    notes["_pbp_source"] = pbp_msg

    def _add_contribution(key: str, weight: float, sig: float, note: str):
        nonlocal logit
        delta = weight * sig * SCALE
        before = _sigmoid(logit)
        logit += delta
        after = _sigmoid(logit)
        pp = (after - before) * 100.0
        contributions_pp[key] = round(pp, 2)
        notes[key] = note

    for k, w in factors_cfg.items():
        if k in signals:
            sig, note = signals[k]
            _add_contribution(k, float(w), float(sig), note)

    # Always include recent form & vs-team light factors
    r_sig, r_note = signals["_recent"]
    _add_contribution("recent_form", 0.9, r_sig, r_note)

    v_sig, v_note = signals["_vs_team"]
    _add_contribution("vs_team_history", 0.6, v_sig, v_note)

    # Final probabilities
    over_p = _sigmoid(logit)
    under_p = 1 - over_p

    return PredictionResult(
        player=player_name,
        stat_line=stat_line,
        line_value=float(line_value),
        opponent=opponent_team,
        season=season if pbp is None else pbp_season_used,
        over_probability=round(over_p, 4),
        under_probability=round(under_p, 4),
        decision=("OVER" if over_p >= under_p else "UNDER"),
        contributions=contributions_pp,
        notes=notes
    )


# ---------- Example CLI (optional)

if __name__ == "__main__":
    print("Over/Under Predictor")
    player = input("Player name (e.g., Christian McCaffrey): ").strip()
    stat   = input("Stat line (e.g., rushing_yards): ").strip()
    line   = float(input("Line value (e.g., 75.5): ").strip())
    opp    = input("Opponent team (e.g., SEA): ").upper().strip()
    season = 2024

    pred = predict_over_under(player, stat, line, opp, season)
    print("\n--- Prediction ---")
    print(f"{pred.player} — {pred.stat_line} vs {pred.opponent} (line {pred.line_value}) [{pred.season}]")
    print(f"OVER:  {pred.over_probability:.2%}")
    print(f"UNDER: {pred.under_probability:.2%}")
    print(f"Decision: {pred.decision}")
    print("\n--- Contributions (pp) ---")
    for k, v in pred.contributions.items():
        print(f"{k:20s} {v:+6.2f} pp")
    print("\n--- Notes ---")
    for k, v in pred.notes.items():
        print(f"{k:20s} {v}")
