#property strict
#property version   "1.20"
#property description "Long-only XAUUSD regime strategy approximating the promoted MA native state-quality exit winner"

#include <Trade/Trade.mqh>

input ulong   InpMagicNumber                        = 260605;
input double  InpRiskPercentPerTrade                = 1.0;
input double  InpMaxOpenRiskPercent                 = 10.0;
input int     InpMaxConcurrentTrades                = 5;
input int     InpCooldownBars                       = 5;
input bool    InpContinuationRequireFreshHighAfterLoss = true;
input int     InpRecentLosingSameLevelLookbackBars  = 50;
input double  InpRecentLosingSameLevelDistanceATR   = 0.40;
input int     InpBlockAfterRecentLosingSameLevelCount = 1;
input int     InpHighestHighLookback                = 100;
input int     InpEMAPeriod                          = 200;
input int     InpEMAFastPeriod                      = 20;
input int     InpEMAMidPeriod                       = 50;
input int     InpATRPeriod                          = 14;
input int     InpEfficiencyLookback                 = 20;
input int     InpEMASlopeLookback                   = 10;
input double  InpContinuationMinEfficiency          = 0.32;
input double  InpStopDistancePrice                  = 10.0;   // 1R on XAUUSD
input int     InpTimedTrailActivationBars           = 50;
input double  InpTimedTrailR                        = 0.7;
input double  InpBreakEvenTriggerR                  = 2.0;
input double  InpTrendTrailAfter2RR                 = 2.5;
input double  InpContinuationTakeProfitR            = 4.0;
input double  InpContinuationBreakEvenOffsetR       = 0.25;
input int     InpRegimeDeathFailedAddsThreshold     = 1;
input int     InpRegimeDeathNoFreshHighBars         = 10;
input double  InpRegimeDeathMaxEfficiency           = 0.32;
input double  InpRegimeRearmMinEfficiency           = 0.40;
input int     InpSurvivorFailedHighsThreshold       = 3;
input double  InpSurvivorMinProfitR                 = 0.5;
input double  InpSurvivorTightenOffsetR             = 1.0;
input bool    InpSurvivorRequireLowerHighBreak      = true;
input bool    InpSurvivorRequireFailedBreakout      = true;
input double  InpSurvivorMinEMA20SlopeDecay         = 0.15;
input bool    InpSurvivorMomentumRequireRegimeDead  = true;
input int     InpStateZScoreLookback                = 100;
input double  InpStateTwoScorePenaltyWeight         = 0.75;
input double  InpStateHighMinScore                  = 0.15;
input double  InpStateMediumMinScore                = -0.05;
input double  InpStateHighRiskMultiplier            = 1.50;
input double  InpStateMediumRiskMultiplier          = 1.00;
input double  InpStateLowRiskMultiplier             = 0.75;
input int     InpLowNoNewHighExitBars               = 3;
input double  InpLowNoNewHighMinMFER                = 0.50;
input int     InpLowNoNewHighMinBarsOpen            = 6;
input double  InpMediumRetraceMinMFER               = 1.00;
input double  InpMediumRetraceFrac                  = 0.33;
input int     InpMediumRetraceMinBarsOpen           = 8;
input int     InpMacdFast                           = 12;
input int     InpMacdSlow                           = 26;
input int     InpMacdSignal                         = 9;
input int     InpMaxScanBars                        = 5000;
input ulong   InpDeviationPoints                    = 30;

CTrade trade;
int g_ema_handle = INVALID_HANDLE;
int g_ema_fast_handle = INVALID_HANDLE;
int g_ema_mid_handle = INVALID_HANDLE;
int g_atr_handle = INVALID_HANDLE;
int g_osma_handle = INVALID_HANDLE;
datetime g_last_bar_time = 0;
datetime g_state_regime_start = 0;
bool g_regime_continuation_dead = false;
string g_regime_dead_reason = "";
int g_regime_rearm_count = 0;
datetime g_regime_last_rearm_bar = 0;
string g_regime_last_rearm_reason = "";

enum TradeKind
{
   TRADE_UNKNOWN = 0,
   TRADE_FIRST   = 1,
   TRADE_CONT    = 2
};

enum StateTier
{
   STATE_TIER_UNKNOWN = 0,
   STATE_TIER_LOW     = 1,
   STATE_TIER_MEDIUM  = 2,
   STATE_TIER_HIGH    = 3
};

struct StateQualitySnapshot
{
   double    score;
   double    continuation_score;
   double    fragility_score;
   StateTier tier;
   double    risk_multiplier;
};

struct RegimeStats
{
   datetime regime_start;
   datetime latest_entry_time;
   int      entry_count;
   int      failed_continuation_adds;
};

struct RegimePositionSnapshot
{
   ulong    position_id;
   TradeKind kind;
   datetime entry_time;
   datetime exit_time;
   double   entry_price;
   double   realized_pnl;
   bool     has_entry;
};

struct SurvivorStructureState
{
   int    failed_high_attempts;
   bool   lower_high_confirmed;
   bool   lower_low_confirmed;
   bool   lower_high_then_swing_low_break;
   bool   failed_breakout_flag;
   double last_confirmed_swing_low;
   double last_confirmed_swing_high;
};

string KindTag(const TradeKind kind)
{
   if(kind == TRADE_FIRST)
      return "XAUF";
   if(kind == TRADE_CONT)
      return "XAUC";
   return "XAU?";
}

string TierName(const StateTier tier)
{
   if(tier == STATE_TIER_HIGH)
      return "high";
   if(tier == STATE_TIER_MEDIUM)
      return "medium";
   if(tier == STATE_TIER_LOW)
      return "low";
   return "unknown";
}

TradeKind ParseTradeKind(const string comment)
{
   if(StringFind(comment, "XAUF|") == 0)
      return TRADE_FIRST;
   if(StringFind(comment, "XAUC|") == 0)
      return TRADE_CONT;
   return TRADE_UNKNOWN;
}

datetime ParseRegimeStart(const string comment)
{
   string parts[];
   const int count = StringSplit(comment, '|', parts);
   if(count < 2)
      return 0;
   if(StringLen(parts[1]) == 0)
      return 0;
   return (datetime)StringToInteger(parts[1]);
}

StateTier ParseStateTier(const string comment)
{
   string parts[];
   const int count = StringSplit(comment, '|', parts);
   if(count < 3)
      return STATE_TIER_UNKNOWN;

   string tier = parts[2];
   StringToLower(tier);
   if(tier == "high")
      return STATE_TIER_HIGH;
   if(tier == "medium")
      return STATE_TIER_MEDIUM;
   if(tier == "low")
      return STATE_TIER_LOW;
   return STATE_TIER_UNKNOWN;
}

bool IsHedgingAccount()
{
   return ((ENUM_ACCOUNT_MARGIN_MODE)AccountInfoInteger(ACCOUNT_MARGIN_MODE) == ACCOUNT_MARGIN_MODE_RETAIL_HEDGING);
}

double NormalizePrice(const double price)
{
   return NormalizeDouble(price, (int)SymbolInfoInteger(_Symbol, SYMBOL_DIGITS));
}

double ClampValue(const double value, const double low, const double high)
{
   return MathMax(low, MathMin(high, value));
}

double NormalizeSymmetric(const double value, const double cap)
{
   if(cap <= 0.0)
      return 0.0;
   return ClampValue(value / cap, -1.0, 1.0);
}

double NormalizeWindow01ToSymmetric(const double value, const double low, const double high)
{
   if(high <= low)
      return 0.0;
   const double pct = ClampValue((value - low) / (high - low), 0.0, 1.0);
   return pct * 2.0 - 1.0;
}

double StopLevelDistance()
{
   return (double)SymbolInfoInteger(_Symbol, SYMBOL_TRADE_STOPS_LEVEL) * SymbolInfoDouble(_Symbol, SYMBOL_POINT);
}

bool IsNewBar()
{
   const datetime current_bar = iTime(_Symbol, _Period, 0);
   if(current_bar == 0)
      return false;

   if(g_last_bar_time == 0)
   {
      g_last_bar_time = current_bar;
      return false;
   }

   if(current_bar != g_last_bar_time)
   {
      g_last_bar_time = current_bar;
      return true;
   }

   return false;
}

bool CopyIndicatorBuffer(const int handle, const int count, double &buffer[])
{
   ArrayResize(buffer, count);
   if(CopyBuffer(handle, 0, 0, count, buffer) != count)
      return false;
   return true;
}

double BufferValueAtShift(const double &buffer[], const int copied_count, const int shift)
{
   const int index = copied_count - 1 - shift;
   if(index < 0 || index >= copied_count)
      return EMPTY_VALUE;
   return buffer[index];
}

bool IsConfirmedSwingHighAtShift(const int shift)
{
   if(shift < 2)
      return false;
   const double value = iHigh(_Symbol, _Period, shift);
   return (value > iHigh(_Symbol, _Period, shift - 1)
      && value > iHigh(_Symbol, _Period, shift - 2)
      && value > iHigh(_Symbol, _Period, shift + 1)
      && value > iHigh(_Symbol, _Period, shift + 2));
}

bool IsConfirmedSwingLowAtShift(const int shift)
{
   if(shift < 2)
      return false;
   const double value = iLow(_Symbol, _Period, shift);
   return (value < iLow(_Symbol, _Period, shift - 1)
      && value < iLow(_Symbol, _Period, shift - 2)
      && value < iLow(_Symbol, _Period, shift + 1)
      && value < iLow(_Symbol, _Period, shift + 2));
}

double EMAFastSlopeAtShift(const int shift)
{
   const int need = shift + InpEMASlopeLookback + 2;
   double ema_fast_vals[];
   if(!CopyIndicatorBuffer(g_ema_fast_handle, need, ema_fast_vals))
      return 0.0;

   const double current = BufferValueAtShift(ema_fast_vals, need, shift);
   const double prior   = BufferValueAtShift(ema_fast_vals, need, shift + InpEMASlopeLookback);
   if(current == EMPTY_VALUE || prior == EMPTY_VALUE)
      return 0.0;
   return (current - prior) / (double)InpEMASlopeLookback;
}

double EMAMidSlopeAtShift(const int shift)
{
   const int need = shift + InpEMASlopeLookback + 2;
   double ema_mid_vals[];
   if(!CopyIndicatorBuffer(g_ema_mid_handle, need, ema_mid_vals))
      return 0.0;

   const double current = BufferValueAtShift(ema_mid_vals, need, shift);
   const double prior   = BufferValueAtShift(ema_mid_vals, need, shift + InpEMASlopeLookback);
   if(current == EMPTY_VALUE || prior == EMPTY_VALUE)
      return 0.0;
   return (current - prior) / (double)InpEMASlopeLookback;
}

double EMAValueAtShift(const int handle, const int shift)
{
   const int need = shift + 2;
   double values[];
   if(!CopyIndicatorBuffer(handle, need, values))
      return 0.0;
   const double value = BufferValueAtShift(values, need, shift);
   return (value == EMPTY_VALUE ? 0.0 : value);
}

double EMA200ValueAtShift(const int shift)
{
   return EMAValueAtShift(g_ema_handle, shift);
}

double EMAFastValueAtShift(const int shift)
{
   return EMAValueAtShift(g_ema_fast_handle, shift);
}

bool CurrentBullRegime(datetime &regime_start)
{
   regime_start = 0;

   const int bars_total = Bars(_Symbol, _Period);
   const int need = MathMin(bars_total, InpMaxScanBars + 5);
   if(need < InpEMAPeriod + 5)
      return false;

   double ema[];
   if(!CopyIndicatorBuffer(g_ema_handle, need, ema))
      return false;

   const double close1 = iClose(_Symbol, _Period, 1);
   const double ema1   = BufferValueAtShift(ema, need, 1);
   if(close1 <= ema1)
      return false;

   for(int shift = 1; shift < need - 1; ++shift)
   {
      const double c0 = iClose(_Symbol, _Period, shift);
      const double c1 = iClose(_Symbol, _Period, shift + 1);
      const double e0 = BufferValueAtShift(ema, need, shift);
      const double e1 = BufferValueAtShift(ema, need, shift + 1);

      if(c0 > e0 && c1 <= e1)
      {
         regime_start = iTime(_Symbol, _Period, shift);
         return true;
      }
   }

   regime_start = iTime(_Symbol, _Period, need - 1);
   return true;
}

int FindSnapshotIndexByPositionId(const RegimePositionSnapshot &snapshots[], const ulong position_id)
{
   const int total = ArraySize(snapshots);
   for(int i = 0; i < total; ++i)
   {
      if(snapshots[i].position_id == position_id)
         return i;
   }
   return -1;
}

double HighestPriceBetween(const datetime from_time, const datetime to_time, const bool include_live_bid)
{
   if(from_time <= 0)
      return SymbolInfoDouble(_Symbol, SYMBOL_BID);

   int from_shift = iBarShift(_Symbol, _Period, from_time, false);
   int to_shift   = (to_time > 0 ? iBarShift(_Symbol, _Period, to_time, false) : 0);
   if(from_shift < 0)
      return SymbolInfoDouble(_Symbol, SYMBOL_BID);
   if(to_shift < 0)
      to_shift = 0;
   if(from_shift < to_shift)
      from_shift = to_shift;

   const int count = from_shift - to_shift + 1;
   const int idx = iHighest(_Symbol, _Period, MODE_HIGH, count, to_shift);
   double highest = (idx >= 0 ? iHigh(_Symbol, _Period, idx) : SymbolInfoDouble(_Symbol, SYMBOL_BID));

   if(include_live_bid)
   {
      const double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
      if(bid > highest)
         highest = bid;
   }
   return highest;
}

bool LoadRegimeStats(const datetime regime_start, RegimeStats &stats)
{
   stats.regime_start = regime_start;
   stats.latest_entry_time = 0;
   stats.entry_count = 0;
   stats.failed_continuation_adds = 0;

   if(regime_start <= 0)
      return false;
   if(!HistorySelect(regime_start, TimeCurrent()))
      return false;

   RegimePositionSnapshot snapshots[];
   const int total = HistoryDealsTotal();
   for(int i = 0; i < total; ++i)
   {
      const ulong deal_ticket = HistoryDealGetTicket(i);
      if(deal_ticket == 0)
         continue;

      if((ulong)HistoryDealGetInteger(deal_ticket, DEAL_MAGIC) != InpMagicNumber)
         continue;
      if(HistoryDealGetString(deal_ticket, DEAL_SYMBOL) != _Symbol)
         continue;

      const ulong position_id = (ulong)HistoryDealGetInteger(deal_ticket, DEAL_POSITION_ID);
      const string comment = HistoryDealGetString(deal_ticket, DEAL_COMMENT);
      const datetime comment_regime_start = ParseRegimeStart(comment);
      const TradeKind kind = ParseTradeKind(comment);
      if(comment_regime_start != regime_start || kind == TRADE_UNKNOWN)
         continue;

      const ENUM_DEAL_ENTRY entry_type = (ENUM_DEAL_ENTRY)HistoryDealGetInteger(deal_ticket, DEAL_ENTRY);
      const datetime deal_time = (datetime)HistoryDealGetInteger(deal_ticket, DEAL_TIME);
      int idx = FindSnapshotIndexByPositionId(snapshots, position_id);
      if(idx < 0)
      {
         idx = ArraySize(snapshots);
         ArrayResize(snapshots, idx + 1);
         snapshots[idx].position_id = position_id;
         snapshots[idx].kind = kind;
         snapshots[idx].entry_time = 0;
         snapshots[idx].exit_time = 0;
         snapshots[idx].entry_price = 0.0;
         snapshots[idx].realized_pnl = 0.0;
         snapshots[idx].has_entry = false;
      }

      if(entry_type == DEAL_ENTRY_IN && (ENUM_DEAL_TYPE)HistoryDealGetInteger(deal_ticket, DEAL_TYPE) == DEAL_TYPE_BUY)
      {
         ++stats.entry_count;
         if(deal_time > stats.latest_entry_time)
            stats.latest_entry_time = deal_time;

         snapshots[idx].kind = kind;
         snapshots[idx].entry_time = deal_time;
         snapshots[idx].entry_price = HistoryDealGetDouble(deal_ticket, DEAL_PRICE);
         snapshots[idx].has_entry = true;
      }
      else if(entry_type == DEAL_ENTRY_OUT)
      {
         snapshots[idx].realized_pnl += HistoryDealGetDouble(deal_ticket, DEAL_PROFIT);
         snapshots[idx].realized_pnl += HistoryDealGetDouble(deal_ticket, DEAL_COMMISSION);
         snapshots[idx].realized_pnl += HistoryDealGetDouble(deal_ticket, DEAL_SWAP);
         if(deal_time > snapshots[idx].exit_time)
            snapshots[idx].exit_time = deal_time;
      }
   }

   for(int i = 0; i < ArraySize(snapshots); ++i)
   {
      if(!snapshots[i].has_entry || snapshots[i].entry_price <= 0.0)
         continue;

      const double highest = HighestPriceBetween(
         snapshots[i].entry_time,
         snapshots[i].exit_time,
         snapshots[i].exit_time == 0
      );
      const double mfe = highest - snapshots[i].entry_price;
      if(snapshots[i].kind == TRADE_CONT && snapshots[i].exit_time > 0 && mfe < InpStopDistancePrice)
         ++stats.failed_continuation_adds;
   }

   return true;
}

int StrategyPositionCount()
{
   int count = 0;
   for(int i = PositionsTotal() - 1; i >= 0; --i)
   {
      const ulong ticket = PositionGetTicket(i);
      if(ticket == 0 || !PositionSelectByTicket(ticket))
         continue;
      if(PositionGetString(POSITION_SYMBOL) != _Symbol)
         continue;
      if((ulong)PositionGetInteger(POSITION_MAGIC) != InpMagicNumber)
         continue;
      if((ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE) != POSITION_TYPE_BUY)
         continue;
      ++count;
   }
   return count;
}

double PositionRiskMoney(const ulong ticket)
{
   if(ticket == 0 || !PositionSelectByTicket(ticket))
      return 0.0;

   const double volume = PositionGetDouble(POSITION_VOLUME);
   const double sl     = PositionGetDouble(POSITION_SL);
   if(volume <= 0.0 || sl <= 0.0)
      return 0.0;

   const double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   if(sl >= bid)
      return 0.0;

   double pnl = 0.0;
   if(!OrderCalcProfit(ORDER_TYPE_BUY, _Symbol, volume, bid, sl, pnl))
      return 0.0;

   return MathMax(0.0, -pnl);
}

double OpenRiskMoney()
{
   double risk = 0.0;
   for(int i = PositionsTotal() - 1; i >= 0; --i)
   {
      const ulong ticket = PositionGetTicket(i);
      if(ticket == 0 || !PositionSelectByTicket(ticket))
         continue;
      if(PositionGetString(POSITION_SYMBOL) != _Symbol)
         continue;
      if((ulong)PositionGetInteger(POSITION_MAGIC) != InpMagicNumber)
         continue;
      if((ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE) != POSITION_TYPE_BUY)
         continue;
      risk += PositionRiskMoney(ticket);
   }
   return risk;
}

double LossForVolume(const double volume, const double entry_price, const double stop_price)
{
   double pnl = 0.0;
   if(!OrderCalcProfit(ORDER_TYPE_BUY, _Symbol, volume, entry_price, stop_price, pnl))
      return 0.0;
   return MathMax(0.0, -pnl);
}

double FloorToStep(const double value, const double step)
{
   if(step <= 0.0)
      return value;
   return MathFloor((value + 1e-12) / step) * step;
}

double CalculateVolume(const double risk_money, const double entry_price, const double stop_price)
{
   const double min_volume = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   const double max_volume = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);
   const double step       = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);

   const double loss_per_lot = LossForVolume(1.0, entry_price, stop_price);
   if(loss_per_lot <= 0.0)
      return 0.0;

   double raw = risk_money / loss_per_lot;
   if(raw <= 0.0)
      return 0.0;

   double volume = FloorToStep(raw, step);
   if(volume < min_volume)
      volume = min_volume;
   if(volume > max_volume)
      volume = FloorToStep(max_volume, step);

   const int vol_digits = (step >= 1.0 ? 0 : (int)MathRound(-MathLog10(step)));
   return NormalizeDouble(volume, vol_digits);
}

bool Fresh100BarHighAfterEntry(const datetime entry_time)
{
   const int entry_shift = iBarShift(_Symbol, _Period, entry_time, false);
   if(entry_shift < InpCooldownBars)
      return false;

   for(int shift = entry_shift - 1; shift >= 1; --shift)
   {
      if(IsFresh100BarHighAtShift(shift))
         return true;
   }

   return false;
}

bool EntryRearmed(const RegimeStats &stats)
{
   if(stats.latest_entry_time == 0)
      return true;
   return Fresh100BarHighAfterEntry(stats.latest_entry_time);
}

bool AnyFresh100BarHighSince(const datetime since_time)
{
   if(since_time <= 0)
      return false;

   const int since_shift = iBarShift(_Symbol, _Period, since_time, false);
   if(since_shift < 2)
      return false;

   for(int shift = since_shift - 1; shift >= 1; --shift)
   {
      if(IsFresh100BarHighAtShift(shift))
         return true;
   }

   return false;
}

bool ContinuationPassesRecentLossFilters(const double candidate_price)
{
   if(!InpContinuationRequireFreshHighAfterLoss
      && (InpBlockAfterRecentLosingSameLevelCount <= 0
      || InpRecentLosingSameLevelLookbackBars <= 0
      || InpRecentLosingSameLevelDistanceATR <= 0.0))
      return true;

   if(!HistorySelect(0, TimeCurrent()))
      return true;

   const double atr = ATRAtShift(1);
   const int lookback_bars = MathMax(1, InpRecentLosingSameLevelLookbackBars);
   const double max_same_level_distance = atr * InpRecentLosingSameLevelDistanceATR;

   RegimePositionSnapshot snapshots[];
   const int total = HistoryDealsTotal();
   for(int i = 0; i < total; ++i)
   {
      const ulong deal_ticket = HistoryDealGetTicket(i);
      if(deal_ticket == 0)
         continue;
      if((ulong)HistoryDealGetInteger(deal_ticket, DEAL_MAGIC) != InpMagicNumber)
         continue;
      if(HistoryDealGetString(deal_ticket, DEAL_SYMBOL) != _Symbol)
         continue;

      const ulong position_id = (ulong)HistoryDealGetInteger(deal_ticket, DEAL_POSITION_ID);
      int idx = FindSnapshotIndexByPositionId(snapshots, position_id);
      if(idx < 0)
      {
         idx = ArraySize(snapshots);
         ArrayResize(snapshots, idx + 1);
         snapshots[idx].position_id = position_id;
         snapshots[idx].kind = TRADE_UNKNOWN;
         snapshots[idx].entry_time = 0;
         snapshots[idx].exit_time = 0;
         snapshots[idx].entry_price = 0.0;
         snapshots[idx].realized_pnl = 0.0;
         snapshots[idx].has_entry = false;
      }

      const string comment = HistoryDealGetString(deal_ticket, DEAL_COMMENT);
      const TradeKind kind = ParseTradeKind(comment);
      if(kind != TRADE_UNKNOWN)
         snapshots[idx].kind = kind;

      const ENUM_DEAL_ENTRY entry_type = (ENUM_DEAL_ENTRY)HistoryDealGetInteger(deal_ticket, DEAL_ENTRY);
      const datetime deal_time = (datetime)HistoryDealGetInteger(deal_ticket, DEAL_TIME);
      if(entry_type == DEAL_ENTRY_IN && (ENUM_DEAL_TYPE)HistoryDealGetInteger(deal_ticket, DEAL_TYPE) == DEAL_TYPE_BUY)
      {
         snapshots[idx].entry_time = deal_time;
         snapshots[idx].entry_price = HistoryDealGetDouble(deal_ticket, DEAL_PRICE);
         snapshots[idx].has_entry = true;
      }
      else if(entry_type == DEAL_ENTRY_OUT)
      {
         snapshots[idx].realized_pnl += HistoryDealGetDouble(deal_ticket, DEAL_PROFIT);
         snapshots[idx].realized_pnl += HistoryDealGetDouble(deal_ticket, DEAL_COMMISSION);
         snapshots[idx].realized_pnl += HistoryDealGetDouble(deal_ticket, DEAL_SWAP);
         if(deal_time > snapshots[idx].exit_time)
            snapshots[idx].exit_time = deal_time;
      }
   }

   datetime latest_loss_exit_time = 0;
   int same_level_recent_losses = 0;
   for(int i = 0; i < ArraySize(snapshots); ++i)
   {
      const RegimePositionSnapshot snapshot = snapshots[i];
      if(!snapshot.has_entry || snapshot.exit_time <= 0 || snapshot.realized_pnl > 0.0)
         continue;

      if(snapshot.exit_time > latest_loss_exit_time)
         latest_loss_exit_time = snapshot.exit_time;

      if(snapshot.kind != TRADE_CONT)
         continue;
      if(atr <= 0.0 || InpBlockAfterRecentLosingSameLevelCount <= 0)
         continue;

      const int entry_shift = iBarShift(_Symbol, _Period, snapshot.entry_time, false);
      if(entry_shift < 1 || entry_shift > lookback_bars)
         continue;

      if(MathAbs(candidate_price - snapshot.entry_price) <= max_same_level_distance + SymbolInfoDouble(_Symbol, SYMBOL_POINT))
         ++same_level_recent_losses;
   }

   if(InpContinuationRequireFreshHighAfterLoss
      && latest_loss_exit_time > 0
      && !AnyFresh100BarHighSince(latest_loss_exit_time))
      return false;

   if(InpBlockAfterRecentLosingSameLevelCount > 0
      && same_level_recent_losses >= InpBlockAfterRecentLosingSameLevelCount)
      return false;

   return true;
}

bool IsFresh100BarHighAtShift(const int shift)
{
   if(shift < 1)
      return false;
   const int idx = iHighest(_Symbol, _Period, MODE_HIGH, InpHighestHighLookback, shift + 1);
   if(idx < 0)
      return false;
   return (iHigh(_Symbol, _Period, shift) > iHigh(_Symbol, _Period, idx));
}

int BarsSinceLastFreshHighInRegime(const datetime regime_start)
{
   for(int shift = 1; shift < InpMaxScanBars; ++shift)
   {
      const datetime bar_time = iTime(_Symbol, _Period, shift);
      if(bar_time == 0 || bar_time < regime_start)
         break;
      if(IsFresh100BarHighAtShift(shift))
         return shift - 1;
   }

   const int start_shift = iBarShift(_Symbol, _Period, regime_start, false);
   if(start_shift < 1)
      return InpMaxScanBars;
   return MathMax(0, start_shift - 1);
}

double RollingEfficiencyAtShift(const int shift, const int lookback)
{
   if(shift < 1 || Bars(_Symbol, _Period) <= shift + lookback)
      return 0.0;

   const double current_close = iClose(_Symbol, _Period, shift);
   const double prior_close   = iClose(_Symbol, _Period, shift + lookback);
   double volatility = 0.0;
   for(int i = 0; i < lookback; ++i)
   {
      const double c0 = iClose(_Symbol, _Period, shift + i);
      const double c1 = iClose(_Symbol, _Period, shift + i + 1);
      volatility += MathAbs(c0 - c1);
   }

   if(volatility <= 0.0)
      return 0.0;
   return MathAbs(current_close - prior_close) / volatility;
}

bool EntrySignal(double &reference_high)
{
   reference_high = 0.0;

   if(Bars(_Symbol, _Period) < InpEMAPeriod + InpHighestHighLookback + 5)
      return false;

   double ema_vals[];
   if(!CopyIndicatorBuffer(g_ema_handle, 3, ema_vals))
      return false;

   const double close1 = iClose(_Symbol, _Period, 1);
   const double open1  = iOpen(_Symbol, _Period, 1);
   const double close2 = iClose(_Symbol, _Period, 2);
   const double open2  = iOpen(_Symbol, _Period, 2);
   const double ema1   = BufferValueAtShift(ema_vals, 3, 1);

   if(close1 <= ema1)
      return false;
   if(close1 <= open1 || close2 <= open2)
      return false;

   const int hh_idx = iHighest(_Symbol, _Period, MODE_HIGH, InpHighestHighLookback, 2);
   if(hh_idx < 0)
      return false;

   reference_high = iHigh(_Symbol, _Period, hh_idx);
   if(close1 >= reference_high)
      return false;

   return true;
}

bool IsConfirmedPeak(const double &osma[], const int copied_count, const int shift)
{
   const double v  = BufferValueAtShift(osma, copied_count, shift);
   const double v1 = BufferValueAtShift(osma, copied_count, shift - 1);
   const double v2 = BufferValueAtShift(osma, copied_count, shift - 2);
   const double v3 = BufferValueAtShift(osma, copied_count, shift + 1);
   const double v4 = BufferValueAtShift(osma, copied_count, shift + 2);

   if(v == EMPTY_VALUE || v1 == EMPTY_VALUE || v2 == EMPTY_VALUE || v3 == EMPTY_VALUE || v4 == EMPTY_VALUE)
      return false;

   return (v > 0.0 && v > v1 && v > v2 && v > v3 && v > v4);
}

bool BearishMacdDoublePeakSinceEntry(const datetime entry_time)
{
   const int entry_shift = iBarShift(_Symbol, _Period, entry_time, false);
   if(entry_shift < 6)
      return false;

   const int count = entry_shift + 3;
   double osma[];
   if(!CopyIndicatorBuffer(g_osma_handle, count, osma))
      return false;

   double prev_peak = EMPTY_VALUE;
   double last_peak = EMPTY_VALUE;

   for(int shift = entry_shift - 1; shift >= 2; --shift)
   {
      if(!IsConfirmedPeak(osma, count, shift))
         continue;

      prev_peak = last_peak;
      last_peak = BufferValueAtShift(osma, count, shift);
   }

   if(prev_peak == EMPTY_VALUE || last_peak == EMPTY_VALUE)
      return false;

   return (last_peak < prev_peak);
}

int BarsInTrade(const datetime entry_time)
{
   const int shift = iBarShift(_Symbol, _Period, entry_time, false);
   return MathMax(0, shift);
}

double HighestSinceEntry(const datetime entry_time)
{
   return HighestPriceBetween(entry_time, 0, true);
}

double HighestSinceEntryClosedBars(const datetime entry_time)
{
   const int entry_shift = iBarShift(_Symbol, _Period, entry_time, false);
   if(entry_shift < 1)
      return SymbolInfoDouble(_Symbol, SYMBOL_BID);

   const int idx = iHighest(_Symbol, _Period, MODE_HIGH, entry_shift, 1);
   if(idx < 0)
      return SymbolInfoDouble(_Symbol, SYMBOL_BID);

   return iHigh(_Symbol, _Period, idx);
}

double ATRAtShift(const int shift)
{
   const int need = shift + 2;
   double atr_vals[];
   if(!CopyIndicatorBuffer(g_atr_handle, need, atr_vals))
      return 0.0;
   const double atr = BufferValueAtShift(atr_vals, need, shift);
   if(atr == EMPTY_VALUE)
      return 0.0;
   return atr;
}

string SessionBucketOfTime(const datetime bar_time)
{
   MqlDateTime parts;
   TimeToStruct(bar_time, parts);
   if(parts.hour < 8)
      return "Asia";
   if(parts.hour < 13)
      return "London";
   if(parts.hour < 22)
      return "NewYork";
   return "LateUS";
}

double TrendPersistenceAtShift(const int shift, const int lookback)
{
   if(shift < 1 || Bars(_Symbol, _Period) <= shift + lookback)
      return 0.0;

   const double direction = MathAbs(iClose(_Symbol, _Period, shift) - iClose(_Symbol, _Period, shift + lookback));
   double path = 0.0;
   for(int i = shift; i < shift + lookback; ++i)
      path += MathAbs(iClose(_Symbol, _Period, i) - iClose(_Symbol, _Period, i + 1));

   if(path <= 0.0)
      return 0.0;
   return direction / path;
}

double FreshHighFrequencyAtShift(const int shift, const int lookback)
{
   int hits = 0;
   int n = 0;
   for(int i = shift; i < shift + lookback && i < Bars(_Symbol, _Period) - 2; ++i)
   {
      if(IsFresh100BarHighAtShift(i))
         ++hits;
      ++n;
   }
   if(n <= 0)
      return 0.0;
   return (double)hits / (double)n;
}

double FractionClosesAboveEMAFastAtShift(const int shift, const int lookback)
{
   const int need = shift + lookback + 2;
   double ema_fast_vals[];
   if(!CopyIndicatorBuffer(g_ema_fast_handle, need, ema_fast_vals))
      return 0.0;

   int hits = 0;
   int n = 0;
   for(int i = shift; i < shift + lookback && i < Bars(_Symbol, _Period) - 1; ++i)
   {
      const double ema_fast = BufferValueAtShift(ema_fast_vals, need, i);
      if(ema_fast == EMPTY_VALUE)
         continue;
      if(iClose(_Symbol, _Period, i) > ema_fast)
         ++hits;
      ++n;
   }
   if(n <= 0)
      return 0.0;
   return (double)hits / (double)n;
}

double DistanceAboveEMA200ZAtShift(const int shift)
{
   const int lookback = MathMax(20, InpStateZScoreLookback);
   const int need = shift + lookback + 2;
   if(Bars(_Symbol, _Period) <= shift + lookback)
      return 0.0;

   double ema_vals[];
   if(!CopyIndicatorBuffer(g_ema_handle, need, ema_vals))
      return 0.0;

   double current = 0.0;
   double sum = 0.0;
   double sum_sq = 0.0;
   int n = 0;
   for(int i = shift; i < shift + lookback; ++i)
   {
      const double ema_value = BufferValueAtShift(ema_vals, need, i);
      if(ema_value == EMPTY_VALUE)
         continue;
      const double distance = iClose(_Symbol, _Period, i) - ema_value;
      if(i == shift)
         current = distance;
      sum += distance;
      sum_sq += distance * distance;
      ++n;
   }
   if(n < 2)
      return 0.0;

   const double mean = sum / (double)n;
   const double variance = MathMax(0.0, (sum_sq / (double)n) - mean * mean);
   const double std = MathSqrt(variance);
   if(std <= 1e-12)
      return 0.0;
   return NormalizeSymmetric((current - mean) / std, 3.0);
}

double PullbackDepthATRAtShift(const int shift)
{
   const double atr = ATRAtShift(shift);
   if(atr <= 0.0)
      return 0.0;

   const int hh_idx = iHighest(_Symbol, _Period, MODE_HIGH, InpHighestHighLookback, shift + 1);
   if(hh_idx < 0)
      return 0.0;
   const double prior_high = iHigh(_Symbol, _Period, hh_idx);
   return (prior_high - iClose(_Symbol, _Period, shift)) / atr;
}

int BarsSinceLastFreshHighAtShift(const datetime regime_start, const int shift)
{
   for(int i = shift; i < InpMaxScanBars; ++i)
   {
      const datetime bar_time = iTime(_Symbol, _Period, i);
      if(bar_time == 0 || bar_time < regime_start)
         break;
      if(IsFresh100BarHighAtShift(i))
         return i - shift;
   }

   const int start_shift = iBarShift(_Symbol, _Period, regime_start, false);
   if(start_shift < shift)
      return 0;
   return MathMax(0, start_shift - shift);
}

double RegimePeakPriceAtShift(const datetime regime_start, const int shift)
{
   const int start_shift = iBarShift(_Symbol, _Period, regime_start, false);
   if(start_shift < shift)
      return iHigh(_Symbol, _Period, shift);

   double peak = iHigh(_Symbol, _Period, shift);
   for(int i = shift; i <= start_shift; ++i)
   {
      const double high = iHigh(_Symbol, _Period, i);
      if(high > peak)
         peak = high;
   }
   return peak;
}

double RegimePeakToCloseRetracementATRAtShift(const datetime regime_start, const int shift)
{
   const double atr = ATRAtShift(shift);
   if(atr <= 0.0 || regime_start <= 0)
      return 0.0;
   const double peak = RegimePeakPriceAtShift(regime_start, shift);
   return (peak - iClose(_Symbol, _Period, shift)) / atr;
}

double ZEma20ATRAtShift(const int shift)
{
   const double atr = ATRAtShift(shift);
   const double ema_fast = EMAFastValueAtShift(shift);
   if(atr <= 0.0)
      return 0.0;
   return (iClose(_Symbol, _Period, shift) - ema_fast) / atr;
}

double ZDecayFromPeakAtShift(const datetime regime_start, const int shift)
{
   if(regime_start <= 0)
      return 0.0;

   const int start_shift = iBarShift(_Symbol, _Period, regime_start, false);
   if(start_shift < shift)
      return 0.0;

   double peak = -DBL_MAX;
   for(int i = shift; i <= start_shift; ++i)
   {
      const double current_z = ZEma20ATRAtShift(i);
      if(current_z > peak)
         peak = current_z;
   }
   if(peak <= -DBL_MAX / 2.0)
      return 0.0;
   return peak - ZEma20ATRAtShift(shift);
}

double BollingerBandwidthChangeAtShift(const int shift, const int lookback)
{
   if(Bars(_Symbol, _Period) <= shift + lookback + 1)
      return 0.0;

   double sum_now = 0.0;
   double sum_sq_now = 0.0;
   double sum_prev = 0.0;
   double sum_sq_prev = 0.0;
   for(int i = shift; i < shift + lookback; ++i)
   {
      const double close_now = iClose(_Symbol, _Period, i);
      const double close_prev = iClose(_Symbol, _Period, i + 1);
      sum_now += close_now;
      sum_sq_now += close_now * close_now;
      sum_prev += close_prev;
      sum_sq_prev += close_prev * close_prev;
   }

   const double mean_now = sum_now / (double)lookback;
   const double mean_prev = sum_prev / (double)lookback;
   const double std_now = MathSqrt(MathMax(0.0, (sum_sq_now / (double)lookback) - mean_now * mean_now));
   const double std_prev = MathSqrt(MathMax(0.0, (sum_sq_prev / (double)lookback) - mean_prev * mean_prev));
   return (4.0 * std_now) - (4.0 * std_prev);
}

bool FailedBreakoutFlagAtShift(const int shift)
{
   const double atr = ATRAtShift(shift);
   if(atr <= 0.0)
      return false;

   const int hh_idx = iHighest(_Symbol, _Period, MODE_HIGH, InpHighestHighLookback, shift + 1);
   if(hh_idx < 0)
      return false;
   const double prior_high = iHigh(_Symbol, _Period, hh_idx);
   const double high = iHigh(_Symbol, _Period, shift);
   const double close = iClose(_Symbol, _Period, shift);
   return (high >= prior_high - atr * 0.25 && high <= prior_high + atr * 0.25 && close < prior_high);
}

double FailedBreakoutCountAtShift(const int shift, const int lookback)
{
   int count = 0;
   for(int i = shift; i < shift + lookback && i < Bars(_Symbol, _Period) - 2; ++i)
   {
      if(FailedBreakoutFlagAtShift(i))
         ++count;
   }
   return (double)count;
}

bool StalledCompressionFlagAtShift(const datetime regime_start, const int shift)
{
   const int lookback = MathMax(InpEfficiencyLookback, 5);
   const double efficiency = RollingEfficiencyAtShift(shift, InpEfficiencyLookback);
   const double bandwidth_change = BollingerBandwidthChangeAtShift(shift, lookback);
   const int stale_bars = BarsSinceLastFreshHighAtShift(regime_start, shift);
   return (efficiency < 0.25 && bandwidth_change <= 0.0 && stale_bars >= lookback / 2);
}

int ConsecutiveBarsWithoutNewHighSinceEntry(const datetime entry_time)
{
   const int entry_shift = iBarShift(_Symbol, _Period, entry_time, false);
   if(entry_shift < 2)
      return 0;

   double highest = iHigh(_Symbol, _Period, entry_shift);
   int bars_without = 0;
   for(int shift = entry_shift - 1; shift >= 1; --shift)
   {
      const double high = iHigh(_Symbol, _Period, shift);
      if(high > highest)
      {
         highest = high;
         bars_without = 0;
      }
      else
      {
         ++bars_without;
      }
   }
   return bars_without;
}

double ScoreContinuationQuality(const datetime regime_start, const int shift)
{
   const int lookback = MathMax(InpEfficiencyLookback, 5);
   const double atr = ATRAtShift(shift);
   const double fast_slope = (atr > 0.0 ? EMAFastSlopeAtShift(shift) / atr : 0.0);
   const double mid_slope = (atr > 0.0 ? EMAMidSlopeAtShift(shift) / atr : 0.0);

   double total = 0.0;
   double total_weight = 0.0;

   total += NormalizeWindow01ToSymmetric(RollingEfficiencyAtShift(shift, InpEfficiencyLookback), 0.0, 1.0) * 0.20;
   total_weight += 0.20;
   total += NormalizeWindow01ToSymmetric(TrendPersistenceAtShift(shift, 20), 0.0, 1.0) * 0.20;
   total_weight += 0.20;
   total += NormalizeSymmetric(fast_slope, 0.25) * 0.15;
   total_weight += 0.15;
   total += NormalizeSymmetric(mid_slope, 0.20) * 0.10;
   total_weight += 0.10;
   total += NormalizeWindow01ToSymmetric(FreshHighFrequencyAtShift(shift, lookback), 0.0, 1.0) * 0.15;
   total_weight += 0.15;
   total += NormalizeWindow01ToSymmetric(FractionClosesAboveEMAFastAtShift(shift, lookback), 0.0, 1.0) * 0.10;
   total_weight += 0.10;
   total += DistanceAboveEMA200ZAtShift(shift) * 0.10;
   total_weight += 0.10;

   if(total_weight <= 0.0)
      return 0.0;
   return ClampValue(total / total_weight, -1.0, 1.0);
}

double ScoreFragilityQuality(const datetime regime_start, const int shift)
{
   const int lookback = MathMax(InpEfficiencyLookback, 5);
   double total = 0.0;
   double total_weight = 0.0;

   total += NormalizeWindow01ToSymmetric(PullbackDepthATRAtShift(shift), 0.0, 3.0) * 0.20;
   total_weight += 0.20;
   total += NormalizeWindow01ToSymmetric(RegimePeakToCloseRetracementATRAtShift(regime_start, shift), 0.0, 3.0) * 0.20;
   total_weight += 0.20;
   total += NormalizeWindow01ToSymmetric(ZDecayFromPeakAtShift(regime_start, shift), 0.0, 3.0) * 0.20;
   total_weight += 0.20;
   total += NormalizeWindow01ToSymmetric((double)BarsSinceLastFreshHighAtShift(regime_start, shift), 0.0, (double)lookback) * 0.15;
   total_weight += 0.15;
   total += NormalizeWindow01ToSymmetric(FailedBreakoutCountAtShift(shift, lookback), 0.0, 4.0) * 0.15;
   total_weight += 0.15;
   total += (StalledCompressionFlagAtShift(regime_start, shift) ? 1.0 : -1.0) * 0.10;
   total_weight += 0.10;

   if(total_weight <= 0.0)
      return 0.0;
   return ClampValue(total / total_weight, -1.0, 1.0);
}

void ComputeStateQualitySnapshot(const datetime regime_start, StateQualitySnapshot &snapshot)
{
   const int shift = 1;
   snapshot.continuation_score = ScoreContinuationQuality(regime_start, shift);
   snapshot.fragility_score = ScoreFragilityQuality(regime_start, shift);
   snapshot.score = ClampValue(
      snapshot.continuation_score - InpStateTwoScorePenaltyWeight * snapshot.fragility_score,
      -1.0,
      1.0
   );

   if(snapshot.score >= InpStateHighMinScore)
      snapshot.tier = STATE_TIER_HIGH;
   else if(snapshot.score >= InpStateMediumMinScore)
      snapshot.tier = STATE_TIER_MEDIUM;
   else
      snapshot.tier = STATE_TIER_LOW;

   if(snapshot.tier == STATE_TIER_HIGH)
      snapshot.risk_multiplier = InpStateHighRiskMultiplier;
   else if(snapshot.tier == STATE_TIER_MEDIUM)
      snapshot.risk_multiplier = InpStateMediumRiskMultiplier;
   else
      snapshot.risk_multiplier = InpStateLowRiskMultiplier;
}

bool LowTierNoNewHighExitTriggered(const datetime entry_time, const double entry_price)
{
   const int bars_held = BarsInTrade(entry_time);
   if(bars_held < InpLowNoNewHighMinBarsOpen)
      return false;

   const double highest_closed = HighestSinceEntryClosedBars(entry_time);
   const double mfe_r = (highest_closed - entry_price) / InpStopDistancePrice;
   if(mfe_r < InpLowNoNewHighMinMFER)
      return false;

   return (ConsecutiveBarsWithoutNewHighSinceEntry(entry_time) >= InpLowNoNewHighExitBars);
}

bool MediumTierRetracementExitTriggered(const datetime entry_time, const double entry_price)
{
   const int bars_held = BarsInTrade(entry_time);
   if(bars_held < InpMediumRetraceMinBarsOpen)
      return false;

   const double highest_closed = HighestSinceEntryClosedBars(entry_time);
   const double mfe = highest_closed - entry_price;
   const double mfe_r = mfe / InpStopDistancePrice;
   if(mfe <= 0.0 || mfe_r < InpMediumRetraceMinMFER)
      return false;

   const double current_gain = MathMax(iClose(_Symbol, _Period, 1) - entry_price, 0.0);
   const double retracement_frac = (mfe - current_gain) / mfe;
   return (retracement_frac >= InpMediumRetraceFrac);
}

double EMA20SlopeDecaySinceRegime(const datetime regime_start)
{
   if(regime_start <= 0)
      return 0.0;

   const int start_shift = iBarShift(_Symbol, _Period, regime_start, false);
   if(start_shift < InpEMASlopeLookback + 1)
      return 0.0;

   double peak_slope = -1.0e100;
   for(int shift = start_shift - 1; shift >= 1; --shift)
   {
      const double slope = EMAFastSlopeAtShift(shift);
      if(slope > peak_slope)
         peak_slope = slope;
   }

   if(peak_slope <= -1.0e99)
      return 0.0;

   const double current_slope = EMAFastSlopeAtShift(1);
   return MathMax(0.0, peak_slope - current_slope);
}

bool BuildSurvivorStructureState(const datetime entry_time, const double entry_price, SurvivorStructureState &state)
{
   state.failed_high_attempts = 0;
   state.lower_high_confirmed = false;
   state.lower_low_confirmed = false;
   state.lower_high_then_swing_low_break = false;
   state.failed_breakout_flag = false;
   state.last_confirmed_swing_low = 0.0;
   state.last_confirmed_swing_high = 0.0;

   const int lookback = MathMax(InpEfficiencyLookback, 5);
   const int entry_shift = iBarShift(_Symbol, _Period, entry_time, false);
   if(entry_shift < 4)
      return false;

   double previous_highest = iHigh(_Symbol, _Period, entry_shift);
   bool failed_high_zone_active = false;
   double prior_swing_high = 0.0;
   bool has_prior_swing_high = false;
   double prior_swing_low = 0.0;
   bool has_prior_swing_low = false;
   int lower_high_shift = -1;

   for(int shift = entry_shift - 1; shift >= 1; --shift)
   {
      const double high = iHigh(_Symbol, _Period, shift);
      const double low = iLow(_Symbol, _Period, shift);
      const double close = iClose(_Symbol, _Period, shift);

      if(previous_highest > entry_price)
      {
         const double retest_band = 0.25 * InpStopDistancePrice;
         const double reset_band = 0.50 * InpStopDistancePrice;
         const bool near_prior_high = high >= (previous_highest - retest_band);
         const bool failed_retest = high <= previous_highest && near_prior_high;
         if(failed_retest && !failed_high_zone_active)
         {
            ++state.failed_high_attempts;
            failed_high_zone_active = true;
         }
         else if(high < (previous_highest - reset_band))
         {
            failed_high_zone_active = false;
         }
      }

      if(high > previous_highest)
      {
         previous_highest = high;
         failed_high_zone_active = false;
      }

      if(IsConfirmedSwingHighAtShift(shift))
      {
         const double swing_high = high;
         if(has_prior_swing_high && swing_high < prior_swing_high)
         {
            state.lower_high_confirmed = true;
            lower_high_shift = shift;
         }
         prior_swing_high = swing_high;
         has_prior_swing_high = true;
         state.last_confirmed_swing_high = swing_high;
      }

      if(IsConfirmedSwingLowAtShift(shift))
      {
         const double swing_low = low;
         if(has_prior_swing_low && swing_low < prior_swing_low)
            state.lower_low_confirmed = true;
         prior_swing_low = swing_low;
         has_prior_swing_low = true;
         state.last_confirmed_swing_low = swing_low;
      }

      if(state.last_confirmed_swing_high > 0.0)
      {
         const double atr = ATRAtShift(shift);
         const double band = atr * 0.25;
         const bool near_high_zone = high >= (state.last_confirmed_swing_high - band);
         if(near_high_zone && high <= (state.last_confirmed_swing_high + band) && close < state.last_confirmed_swing_high && shift == 1)
            state.failed_breakout_flag = true;
      }
   }

   if(lower_high_shift > 0 && state.last_confirmed_swing_low > 0.0)
   {
      const int bars_since_lower_high = lower_high_shift - 1;
      if(bars_since_lower_high >= 0 && bars_since_lower_high <= lookback)
      {
         const double close1 = iClose(_Symbol, _Period, 1);
         state.lower_high_then_swing_low_break = (close1 < state.last_confirmed_swing_low);
      }
   }

   return true;
}

bool ContinuationSurvivorTightenShouldFire(
   const datetime regime_start,
   const datetime entry_time,
   const double entry_price,
   const double highest_since_entry
)
{
   if(InpSurvivorFailedHighsThreshold <= 0 || InpSurvivorMinProfitR <= 0.0 || InpSurvivorTightenOffsetR <= 0.0)
      return false;

   const double mfe_r = (highest_since_entry - entry_price) / InpStopDistancePrice;
   if(mfe_r < InpSurvivorMinProfitR)
      return false;

   SurvivorStructureState structure;
   if(!BuildSurvivorStructureState(entry_time, entry_price, structure))
      return false;
   if(structure.failed_high_attempts < InpSurvivorFailedHighsThreshold)
      return false;
   if(InpSurvivorRequireLowerHighBreak && !structure.lower_high_then_swing_low_break)
      return false;
   if(InpSurvivorRequireFailedBreakout && !structure.failed_breakout_flag)
      return false;
   if(InpSurvivorMomentumRequireRegimeDead && !g_regime_continuation_dead)
      return false;
   if(EMA20SlopeDecaySinceRegime(regime_start) < InpSurvivorMinEMA20SlopeDecay)
      return false;
   return true;
}

void ResetRegimeControlState()
{
   g_state_regime_start = 0;
   g_regime_continuation_dead = false;
   g_regime_dead_reason = "";
   g_regime_rearm_count = 0;
   g_regime_last_rearm_bar = 0;
   g_regime_last_rearm_reason = "";
}

void SyncRegimeControlState(const datetime regime_start, const RegimeStats &stats)
{
   if(regime_start <= 0)
   {
      ResetRegimeControlState();
      return;
   }

   if(g_state_regime_start != regime_start)
   {
      ResetRegimeControlState();
      g_state_regime_start = regime_start;
   }

   const double efficiency = RollingEfficiencyAtShift(1, InpEfficiencyLookback);
   const int stale_bars = BarsSinceLastFreshHighInRegime(regime_start);

   double ema_mid_vals[];
   if(!CopyIndicatorBuffer(g_ema_mid_handle, 3, ema_mid_vals))
      return;
   const double ema50_1 = BufferValueAtShift(ema_mid_vals, 3, 1);
   const double close1 = iClose(_Symbol, _Period, 1);

   if(!g_regime_continuation_dead
      && stats.entry_count > 0
      && stats.failed_continuation_adds >= InpRegimeDeathFailedAddsThreshold
      && stale_bars >= InpRegimeDeathNoFreshHighBars
      && ema50_1 != EMPTY_VALUE
      && close1 < ema50_1
      && efficiency <= InpRegimeDeathMaxEfficiency)
   {
      g_regime_continuation_dead = true;
      g_regime_dead_reason = "regime_death_suspend_continuation";
      return;
   }

   if(g_regime_continuation_dead
      && IsFresh100BarHighAtShift(1)
      && efficiency >= InpRegimeRearmMinEfficiency)
   {
      g_regime_continuation_dead = false;
      g_regime_dead_reason = "";
      ++g_regime_rearm_count;
      g_regime_last_rearm_bar = iTime(_Symbol, _Period, 1);
      g_regime_last_rearm_reason = "regime_rearm_on_fresh_high_plus_efficiency_recovery";
   }
}

double AdjustStopForBuy(const double proposed_sl)
{
   const double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   const double min_gap = StopLevelDistance();
   double adjusted = proposed_sl;

   if(adjusted >= bid - min_gap)
      adjusted = bid - min_gap;

   return NormalizePrice(adjusted);
}

bool ModifyPositionStops(const ulong ticket, const double new_sl, const double current_tp)
{
   if(ticket == 0)
      return false;

   const double sl = AdjustStopForBuy(new_sl);
   if(sl <= 0.0)
      return false;

   if(!trade.PositionModify(ticket, sl, current_tp))
      return false;

   const uint retcode = trade.ResultRetcode();
   return (retcode == TRADE_RETCODE_DONE || retcode == TRADE_RETCODE_DONE_PARTIAL || retcode == TRADE_RETCODE_PLACED);
}

bool ClosePosition(const ulong ticket)
{
   if(!trade.PositionClose(ticket, InpDeviationPoints))
      return false;

   const uint retcode = trade.ResultRetcode();
   return (retcode == TRADE_RETCODE_DONE || retcode == TRADE_RETCODE_DONE_PARTIAL || retcode == TRADE_RETCODE_PLACED);
}

void RefreshRegimeState()
{
   datetime regime_start = 0;
   if(!CurrentBullRegime(regime_start))
   {
      ResetRegimeControlState();
      return;
   }

   RegimeStats stats;
   if(!LoadRegimeStats(regime_start, stats))
      return;

   SyncRegimeControlState(regime_start, stats);
}

void ManageOpenPositions()
{
   const double one_r             = InpStopDistancePrice;
   const double two_r             = InpBreakEvenTriggerR * one_r;
   const double timed_trail       = InpTimedTrailR * one_r;
   const double trend_trail       = InpTrendTrailAfter2RR * one_r;

   for(int i = PositionsTotal() - 1; i >= 0; --i)
   {
      const ulong ticket = PositionGetTicket(i);
      if(ticket == 0 || !PositionSelectByTicket(ticket))
         continue;
      if(PositionGetString(POSITION_SYMBOL) != _Symbol)
         continue;
      if((ulong)PositionGetInteger(POSITION_MAGIC) != InpMagicNumber)
         continue;
      if((ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE) != POSITION_TYPE_BUY)
         continue;

      const string comment       = PositionGetString(POSITION_COMMENT);
      const TradeKind kind       = ParseTradeKind(comment);
      const StateTier tier       = ParseStateTier(comment);
      const datetime regime_start = ParseRegimeStart(comment);
      const double entry         = PositionGetDouble(POSITION_PRICE_OPEN);
      const double current_sl    = PositionGetDouble(POSITION_SL);
      const double current_tp    = PositionGetDouble(POSITION_TP);
      const datetime entry_time  = (datetime)PositionGetInteger(POSITION_TIME);
      const int bars_held        = BarsInTrade(entry_time);
      const double highest_closed = HighestSinceEntryClosedBars(entry_time);
      const double mfe           = highest_closed - entry;

      double target_sl = current_sl;
      if(target_sl <= 0.0)
         target_sl = entry - one_r;

      if(kind == TRADE_FIRST)
      {
         if(tier == STATE_TIER_LOW && LowTierNoNewHighExitTriggered(entry_time, entry))
         {
            ClosePosition(ticket);
            continue;
         }

         if(tier == STATE_TIER_MEDIUM && MediumTierRetracementExitTriggered(entry_time, entry))
         {
            ClosePosition(ticket);
            continue;
         }

         if(mfe >= two_r)
         {
            if(entry > target_sl)
               target_sl = entry;
            const double trail_sl = highest_closed - trend_trail;
            if(trail_sl > target_sl)
               target_sl = trail_sl;
         }
         else if(bars_held >= InpTimedTrailActivationBars)
         {
            const double trail_sl = highest_closed - timed_trail;
            if(trail_sl > target_sl)
               target_sl = trail_sl;
         }

         if(target_sl > current_sl + SymbolInfoDouble(_Symbol, SYMBOL_POINT))
            ModifyPositionStops(ticket, target_sl, current_tp);

         if(BearishMacdDoublePeakSinceEntry(entry_time))
            ClosePosition(ticket);
      }
      else if(kind == TRADE_CONT)
      {
         if(mfe >= two_r)
         {
            const double be_plus = entry + InpContinuationBreakEvenOffsetR * one_r;
            if(be_plus > target_sl)
               target_sl = be_plus;
         }

         if(ContinuationSurvivorTightenShouldFire(regime_start, entry_time, entry, highest_closed))
         {
            const double survivor_sl = highest_closed - InpSurvivorTightenOffsetR * one_r;
            if(survivor_sl > target_sl)
               target_sl = survivor_sl;
         }

         if(target_sl > current_sl + SymbolInfoDouble(_Symbol, SYMBOL_POINT))
            ModifyPositionStops(ticket, target_sl, current_tp);
      }
   }
}

bool CanOpenAnotherTrade(const double new_trade_risk)
{
   if(StrategyPositionCount() >= InpMaxConcurrentTrades)
      return false;

   const double equity = AccountInfoDouble(ACCOUNT_EQUITY);
   const double open_risk_limit = equity * InpMaxOpenRiskPercent / 100.0;
   const double open_risk = OpenRiskMoney();
   return (open_risk + new_trade_risk <= open_risk_limit + 1e-8);
}

bool OpenStrategyTrade(
   const TradeKind kind,
   const datetime regime_start,
   const StateTier tier,
   const double risk_multiplier
)
{
   const double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   if(ask <= 0.0)
      return false;

   const double sl = NormalizePrice(ask - InpStopDistancePrice);
   double tp = 0.0;
   if(kind == TRADE_CONT)
      tp = NormalizePrice(ask + InpContinuationTakeProfitR * InpStopDistancePrice);

   const double risk_money = AccountInfoDouble(ACCOUNT_EQUITY)
      * InpRiskPercentPerTrade
      / 100.0
      * MathMax(risk_multiplier, 0.0);
   const double volume = CalculateVolume(risk_money, ask, sl);
   if(volume <= 0.0)
      return false;

   const double actual_risk = LossForVolume(volume, ask, sl);
   if(actual_risk <= 0.0)
      return false;
   if(!CanOpenAnotherTrade(actual_risk))
      return false;

   const string comment = KindTag(kind) + "|" + IntegerToString((long)regime_start) + "|" + TierName(tier);
   trade.SetExpertMagicNumber(InpMagicNumber);
   trade.SetDeviationInPoints(InpDeviationPoints);

   if(!trade.Buy(volume, _Symbol, 0.0, sl, tp, comment))
      return false;

   const uint retcode = trade.ResultRetcode();
   return (retcode == TRADE_RETCODE_DONE || retcode == TRADE_RETCODE_DONE_PARTIAL || retcode == TRADE_RETCODE_PLACED);
}

void EvaluateEntry()
{
   datetime regime_start = 0;
   if(!CurrentBullRegime(regime_start))
   {
      ResetRegimeControlState();
      return;
   }

   RegimeStats stats;
   if(!LoadRegimeStats(regime_start, stats))
      return;

   SyncRegimeControlState(regime_start, stats);

   double reference_high = 0.0;
   if(!EntrySignal(reference_high))
      return;

   StateQualitySnapshot state;
   ComputeStateQualitySnapshot(regime_start, state);

   const TradeKind kind = (stats.entry_count == 0 ? TRADE_FIRST : TRADE_CONT);
   if(kind == TRADE_CONT)
   {
      if(g_regime_continuation_dead)
         return;

      const double entry_efficiency = RollingEfficiencyAtShift(1, InpEfficiencyLookback);
      if(entry_efficiency < InpContinuationMinEfficiency)
         return;

      double candidate_price = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
      if(candidate_price <= 0.0)
         candidate_price = iClose(_Symbol, _Period, 1);
      if(!ContinuationPassesRecentLossFilters(candidate_price))
         return;
   }

   OpenStrategyTrade(kind, regime_start, state.tier, state.risk_multiplier);
}

int OnInit()
{
   if(!IsHedgingAccount())
   {
      Print("This EA requires a hedging account to support multiple concurrent trades.");
      return INIT_FAILED;
   }

   g_ema_handle = iMA(_Symbol, _Period, InpEMAPeriod, 0, MODE_EMA, PRICE_CLOSE);
   if(g_ema_handle == INVALID_HANDLE)
      return INIT_FAILED;

   g_ema_fast_handle = iMA(_Symbol, _Period, InpEMAFastPeriod, 0, MODE_EMA, PRICE_CLOSE);
   if(g_ema_fast_handle == INVALID_HANDLE)
      return INIT_FAILED;

   g_ema_mid_handle = iMA(_Symbol, _Period, InpEMAMidPeriod, 0, MODE_EMA, PRICE_CLOSE);
   if(g_ema_mid_handle == INVALID_HANDLE)
      return INIT_FAILED;

   g_atr_handle = iATR(_Symbol, _Period, InpATRPeriod);
   if(g_atr_handle == INVALID_HANDLE)
      return INIT_FAILED;

   g_osma_handle = iOsMA(_Symbol, _Period, InpMacdFast, InpMacdSlow, InpMacdSignal, PRICE_CLOSE);
   if(g_osma_handle == INVALID_HANDLE)
      return INIT_FAILED;

   trade.SetExpertMagicNumber(InpMagicNumber);
   trade.SetDeviationInPoints(InpDeviationPoints);
   ResetRegimeControlState();
   return INIT_SUCCEEDED;
}

void OnDeinit(const int reason)
{
   if(g_ema_handle != INVALID_HANDLE)
      IndicatorRelease(g_ema_handle);
   if(g_ema_fast_handle != INVALID_HANDLE)
      IndicatorRelease(g_ema_fast_handle);
   if(g_ema_mid_handle != INVALID_HANDLE)
      IndicatorRelease(g_ema_mid_handle);
   if(g_atr_handle != INVALID_HANDLE)
      IndicatorRelease(g_atr_handle);
   if(g_osma_handle != INVALID_HANDLE)
      IndicatorRelease(g_osma_handle);
}

void OnTick()
{
   if(!IsNewBar())
      return;

   RefreshRegimeState();
   ManageOpenPositions();
   EvaluateEntry();
}
