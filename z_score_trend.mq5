#property strict
#property version   "1.00"
#property description "XAUUSD dual-zscore SQ1 winner: p17 basket exit with high-only profit deferral and state-quality sizing"
#copyright JoshuaGrobler
#include <Trade/Trade.mqh>

enum ENUM_POSITION_SIZING_MODE
{
   POSITION_SIZE_FIXED_FRACTIONAL = 0,
   POSITION_SIZE_FIXED_LOTS       = 1
};

enum ENUM_STOP_MODE
{
   STOP_FIXED_POINTS = 0,
   STOP_ATR_BOUNDED  = 1
};

enum ENUM_STATE_TIER
{
   STATE_TIER_LOW    = 0,
   STATE_TIER_MEDIUM = 1,
   STATE_TIER_HIGH   = 2
};

input ulong                     InpMagic                       = 260712;
input ENUM_POSITION_SIZING_MODE InpPositionSizingMode          = POSITION_SIZE_FIXED_LOTS;
input double                    InpRiskPercentPerTrade         = 0.50;
input double                    InpFixedLots                   = 0.01;
input double                    InpMinLot                      = 0.01;
input double                    InpMaxLot                      = 100.0;
input double                    InpLotStep                     = 0.01;
input int                       InpSlippagePoints              = 20;
input bool                      InpDebugPrint                  = false;

input int                       InpRegimeLookback              = 500;
input int                       InpPercentileLookback          = 500;
input double                    InpExitPct                     = 17.0;
input bool                      InpEdgeTrigger                 = true;
input bool                      InpAllowFirstArmedEntry        = true;
input int                       InpMaxPositions                = 5;

input int                       InpA_EmaFast                   = 34;
input int                       InpA_EmaSlow                   = 144;
input int                       InpA_StdSpan                   = 55;
input double                    InpA_ZArm                      = 1.0;
input int                       InpA_ZSlopeN                   = 3;
input double                    InpA_Gamma                     = 0.07;

input bool                      InpUseStopLoss                 = true;
input ENUM_STOP_MODE            InpStopMode                    = STOP_ATR_BOUNDED;
input int                       InpStopLossPoints              = 1500;
input int                       InpStopATRPeriod               = 100;
input double                    InpStopATRMultiple             = 1.5;
input double                    InpStopMinPrice                = 10.0;
input double                    InpStopMaxPrice                = 25.0;

input bool                      InpUseBasketRiskCap            = true;
input double                    InpBasketRiskMultiple          = 2.5;

input double                    InpStateWeightZSlope           = 0.35;
input double                    InpStateWeightZAccel           = 0.25;
input double                    InpStateWeightShortVol         = 0.20;
input double                    InpStateWeightTrendPersistence = 0.20;
input double                    InpStateTierHighMinScore       = 0.15;
input double                    InpStateTierMediumMinScore     = -0.05;
input double                    InpStateHighMultiplier         = 1.50;
input double                    InpStateMediumMultiplier       = 1.00;
input double                    InpStateLowMultiplier          = 0.75;

input bool                      InpBasketExitOnlyIfProfitable  = true;
input ENUM_STATE_TIER           InpProfitOnlyMinStateTier      = STATE_TIER_HIGH;

CTrade trade;

int g_ema_fast_handle = INVALID_HANDLE;
int g_ema_slow_handle = INVALID_HANDLE;
int g_atr_handle      = INVALID_HANDLE;
datetime g_last_bar_time = 0;
bool g_initialized = false;

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

double ClampMin(const double value, const double minimum)
{
   return (value < minimum ? minimum : value);
}

bool IsFiniteNumber(const double value)
{
   return (value == value && value > -DBL_MAX && value < DBL_MAX);
}

string ToLowerText(const string value)
{
   string text = value;
   StringToLower(text);
   return text;
}

double NormalizePrice(const double price)
{
   return NormalizeDouble(price, (int)SymbolInfoInteger(_Symbol, SYMBOL_DIGITS));
}

double StopLevelDistance()
{
   return (double)SymbolInfoInteger(_Symbol, SYMBOL_TRADE_STOPS_LEVEL) * SymbolInfoDouble(_Symbol, SYMBOL_POINT);
}

double FloorToStep(const double value, const double step)
{
   if(step <= 0.0)
      return value;
   return MathFloor(value / step) * step;
}

double NormalizeVolumeToResearchConstraints(const double requested_volume)
{
   const double symbol_step = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
   const double symbol_min = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   const double symbol_max = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);

   const double step = MathMax(InpLotStep, symbol_step);
   const double min_volume = MathMax(InpMinLot, symbol_min);
   const double max_volume = MathMin(InpMaxLot, symbol_max);
   if(requested_volume <= 0.0 || step <= 0.0 || min_volume <= 0.0 || max_volume <= 0.0 || min_volume > max_volume)
      return 0.0;

   double volume = FloorToStep(requested_volume, step);
   if(volume < min_volume)
      return 0.0;
   if(volume > max_volume)
      volume = FloorToStep(max_volume, step);

   return NormalizeDouble(volume, 2);
}

double CalculateRiskSizedVolume(const double risk_money, const double entry_price, const double stop_price)
{
   const double tick_size = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);
   const double tick_value = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
   if(risk_money <= 0.0 || tick_size <= 0.0 || tick_value <= 0.0)
      return 0.0;

   const double stop_distance = MathAbs(entry_price - stop_price);
   if(stop_distance <= 0.0)
      return 0.0;

   const double money_per_lot = (stop_distance / tick_size) * tick_value;
   if(money_per_lot <= 0.0)
      return 0.0;

   const double raw_volume = risk_money / money_per_lot;
   return NormalizeVolumeToResearchConstraints(raw_volume);
}

double TierMultiplierFromRank(const int rank)
{
   if(rank >= (int)STATE_TIER_HIGH)
      return InpStateHighMultiplier;
   if(rank >= (int)STATE_TIER_MEDIUM)
      return InpStateMediumMultiplier;
   return InpStateLowMultiplier;
}

double ResolveEntryVolume(const double entry_price, const double stop_price, const double size_multiplier)
{
   if(InpPositionSizingMode == POSITION_SIZE_FIXED_LOTS)
      return NormalizeVolumeToResearchConstraints(InpFixedLots * MathMax(size_multiplier, 0.0));

   const double equity = AccountInfoDouble(ACCOUNT_EQUITY);
   const double risk_money = equity * InpRiskPercentPerTrade / 100.0 * MathMax(size_multiplier, 0.0);
   return CalculateRiskSizedVolume(risk_money, entry_price, stop_price);
}

int CountPositionsByMagic(const string symbol, const long magic, const ENUM_POSITION_TYPE position_type)
{
   int count = 0;
   for(int i = PositionsTotal() - 1; i >= 0; --i)
   {
      const ulong ticket = PositionGetTicket(i);
      if(ticket == 0 || !PositionSelectByTicket(ticket))
         continue;
      if(PositionGetString(POSITION_SYMBOL) != symbol)
         continue;
      if((long)PositionGetInteger(POSITION_MAGIC) != magic)
         continue;
      if((ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE) != position_type)
         continue;
      ++count;
   }
   return count;
}

bool CopyIndicatorSeries(const int handle, const int count, double &buffer[])
{
   ArrayResize(buffer, count);
   ArraySetAsSeries(buffer, true);
   return (CopyBuffer(handle, 0, 1, count, buffer) == count);
}

bool CopyRatesSeries(const int count, MqlRates &rates[])
{
   ArrayResize(rates, count);
   ArraySetAsSeries(rates, true);
   return (CopyRates(_Symbol, _Period, 1, count, rates) == count);
}

bool EwmStdSeries(const double &values[], const int count, const int span, double &out_std[])
{
   if(span < 2 || count < 2)
      return false;

   ArrayResize(out_std, count);
   ArraySetAsSeries(out_std, true);

   const double alpha = 2.0 / (span + 1.0);
   double xr[];
   double sr[];
   ArrayResize(xr, count);
   ArrayResize(sr, count);

   for(int i = 0; i < count; ++i)
      xr[i] = values[count - 1 - i];

   double mean = xr[0];
   double variance = 0.0;
   sr[0] = 0.0;

   for(int i = 1; i < count; ++i)
   {
      mean = alpha * xr[i] + (1.0 - alpha) * mean;
      const double diff = xr[i] - mean;
      variance = alpha * diff * diff + (1.0 - alpha) * variance;
      if(variance < 0.0)
         variance = 0.0;
      sr[i] = MathSqrt(variance);
   }

   for(int i = 0; i < count; ++i)
      out_std[i] = sr[count - 1 - i];

   return true;
}

int RegimeOf(const double vol, const double q33, const double q66)
{
   if(vol <= q33)
      return 0;
   if(vol <= q66)
      return 1;
   return 2;
}

double QuantileFromSorted(const double &sorted_values[], const int count, const double q)
{
   if(count <= 0)
      return 0.0;
   const int idx = (int)MathFloor(q * (count - 1));
   return sorted_values[MathMax(0, MathMin(count - 1, idx))];
}

double CurrentATRStopDistance()
{
   double atr_vals[];
   if(!CopyIndicatorSeries(g_atr_handle, 2, atr_vals))
      return 0.0;

   double distance = atr_vals[0] * InpStopATRMultiple;
   if(distance <= 0.0)
      return 0.0;

   distance = MathMax(distance, InpStopMinPrice);
   distance = MathMin(distance, InpStopMaxPrice);
   return distance;
}

double ResolveStopDistancePrice()
{
   if(!InpUseStopLoss)
      return 0.0;

   if(InpStopMode == STOP_FIXED_POINTS)
      return (double)InpStopLossPoints * _Point;

   return CurrentATRStopDistance();
}

double BuildStopLossPrice(const double ask_price)
{
   const double stop_distance = ResolveStopDistancePrice();
   if(stop_distance <= 0.0)
      return 0.0;

   double sl = ask_price - stop_distance;
   const double min_gap = StopLevelDistance();
   if(sl >= ask_price - min_gap)
      sl = ask_price - min_gap;

   sl = NormalizePrice(sl);
   if(sl <= 0.0 || sl >= ask_price)
      return 0.0;

   return sl;
}

double TradeRiskMoney(const double volume, const double entry_price, const double stop_price)
{
   if(volume <= 0.0 || stop_price <= 0.0 || stop_price >= entry_price)
      return 0.0;

   double pnl = 0.0;
   if(!OrderCalcProfit(ORDER_TYPE_BUY, _Symbol, volume, entry_price, stop_price, pnl))
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
      if((long)PositionGetInteger(POSITION_MAGIC) != (long)InpMagic)
         continue;
      if((ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE) != POSITION_TYPE_BUY)
         continue;

      const double volume = PositionGetDouble(POSITION_VOLUME);
      const double entry = PositionGetDouble(POSITION_PRICE_OPEN);
      const double sl = PositionGetDouble(POSITION_SL);
      risk += TradeRiskMoney(volume, entry, sl);
   }
   return risk;
}

bool BasketRiskAllows(const double new_trade_risk)
{
   if(!InpUseBasketRiskCap || !InpUseStopLoss || InpBasketRiskMultiple <= 0.0)
      return true;
   if(new_trade_risk <= 0.0)
      return false;

   const double max_open_risk_money = new_trade_risk * InpBasketRiskMultiple;
   const double open_risk_money = OpenRiskMoney();
   return (open_risk_money + new_trade_risk <= max_open_risk_money + 1e-8);
}

int TierRankFromScore(const double state_score)
{
   if(state_score >= InpStateTierHighMinScore)
      return (int)STATE_TIER_HIGH;
   if(state_score >= InpStateTierMediumMinScore)
      return (int)STATE_TIER_MEDIUM;
   return (int)STATE_TIER_LOW;
}

string TierNameFromRank(const int rank)
{
   if(rank >= (int)STATE_TIER_HIGH)
      return "high";
   if(rank >= (int)STATE_TIER_MEDIUM)
      return "medium";
   return "low";
}

int TierRankFromComment(const string comment)
{
   string text = ToLowerText(comment);
   if(StringFind(text, "_high") >= 0)
      return (int)STATE_TIER_HIGH;
   if(StringFind(text, "_medium") >= 0)
      return (int)STATE_TIER_MEDIUM;
   return (int)STATE_TIER_LOW;
}

double MeanAndStdScore(const double current_value, const double &series[], const int count, double &std_out)
{
   double sum = 0.0;
   int n = 0;
   for(int i = 0; i < count; ++i)
   {
      if(!IsFiniteNumber(series[i]))
         continue;
      sum += series[i];
      ++n;
   }
   if(n <= 1)
   {
      std_out = 0.0;
      return 0.0;
   }

   const double mean = sum / (double)n;
   double var_sum = 0.0;
   for(int i = 0; i < count; ++i)
   {
      if(!IsFiniteNumber(series[i]))
         continue;
      const double diff = series[i] - mean;
      var_sum += diff * diff;
   }
   const double std = MathSqrt(var_sum / (double)n);
   std_out = std;
   if(std <= 1e-12)
      return 0.0;

   double score = (current_value - mean) / std;
   score = MathMax(-3.0, MathMin(3.0, score));
   return score / 3.0;
}

double WeightedStateScore(
   const double z_slope0,
   const double z_accel0,
   const double short_vol0,
   const double trend_persistence0,
   const double &z_slope[],
   const double &z_accel[],
   const double &short_vol[],
   const double &trend_persistence[],
   const int count
)
{
   double total = 0.0;
   double total_abs_weight = 0.0;
   double std_dummy = 0.0;

   total += MeanAndStdScore(z_slope0, z_slope, count, std_dummy) * InpStateWeightZSlope;
   total_abs_weight += MathAbs(InpStateWeightZSlope);

   total += MeanAndStdScore(z_accel0, z_accel, count, std_dummy) * InpStateWeightZAccel;
   total_abs_weight += MathAbs(InpStateWeightZAccel);

   total += MeanAndStdScore(short_vol0, short_vol, count, std_dummy) * InpStateWeightShortVol;
   total_abs_weight += MathAbs(InpStateWeightShortVol);

   total += MeanAndStdScore(trend_persistence0, trend_persistence, count, std_dummy) * InpStateWeightTrendPersistence;
   total_abs_weight += MathAbs(InpStateWeightTrendPersistence);

   if(total_abs_weight <= 0.0)
      return 0.0;

   return MathMax(-1.0, MathMin(1.0, total / total_abs_weight));
}

bool ComputeSignals(bool &enter_long, bool &exit_long, int &state_tier_rank, double &state_score_out, double &z0, double &pct0)
{
   enter_long = false;
   exit_long = false;
   state_tier_rank = (int)STATE_TIER_LOW;
   state_score_out = 0.0;
   z0 = 0.0;
   pct0 = 100.0;

   const int slope_n = MathMax(1, InpA_ZSlopeN);
   const int need = MathMax(MathMax(InpRegimeLookback, InpPercentileLookback), 96) + InpA_EmaSlow + slope_n + 40;
   if(Bars(_Symbol, _Period) < need + 10)
      return false;

   MqlRates rates[];
   double ema_fast[];
   double ema_slow[];
   if(!CopyRatesSeries(need, rates))
      return false;
   if(!CopyIndicatorSeries(g_ema_fast_handle, need, ema_fast))
      return false;
   if(!CopyIndicatorSeries(g_ema_slow_handle, need, ema_slow))
      return false;

   double spread[];
   ArrayResize(spread, need);
   ArraySetAsSeries(spread, true);
   for(int i = 0; i < need; ++i)
      spread[i] = ema_fast[i] - ema_slow[i];

   double vol_std[];
   if(!EwmStdSeries(spread, need, InpA_StdSpan, vol_std))
      return false;

   double z[];
   double z_slope[];
   double z_accel[];
   double short_vol[];
   double trend_persistence[];
   double close_series[];
   ArrayResize(z, need);
   ArrayResize(z_slope, need);
   ArrayResize(z_accel, need);
   ArrayResize(short_vol, need);
   ArrayResize(trend_persistence, need);
   ArrayResize(close_series, need);
   ArraySetAsSeries(z, true);
   ArraySetAsSeries(z_slope, true);
   ArraySetAsSeries(z_accel, true);
   ArraySetAsSeries(short_vol, true);
   ArraySetAsSeries(trend_persistence, true);
   ArraySetAsSeries(close_series, true);

   for(int i = 0; i < need; ++i)
   {
      close_series[i] = rates[i].close;
      z[i] = spread[i] / ClampMin(vol_std[i], 1e-12);
   }

   for(int i = 0; i < need; ++i)
   {
      if(i + slope_n < need)
         z_slope[i] = (z[i] - z[i + slope_n]) / (double)slope_n;
      else
         z_slope[i] = 0.0;
   }

   for(int i = 0; i < need; ++i)
   {
      if(i + 1 < need)
         z_accel[i] = z_slope[i] - z_slope[i + 1];
      else
         z_accel[i] = 0.0;
   }

   for(int i = 0; i < need; ++i)
   {
      if(i + 12 < need)
      {
         double sum = 0.0;
         double sum_sq = 0.0;
         int n = 0;
         for(int k = i; k <= i + 11; ++k)
         {
            const double prev_close = close_series[k + 1];
            if(prev_close == 0.0)
               continue;
            const double ret = (close_series[k] / prev_close) - 1.0;
            sum += ret;
            sum_sq += ret * ret;
            ++n;
         }
         if(n > 1)
         {
            const double mean = sum / (double)n;
            const double var = MathMax(0.0, (sum_sq / (double)n) - mean * mean);
            short_vol[i] = MathSqrt(var);
         }
         else
            short_vol[i] = 0.0;
      }
      else
         short_vol[i] = 0.0;

      if(i + 20 < need)
      {
         const double direction = MathAbs(close_series[i] - close_series[i + 20]);
         double path = 0.0;
         for(int k = i; k < i + 20; ++k)
            path += MathAbs(close_series[k] - close_series[k + 1]);
         trend_persistence[i] = (path > 0.0 ? direction / path : 0.0);
      }
      else
         trend_persistence[i] = 0.0;
   }

   z0 = z[0];
   if(1 + slope_n >= need)
      return false;

   const double slope0 = z_slope[0];
   const double slope_prev = z_slope[1];
   const bool now_armed = (z[0] >= InpA_ZArm && slope0 >= InpA_Gamma);
   const bool prev_armed = (z[1] >= InpA_ZArm && slope_prev >= InpA_Gamma);

   if(!g_initialized && InpAllowFirstArmedEntry)
      enter_long = now_armed;
   else
      enter_long = (InpEdgeTrigger ? (now_armed && !prev_armed) : now_armed);

   g_initialized = true;

   const int regime_count = MathMin(InpRegimeLookback, need);
   double vols[];
   ArrayResize(vols, regime_count);
   for(int i = 0; i < regime_count; ++i)
      vols[i] = vol_std[i];
   ArraySort(vols);

   const double q33 = QuantileFromSorted(vols, regime_count, 0.33);
   const double q66 = QuantileFromSorted(vols, regime_count, 0.66);
   const int current_regime = RegimeOf(vol_std[0], q33, q66);

   int n = 0;
   int le = 0;
   const int limit = MathMin(InpPercentileLookback, need);
   for(int i = 0; i < limit; ++i)
   {
      if(RegimeOf(vol_std[i], q33, q66) != current_regime)
         continue;
      ++n;
      if(z[i] <= z0)
         ++le;
   }

   if(n < 30)
   {
      n = limit;
      le = 0;
      for(int i = 0; i < limit; ++i)
      {
         if(z[i] <= z0)
            ++le;
      }
   }

   pct0 = (n > 0 ? 100.0 * (double)le / (double)n : 100.0);
   exit_long = (pct0 <= InpExitPct);

   state_score_out = WeightedStateScore(
      z_slope[0],
      z_accel[0],
      short_vol[0],
      trend_persistence[0],
      z_slope,
      z_accel,
      short_vol,
      trend_persistence,
      need
   );
   state_tier_rank = TierRankFromScore(state_score_out);

   if(InpDebugPrint)
      PrintFormat(
         "SQ1: z=%.5f slope=%.5f enter=%d exit=%d pct=%.2f tier=%s score=%.4f",
         z0,
         slope0,
         (int)enter_long,
         (int)exit_long,
         pct0,
         TierNameFromRank(state_tier_rank),
         state_score_out
      );

   return true;
}

void ApplyExitLogic(const double bid_price)
{
   const int min_tier = (int)InpProfitOnlyMinStateTier;
   for(int i = PositionsTotal() - 1; i >= 0; --i)
   {
      const ulong ticket = PositionGetTicket(i);
      if(ticket == 0 || !PositionSelectByTicket(ticket))
         continue;
      if(PositionGetString(POSITION_SYMBOL) != _Symbol)
         continue;
      if((long)PositionGetInteger(POSITION_MAGIC) != (long)InpMagic)
         continue;
      if((ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE) != POSITION_TYPE_BUY)
         continue;

      const double entry_price = PositionGetDouble(POSITION_PRICE_OPEN);
      const string comment = PositionGetString(POSITION_COMMENT);
      const int entry_tier = TierRankFromComment(comment);

      bool should_close = true;
      if(InpBasketExitOnlyIfProfitable && bid_price <= entry_price && entry_tier >= min_tier)
         should_close = false;

      if(!should_close)
         continue;

      if(!trade.PositionClose(ticket))
      {
         if(InpDebugPrint)
            Print("Close failed ticket=", ticket, " retcode=", trade.ResultRetcode(), " err=", GetLastError());
      }
   }
}

int OnInit()
{
   trade.SetExpertMagicNumber((long)InpMagic);
   trade.SetDeviationInPoints(InpSlippagePoints);

   g_initialized = false;
   g_last_bar_time = iTime(_Symbol, _Period, 0);

   g_ema_fast_handle = iMA(_Symbol, _Period, InpA_EmaFast, 0, MODE_EMA, PRICE_CLOSE);
   g_ema_slow_handle = iMA(_Symbol, _Period, InpA_EmaSlow, 0, MODE_EMA, PRICE_CLOSE);
   g_atr_handle = iATR(_Symbol, _Period, InpStopATRPeriod);

   if(g_ema_fast_handle == INVALID_HANDLE || g_ema_slow_handle == INVALID_HANDLE || g_atr_handle == INVALID_HANDLE)
   {
      Print("Failed to create indicator handles");
      return INIT_FAILED;
   }

   return INIT_SUCCEEDED;
}

void OnDeinit(const int reason)
{
   if(g_ema_fast_handle != INVALID_HANDLE)
      IndicatorRelease(g_ema_fast_handle);
   if(g_ema_slow_handle != INVALID_HANDLE)
      IndicatorRelease(g_ema_slow_handle);
   if(g_atr_handle != INVALID_HANDLE)
      IndicatorRelease(g_atr_handle);
}

void OnTick()
{
   if(!IsNewBar())
      return;

   bool enter_long = false;
   bool exit_long = false;
   int state_tier_rank = (int)STATE_TIER_LOW;
   double state_score = 0.0;
   double z0 = 0.0;
   double pct0 = 100.0;
   if(!ComputeSignals(enter_long, exit_long, state_tier_rank, state_score, z0, pct0))
      return;

   MqlTick tick;
   if(!SymbolInfoTick(_Symbol, tick))
      return;

   if(exit_long)
      ApplyExitLogic(tick.bid);

   if(!enter_long)
      return;

   const int open_buys = CountPositionsByMagic(_Symbol, (long)InpMagic, POSITION_TYPE_BUY);
   if(open_buys >= InpMaxPositions)
      return;

   double sl = 0.0;
   if(InpUseStopLoss)
   {
      sl = BuildStopLossPrice(tick.ask);
      if(sl <= 0.0)
         return;
   }

   const double size_multiplier = TierMultiplierFromRank(state_tier_rank);
   const double volume = ResolveEntryVolume(tick.ask, sl, size_multiplier);
   if(volume <= 0.0)
   {
      if(InpDebugPrint)
         Print("Volume resolution blocked entry");
      return;
   }

   const double new_trade_risk = TradeRiskMoney(volume, tick.ask, sl);
   if(InpUseBasketRiskCap && !BasketRiskAllows(new_trade_risk))
   {
      if(InpDebugPrint)
         PrintFormat("Basket risk cap blocked entry. new_risk=%.2f open_risk=%.2f", new_trade_risk, OpenRiskMoney());
      return;
   }

   const string tier_name = TierNameFromRank(state_tier_rank);
   const string comment = StringFormat("XAUDZSQ1_%s_s%.2f", tier_name, state_score);
   if(!trade.Buy(volume, _Symbol, 0.0, sl, 0.0, comment))
   {
      if(InpDebugPrint)
         Print("BUY failed retcode=", trade.ResultRetcode(), " err=", GetLastError());
   }
}
