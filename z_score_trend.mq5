//+------------------------------------------------------------------+
//|                                                z_score_trend.mq5 |
//+------------------------------------------------------------------+
#property copyright "Joshua Grobler"
#property version   "2.00"
#property strict
#property description "EMA baseline + RETZCUM z trigger + phase-change z_slope gate + shared ATR risk engine + runner trail/exit"

#include <Trade/Trade.mqh>

enum ReturnMode
{
   RET_SIMPLE = 0,
   RET_LOG    = 1
};

enum StallMode
{
   STALL_BY_HIGH  = 0,
   STALL_BY_CLOSE = 1
};

enum ENUM_RISK_PROFILE
{
   RP_FIXED_LOT              = 0,
   RP_FIXED_FRACTIONAL       = 1,
   RP_FIXED_FRACTIONAL_DD    = 2,
   RP_FIXED_FRACTIONAL_HEAT  = 3,
   RP_STEPPED_LOT            = 4
};

// ---------- Trade ----------
input ulong      InpMagic                 = 3452389;
input int        InpSlippagePoints        = 5;
input bool       InpPrintSizingDebug      = false;

// ---------- Baseline (EMA filter) ----------
input int        InpEmaFast               = 21;
input int        InpEmaSlow               = 89;
input bool       InpRequireFastAboveSlow  = true;
input bool       InpRequireCloseAboveSlow = true;

// ---------- Trigger: RETZCUM3_W48_STD_K2 ----------
input int        InpCumBars               = 18;
input int        InpZWindow               = 48;
input double     InpEntryZ                = 0.4;
input ReturnMode InpReturnMode            = RET_SIMPLE;

// ---------- Phase change gate ----------
input int        InpZSlopeN               = 4;
input int        InpZSlopeGateN           = 3;
input double     InpZSlopeMin             = 0.25;
input double     InpGamma                 = 0.07;

// ---------- Runner policy ----------
input int        InpConfirmBars           = 3;
input int        InpStallBars             = 8;
input StallMode  InpStallMode             = STALL_BY_HIGH;

input double     InpArmR                  = 10.0;
input double     InpBaseDistR             = 4.0;
input double     InpMinDistR              = 2.0;
input double     InpContractionStepR      = 2.0;
input int        InpContractionCooldown   = 0;
input bool       InpMonotonicTightening   = true;

// ---------- Shared risk engine: stop construction ----------
input bool       InpUseStop               = true;    // required for shared risk engine
input int        InpAtrN                  = 14;
input double     InpStopAtrMult           = 2;
input double     InpMinStopPoints         = 10.0;   // safety min stop in points

// ---------- Shared risk engine: profile ----------
input ENUM_RISK_PROFILE InpRiskProfile    = RP_FIXED_FRACTIONAL;

// common sizing inputs
input bool       InpUseBalanceForSizing   = false;  // false = equity basis
input double     InpFixedLot              = 0.10;   // RP_FIXED_LOT
input double     InpRiskPct               = 0.5;  // 0.5% = 0.005

// drawdown throttle
input double     InpDD1Pct                = 0.05;   // 5%
input double     InpDD2Pct                = 0.10;   // 10%
input double     InpDDMult1               = 0.50;
input double     InpDDMult2               = 0.25;

// heat-cap profile
input double     InpMaxOpenRiskPct        = 0.02;   // 2% of basis
input bool       InpHeatAccountWide       = true;   // include all account positions

// stepped lot profile
input bool       InpUseEquitySteps        = true;
input double     InpStartLot              = 0.01;
input double     InpEquityStepUSD         = 200.0;
input bool       InpScaleDown             = true;

// optional starting anchor override
input double     InpStartEquityOverride   = 0.0;

CTrade trade;

int ema_fast_h = INVALID_HANDLE;
int ema_slow_h = INVALID_HANDLE;
int atr_h      = INVALID_HANDLE;

datetime g_lastBarTime   = 0;
int      g_barCounter    = 0;

// shared risk engine state
double g_peakEquity      = 0.0;
double g_anchorEquity    = 0.0;

// trade/run state
double g_entryPrice           = 0.0;
double g_Rdist                = 0.0;
double g_peakPrice            = 0.0;
int    g_barsInTrade          = 0;
int    g_barsSincePeak        = 0;

bool   g_armed                = false;
double g_trailDistR           = 0.0;
int    g_lastContractionBar   = -100000;

// -------------------------------------------------------
// utilities
// -------------------------------------------------------
double Clamp(const double x, const double lo, const double hi)
{
   if(x < lo) return lo;
   if(x > hi) return hi;
   return x;
}

bool IsNewBar()
{
   datetime t = iTime(_Symbol, _Period, 0);
   if(t == 0) return false;

   if(t != g_lastBarTime)
   {
      g_lastBarTime = t;
      return true;
   }
   return false;
}

double GetPoint()
{
   return SymbolInfoDouble(_Symbol, SYMBOL_POINT);
}

double NormalizePrice(const double p)
{
   int digits = (int)SymbolInfoInteger(_Symbol, SYMBOL_DIGITS);
   return NormalizeDouble(p, digits);
}

double MinLot()
{
   double v = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   return (v > 0.0 ? v : 0.01);
}

double MaxLot()
{
   double v = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);
   return (v > 0.0 ? v : 100.0);
}

double LotStep()
{
   double v = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
   return (v > 0.0 ? v : 0.01);
}

double NormalizeLotsClip(double vol)
{
   double mn   = MinLot();
   double mx   = MaxLot();
   double step = LotStep();

   if(vol <= 0.0) return 0.0;

   vol = Clamp(vol, mn, mx);
   double k = MathFloor((vol - mn) / step + 1e-12);
   double out = mn + k * step;

   if(out < mn) out = mn;
   if(out > mx) out = mx;

   return NormalizeDouble(out, 2);
}

double NormalizeLotsFloorRisk(double vol)
{
   double mn   = MinLot();
   double mx   = MaxLot();
   double step = LotStep();

   if(vol < mn) return 0.0;
   if(vol > mx) vol = mx;

   double k = MathFloor((vol - mn) / step + 1e-12);
   double out = mn + k * step;

   if(out < mn) return 0.0;
   if(out > mx) out = mx;

   return NormalizeDouble(out, 2);
}

bool SelectOurPosition(ulong &ticket, int &dir, double &vol)
{
   ticket = 0;
   dir    = 0;
   vol    = 0.0;

   for(int i = PositionsTotal() - 1; i >= 0; --i)
   {
      ulong t = PositionGetTicket(i);
      if(t == 0) continue;
      if(!PositionSelectByTicket(t)) continue;

      if(PositionGetString(POSITION_SYMBOL) != _Symbol) continue;
      if((ulong)PositionGetInteger(POSITION_MAGIC) != InpMagic) continue;

      long type = PositionGetInteger(POSITION_TYPE);
      dir = (type == POSITION_TYPE_BUY ? +1 : -1);
      vol = PositionGetDouble(POSITION_VOLUME);
      ticket = t;
      return true;
   }

   return false;
}

bool HasPosition()
{
   ulong ticket;
   int dir;
   double vol;
   return SelectOurPosition(ticket, dir, vol);
}

double GetPosSL()
{
   ulong ticket;
   int dir;
   double vol;
   if(!SelectOurPosition(ticket, dir, vol)) return 0.0;
   return PositionGetDouble(POSITION_SL);
}

double GetPosOpenPrice()
{
   ulong ticket;
   int dir;
   double vol;
   if(!SelectOurPosition(ticket, dir, vol)) return 0.0;
   return PositionGetDouble(POSITION_PRICE_OPEN);
}

double GetATR(const int shift)
{
   if(atr_h == INVALID_HANDLE) return 0.0;

   double buf[1];
   if(CopyBuffer(atr_h, 0, shift, 1, buf) != 1) return 0.0;
   return buf[0];
}

double GetEMA(const int handle, const int shift)
{
   if(handle == INVALID_HANDLE) return 0.0;

   double buf[1];
   if(CopyBuffer(handle, 0, shift, 1, buf) != 1) return 0.0;
   return buf[0];
}

// -------------------------------------------------------
// shared risk engine
// -------------------------------------------------------
double SizingBasis()
{
   return InpUseBalanceForSizing
        ? AccountInfoDouble(ACCOUNT_BALANCE)
        : AccountInfoDouble(ACCOUNT_EQUITY);
}

void UpdatePeakEquity()
{
   double eq = AccountInfoDouble(ACCOUNT_EQUITY);
   if(g_peakEquity <= 0.0 || eq > g_peakEquity)
      g_peakEquity = eq;
}

double CurrentDrawdownFrac()
{
   if(g_peakEquity <= 0.0) return 0.0;

   double eq = AccountInfoDouble(ACCOUNT_EQUITY);
   double dd = (g_peakEquity - eq) / g_peakEquity;
   return MathMax(0.0, dd);
}

double DrawdownMultiplier()
{
   double dd = CurrentDrawdownFrac();
   if(dd >= InpDD2Pct) return InpDDMult2;
   if(dd >= InpDD1Pct) return InpDDMult1;
   return 1.0;
}

double TickSizeValue()
{
   double v = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);
   if(v <= 0.0) v = _Point;
   return v;
}

double TickValueLoss()
{
   double v = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE_LOSS);
   if(v <= 0.0) v = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
   if(v <= 0.0) v = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE_PROFIT);
   return v;
}

double CashRiskPerLot(const double stop_dist_price)
{
   if(stop_dist_price <= 0.0) return 0.0;

   double tick_size  = TickSizeValue();
   double tick_value = TickValueLoss();

   if(tick_size <= 0.0 || tick_value <= 0.0) return 0.0;

   return (stop_dist_price / tick_size) * tick_value;
}

double BrokerMinStopDistancePrice()
{
   long stops_level_pts = SymbolInfoInteger(_Symbol, SYMBOL_TRADE_STOPS_LEVEL);
   if(stops_level_pts < 0) stops_level_pts = 0;
   return (double)stops_level_pts * _Point;
}

double ComputeSteppedLotProfile()
{
   double base = NormalizeLotsClip(MathMax(InpStartLot, MinLot()));

   if(!InpUseEquitySteps || InpEquityStepUSD <= 0.0)
      return base;

   double eq   = AccountInfoDouble(ACCOUNT_EQUITY);
   double diff = eq - g_anchorEquity;

   if(!InpScaleDown && diff < 0.0)
      diff = 0.0;

   double steps = MathFloor(diff / InpEquityStepUSD);
   double vol   = base + steps * LotStep();

   return NormalizeLotsClip(vol);
}

double CurrentOpenRiskDollars()
{
   double total = 0.0;

   for(int i = PositionsTotal() - 1; i >= 0; --i)
   {
      ulong ticket = PositionGetTicket(i);
      if(ticket == 0) continue;
      if(!PositionSelectByTicket(ticket)) continue;

      string sym = PositionGetString(POSITION_SYMBOL);
      if(!InpHeatAccountWide && sym != _Symbol)
         continue;

      double entry = PositionGetDouble(POSITION_PRICE_OPEN);
      double sl    = PositionGetDouble(POSITION_SL);
      double vol   = PositionGetDouble(POSITION_VOLUME);

      if(entry <= 0.0 || sl <= 0.0 || vol <= 0.0)
         continue;

      double dist = MathAbs(entry - sl);
      double one_lot_risk = CashRiskPerLot(dist);

      total += one_lot_risk * vol;
   }

   return total;
}

bool BuildOrderPricesLong(double &entry_price, double &stop_price, double &stop_dist_price)
{
   entry_price     = 0.0;
   stop_price      = 0.0;
   stop_dist_price = 0.0;

   if(!InpUseStop)
      return false;

   MqlTick tick;
   if(!SymbolInfoTick(_Symbol, tick))
      return false;

   double atr = GetATR(1);
   if(atr <= 0.0)
      return false;

   double min_dist = MathMax((double)InpMinStopPoints * _Point, BrokerMinStopDistancePrice());
   stop_dist_price = MathMax(atr * InpStopAtrMult, min_dist);

   entry_price = tick.ask;
   if(entry_price <= 0.0) return false;

   stop_price = entry_price - stop_dist_price;

   entry_price = NormalizePrice(entry_price);
   stop_price  = NormalizePrice(stop_price);

   return (stop_price > 0.0);
}

double ComputeVolumeByRiskProfile(const double stop_dist_price, bool &allowed, double &target_risk_dollars)
{
   allowed = true;
   target_risk_dollars = 0.0;

   if(InpRiskProfile == RP_FIXED_LOT)
      return NormalizeLotsClip(InpFixedLot);

   if(InpRiskProfile == RP_STEPPED_LOT)
      return ComputeSteppedLotProfile();

   double basis = SizingBasis();
   if(basis <= 0.0 || stop_dist_price <= 0.0)
   {
      allowed = false;
      return 0.0;
   }

   double one_lot_risk = CashRiskPerLot(stop_dist_price);
   if(one_lot_risk <= 0.0)
   {
      allowed = false;
      return 0.0;
   }

   double risk_pct = InpRiskPct;

   if(InpRiskProfile == RP_FIXED_FRACTIONAL_DD)
      risk_pct *= DrawdownMultiplier();

   target_risk_dollars = basis * risk_pct;

   if(InpRiskProfile == RP_FIXED_FRACTIONAL_HEAT)
   {
      double open_risk = CurrentOpenRiskDollars();
      double max_risk  = basis * InpMaxOpenRiskPct;
      double remain    = max_risk - open_risk;

      if(remain <= 0.0)
      {
         allowed = false;
         return 0.0;
      }

      if(target_risk_dollars > remain)
         target_risk_dollars = remain;
   }

   if(target_risk_dollars <= 0.0)
   {
      allowed = false;
      return 0.0;
   }

   double raw_lots = target_risk_dollars / one_lot_risk;
   double lots     = NormalizeLotsFloorRisk(raw_lots);

   if(lots <= 0.0)
   {
      allowed = false;
      return 0.0;
   }

   return lots;
}

bool EnsurePositionStop()
{
   ulong ticket;
   int dir;
   double vol;
   if(!SelectOurPosition(ticket, dir, vol)) return false;

   double sl = PositionGetDouble(POSITION_SL);
   if(sl > 0.0) return true;

   if(!InpUseStop) return false;

   double atr = GetATR(1);
   if(atr <= 0.0) return false;

   double dist = MathMax(atr * InpStopAtrMult,
                  MathMax((double)InpMinStopPoints * _Point, BrokerMinStopDistancePrice()));

   double priceOpen = PositionGetDouble(POSITION_PRICE_OPEN);
   double newSL = (dir > 0) ? (priceOpen - dist) : (priceOpen + dist);
   newSL = NormalizePrice(newSL);

   return trade.PositionModify(ticket, newSL, 0.0);
}

// -------------------------------------------------------
// signal computations
// -------------------------------------------------------
double CumReturn(const double &cl[], const int shift, const int cumBars, const ReturnMode mode)
{
   int i0 = shift;
   int i1 = shift + cumBars;
   if(i1 >= ArraySize(cl)) return 0.0;

   double c0 = cl[i0];
   double c1 = cl[i1];
   if(c0 <= 0.0 || c1 <= 0.0) return 0.0;

   if(mode == RET_LOG)
      return MathLog(c0 / c1);

   return (c0 / c1) - 1.0;
}

bool ZScoreCumReturn(const int shift, const int cumBars, const int win, const ReturnMode mode, double &z_out)
{
   int need = shift + cumBars + win + 5;

   double cl[];
   ArraySetAsSeries(cl, true);
   if(CopyClose(_Symbol, _Period, 0, need, cl) < need) return false;

   double r0 = CumReturn(cl, shift, cumBars, mode);

   double sum = 0.0;
   double sumsq = 0.0;

   for(int i = 1; i <= win; i++)
   {
      double ri = CumReturn(cl, shift + i, cumBars, mode);
      sum   += ri;
      sumsq += ri * ri;
   }

   double mean = sum / win;
   double var  = (sumsq / win) - (mean * mean);
   if(var <= 1e-12)
   {
      z_out = 0.0;
      return true;
   }

   z_out = (r0 - mean) / MathSqrt(var);
   return true;
}

bool ZSlope(const int shift, const int zSlopeN, const int cumBars, const int win, const ReturnMode mode, double &slope_out)
{
   double z_now  = 0.0;
   double z_past = 0.0;

   if(!ZScoreCumReturn(shift,          cumBars, win, mode, z_now))  return false;
   if(!ZScoreCumReturn(shift+zSlopeN,  cumBars, win, mode, z_past)) return false;

   slope_out = (z_now - z_past) / (double)zSlopeN;
   return true;
}

bool BaselineOK()
{
   double emaF = GetEMA(ema_fast_h, 1);
   double emaS = GetEMA(ema_slow_h, 1);
   double cl1  = iClose(_Symbol, _Period, 1);

   if(InpRequireFastAboveSlow && !(emaF > emaS)) return false;
   if(InpRequireCloseAboveSlow && !(cl1 > emaS)) return false;

   return true;
}

bool PhaseChangeGateOK()
{
   double alpha = Clamp(InpGamma, 0.0, 1.0);
   if(alpha <= 0.0) alpha = 1.0;

   bool init = false;
   double ewma = 0.0;

   for(int k = InpZSlopeGateN; k >= 1; --k)
   {
      double s = 0.0;
      if(!ZSlope(k, InpZSlopeN, InpCumBars, InpZWindow, InpReturnMode, s))
         return false;

      if(!init)
      {
         ewma = s;
         init = true;
      }
      else
      {
         ewma = alpha * s + (1.0 - alpha) * ewma;
      }

      if(ewma <= InpZSlopeMin)
         return false;
   }

   return init;
}

bool TriggerCrossUp(double &z_now_out)
{
   double z_now  = 0.0;
   double z_prev = 0.0;

   if(!ZScoreCumReturn(1, InpCumBars, InpZWindow, InpReturnMode, z_now))  return false;
   if(!ZScoreCumReturn(2, InpCumBars, InpZWindow, InpReturnMode, z_prev)) return false;

   z_now_out = z_now;
   return (z_prev <= InpEntryZ && z_now > InpEntryZ);
}

// -------------------------------------------------------
// runner / exits
// -------------------------------------------------------
double GetFavorablePrice(const int shift)
{
   if(InpStallMode == STALL_BY_CLOSE)
      return iClose(_Symbol, _Period, shift);

   return iHigh(_Symbol, _Period, shift);
}

void ResetState()
{
   g_entryPrice         = 0.0;
   g_Rdist              = 0.0;
   g_peakPrice          = 0.0;
   g_barsInTrade        = 0;
   g_barsSincePeak      = 0;
   g_armed              = false;
   g_trailDistR         = 0.0;
   g_lastContractionBar = -100000;
}

void SyncStateFromPosition()
{
   ulong ticket;
   int dir;
   double vol;

   if(!SelectOurPosition(ticket, dir, vol))
   {
      ResetState();
      return;
   }

   if(g_entryPrice != 0.0)
      return;

   g_entryPrice = PositionGetDouble(POSITION_PRICE_OPEN);

   double sl = PositionGetDouble(POSITION_SL);
   if(sl > 0.0)
   {
      g_Rdist = MathAbs(g_entryPrice - sl);
   }
   else
   {
      double atr = GetATR(1);
      double dist = MathMax(atr * InpStopAtrMult,
                     MathMax((double)InpMinStopPoints * _Point, BrokerMinStopDistancePrice()));
      g_Rdist = dist;
   }

   g_peakPrice          = g_entryPrice;
   g_barsInTrade        = 0;
   g_barsSincePeak      = 0;
   g_armed              = false;
   g_trailDistR         = 0.0;
   g_lastContractionBar = -100000;
}

void UpdateStallTracking()
{
   double fav = GetFavorablePrice(1);

   if(fav > g_peakPrice + 1e-12)
   {
      g_peakPrice = fav;
      g_barsSincePeak = 0;
   }
   else
   {
      g_barsSincePeak++;
   }
}

bool StallTriggered()
{
   return (InpStallBars > 0 && g_barsSincePeak >= InpStallBars);
}

void MaybeTightenTrail()
{
   if(!g_armed) return;
   if(InpContractionStepR <= 0.0) return;
   if(g_trailDistR <= InpMinDistR + 1e-12) return;

   if((g_barCounter - g_lastContractionBar) < InpContractionCooldown)
      return;

   if(StallTriggered())
   {
      g_trailDistR = MathMax(InpMinDistR, g_trailDistR - InpContractionStepR);
      g_lastContractionBar = g_barCounter;
   }
}

void UpdateTrailingStop()
{
   if(!InpUseStop) return;
   if(g_Rdist <= 0.0) return;

   ulong ticket;
   int dir;
   double vol;
   if(!SelectOurPosition(ticket, dir, vol)) return;

   double mfe_r = (g_peakPrice - g_entryPrice) / g_Rdist;
   if(!g_armed && mfe_r >= InpArmR)
   {
      g_armed = true;
      g_trailDistR = InpBaseDistR;
   }

   if(!g_armed) return;

   double new_sl = g_peakPrice - g_trailDistR * g_Rdist;
   double cur_sl = PositionGetDouble(POSITION_SL);

   if(cur_sl <= 0.0)
      cur_sl = g_entryPrice - g_Rdist;

   if(InpMonotonicTightening)
      new_sl = MathMax(new_sl, cur_sl);

   double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   double max_sl = bid - MathMax(5.0 * GetPoint(), BrokerMinStopDistancePrice());
   if(new_sl > max_sl)
      new_sl = max_sl;

   new_sl = NormalizePrice(new_sl);

   if(new_sl > 0.0 && new_sl > cur_sl + 0.5 * _Point)
      trade.PositionModify(ticket, new_sl, 0.0);
}

void MaybeExit()
{
   if(g_barsInTrade < InpConfirmBars) return;
   if(!StallTriggered()) return;

   ulong ticket;
   int dir;
   double vol;
   if(!SelectOurPosition(ticket, dir, vol)) return;

   if(trade.PositionClose(ticket, InpSlippagePoints))
      ResetState();
}

void ManagePosition()
{
   EnsurePositionStop();

   g_barsInTrade++;
   UpdateStallTracking();
   MaybeTightenTrail();
   UpdateTrailingStop();
   MaybeExit();
}

// -------------------------------------------------------
// entry / orders
// -------------------------------------------------------
bool OpenLong()
{
   if(HasPosition())
      return false;

   double entry_price = 0.0;
   double stop_price  = 0.0;
   double stop_dist   = 0.0;

   if(!BuildOrderPricesLong(entry_price, stop_price, stop_dist))
      return false;

   bool allowed = false;
   double target_risk_dollars = 0.0;
   double lots = ComputeVolumeByRiskProfile(stop_dist, allowed, target_risk_dollars);

   if(!allowed || lots <= 0.0)
   {
      if(InpPrintSizingDebug)
         PrintFormat("ENTRY BLOCKED | profile=%d | targetRisk=%.2f | stopDist=%.5f",
                     (int)InpRiskProfile, target_risk_dollars, stop_dist);
      return false;
   }

   bool ok = trade.Buy(lots, _Symbol, 0.0, stop_price, 0.0, "z_score_trend");
   if(!ok)
   {
      if(InpPrintSizingDebug)
         PrintFormat("ORDER FAILED | vol=%.2f | sl=%.5f | err=%d",
                     lots, stop_price, GetLastError());
      return false;
   }

   g_entryPrice = 0.0;
   SyncStateFromPosition();

   if(InpPrintSizingDebug)
      PrintFormat("ENTRY OK | profile=%d | vol=%.2f | targetRisk=%.2f | stopDist=%.5f | DD=%.2f%% | openRisk=%.2f",
                  (int)InpRiskProfile,
                  lots,
                  target_risk_dollars,
                  stop_dist,
                  100.0 * CurrentDrawdownFrac(),
                  CurrentOpenRiskDollars());

   return true;
}

void TryEnter()
{
   if(!BaselineOK()) return;

   double z_now = 0.0;
   if(!TriggerCrossUp(z_now)) return;

   if(!PhaseChangeGateOK()) return;

   OpenLong();
}

// -------------------------------------------------------
// lifecycle
// -------------------------------------------------------
int OnInit()
{
   trade.SetExpertMagicNumber(InpMagic);
   trade.SetDeviationInPoints(InpSlippagePoints);

   ema_fast_h = iMA(_Symbol, _Period, InpEmaFast, 0, MODE_EMA, PRICE_CLOSE);
   ema_slow_h = iMA(_Symbol, _Period, InpEmaSlow, 0, MODE_EMA, PRICE_CLOSE);
   atr_h      = iATR(_Symbol, _Period, InpAtrN);

   if(ema_fast_h == INVALID_HANDLE || ema_slow_h == INVALID_HANDLE || atr_h == INVALID_HANDLE)
      return INIT_FAILED;

   double start = (InpStartEquityOverride > 0.0)
                ? InpStartEquityOverride
                : AccountInfoDouble(ACCOUNT_EQUITY);

   g_anchorEquity = start;
   g_peakEquity   = start;

   g_lastBarTime = iTime(_Symbol, _Period, 0);
   g_barCounter  = 0;

   ResetState();
   SyncStateFromPosition();

   return INIT_SUCCEEDED;
}

void OnDeinit(const int reason)
{
   if(ema_fast_h != INVALID_HANDLE) IndicatorRelease(ema_fast_h);
   if(ema_slow_h != INVALID_HANDLE) IndicatorRelease(ema_slow_h);
   if(atr_h      != INVALID_HANDLE) IndicatorRelease(atr_h);
}

void OnTick()
{
   UpdatePeakEquity();

   if(!IsNewBar()) return;

   g_barCounter++;
   SyncStateFromPosition();

   if(HasPosition())
      ManagePosition();
   else
      TryEnter();
}