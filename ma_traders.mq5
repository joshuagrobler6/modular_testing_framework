#property strict

#include <Trade/Trade.mqh>

enum ENUM_RUN_MODE
{
  RUN_A = 0,
  RUN_B = 1,
  RUN_C = 2
};

enum ENUM_PULLVEL_WINDOW
{
  PULLVEL_W24 = 24,
  PULLVEL_W48 = 48
};

input ENUM_RUN_MODE        InpRunMode              = RUN_C;

// Execution model
input bool                 InpNextOpenExecution    = true;   // entries/exits fire on bar close, execute next bar open

// Lot scaling (your stepped equity sizing)
input bool                 InpUseEquitySteps       = true;
input double               InpStartLot             = 0.01;    // user-defined minimum lot to start at
input double               InpEquityStepUSD        = 200.0;   // +1 volume step per $200
input bool                 InpScaleDown            = true;    // reduce size if equity falls
input ulong                InpMagicBase            = 345000;

// Risk (all three)
input int                  InpATR_N                = 14;
input double               InpStopATRMult          = 1.;
input double               InpMinStopPoints        = 10;      // safety min (points); set 0 to disable

// Arm
input double               InpArmMinPeakPnL_R      = 3.0;

// Gates (HVOL + age)
input int                  InpAgeBarsSinceMFE_GTE  = 12;
input int                  InpVolRegimeWindow      = 96;

// Gate (WHIP)
input int                  InpWhipsawWindow        = 48;
input double               InpWhipsawRatioThresh   = 3.0;     // proxy: sum(|ret|)/|net_move|

// Detector params (A/C: return_z_shock_cum)
input int                  InpRetZ_Window          = 448;
input double               InpRetZ_K               = 2.0;
input int                  InpRetZ_CumHorizon_A    = 2;
input int                  InpRetZ_CumHorizon_C    = 3;

// Detector params (B: pullback_vel_z)
input ENUM_PULLVEL_WINDOW  InpPullVelWindow        = PULLVEL_W48;
input double               InpPullVelK             = 3.0;

// === Strategy params from your sheet ===

// A/B entry: ma_p5
input int                  InpA_FastN              = 20;
input int                  InpA_MidN               = 50;
input int                  InpA_SlowN              = 200;
input int                  InpA_NRed               = 3;

// C entry: ma_m1
input int                  InpC_MidN               = 48;
input int                  InpC_SlowN              = 144;
input int                  InpC_VolSpan            = 48;      // EWMA std span
input double               InpC_ZEntry             = 1.25;


enum ENUM_RISK_PROFILE
{
  RP_FIXED_LOT = 0,
  RP_FIXED_FRACTIONAL = 1,
  RP_FIXED_FRACTIONAL_DD = 2,
  RP_FIXED_FRACTIONAL_HEAT = 3,
  RP_STEPPED_LOT = 4
};

input ENUM_RISK_PROFILE   InpRiskProfile          = RP_FIXED_FRACTIONAL;

// Common risk-engine inputs
input bool                InpUseBalanceForSizing  = false;   // false = equity basis
input double              InpFixedLot             = 0.01;    // profile RP_FIXED_LOT
input double              InpRiskPct              = 0.0025;  // 0.25% per trade for fractional profiles

// Drawdown throttle profile
input double              InpDD1Pct               = 0.05;    // 5%
input double              InpDD2Pct               = 0.10;    // 10%
input double              InpDDMult1              = 0.50;    // after DD1, risk *= 0.50
input double              InpDDMult2              = 0.25;    // after DD2, risk *= 0.25

// Heat-cap profile
input double              InpMaxOpenRiskPct       = 0.1;    // max total open risk = 1% of basis
input bool                InpHeatAccountWide      = true;    // true = include all account positions
// -----------------------------
double g_peakEquity   = 0.0;
double g_anchorEquity = 0.0;

CTrade trade;

datetime g_lastBarTime = 0;

// indicator handles
int hEmaSlow200 = INVALID_HANDLE;
int hEmaMid50   = INVALID_HANDLE;
int hEmaFast20  = INVALID_HANDLE;

int hEmaMid48   = INVALID_HANDLE;
int hEmaSlow144 = INVALID_HANDLE;

int hATR        = INVALID_HANDLE;

// State for next-open execution
int  g_pendingDir = 0;   // +1 long, -1 short
bool g_pending    = false;

// Edge-trigger memory (suppress consecutive duplicates)
bool g_prevSigA = false;
bool g_prevSigB = false;
bool g_prevSigC_L = false;
bool g_prevSigC_S = false;

// Rolling stats for return_z_shock_cum (A horizon=2, C horizon=3)
double g_retBufA[];
double g_retBufC[];
int    g_retIdxA = 0, g_retIdxC = 0;
int    g_retCountA = 0, g_retCountC = 0;
double g_retSumA = 0, g_retSumSqA = 0;
double g_retSumC = 0, g_retSumSqC = 0;

// Rolling stats for vol regime proxy (std96 vs std384)
double g_r96Buf[];
double g_r384Buf[];
int    g_r96Idx=0, g_r384Idx=0;
int    g_r96Count=0, g_r384Count=0;
double g_r96Sum=0, g_r96SumSq=0;
double g_r384Sum=0, g_r384SumSq=0;

// Position run-state
bool   g_armed = false;
double g_entryPrice = 0.0;
double g_Rdist = 0.0;
double g_bestFavorable = 0.0; // highest high (long) / lowest low (short)
int    g_barsSinceNewMFE = 0;

// Pullback velocity rolling stats (only meaningful in-trade)
double g_pvBuf[];
int    g_pvIdx=0, g_pvCount=0;
double g_pvSum=0, g_pvSumSq=0;

// EWMA std state for C entry residual = close - EMA(mid)
bool   g_ewmInit = false;
double g_ewmMean = 0.0;
double g_ewmVar  = 0.0;

double Clamp(const double x, const double lo, const double hi)
{
  if(x < lo) return lo;
  if(x > hi) return hi;
  return x;
}

int Sign(const double x)
{
  if(x > 0.0) return  1;
  if(x < 0.0) return -1;
  return 0;
}

bool IsNewBar()
{
  datetime t0 = (datetime)iTime(_Symbol, _Period, 0);
  if(t0 != g_lastBarTime)
  {
    g_lastBarTime = t0;
    return true;
  }
  return false;
}

bool GetMA(int handle, int shift, double &out)
{
  double buf[1];
  if(CopyBuffer(handle, 0, shift, 1, buf) != 1) return false;
  out = buf[0];
  return true;
}

bool GetATR(int shift, double &out)
{
  double buf[1];
  if(CopyBuffer(hATR, 0, shift, 1, buf) != 1) return false;
  out = buf[0];
  return true;
}

double StdFromSums(double sum, double sumsq, int n)
{
  if(n <= 1) return 0.0;
  double mean = sum / n;
  double var  = (sumsq / n) - (mean * mean);
  if(var < 0.0) var = 0.0;
  return MathSqrt(var);
}

void RollPush(double &sum, double &sumsq, double &buf[], int &idx, int &count, int win, double x)
{
  if(ArraySize(buf) != win) ArrayResize(buf, win);

  if(count < win)
  {
    buf[idx] = x;
    sum += x;
    sumsq += x*x;
    idx = (idx + 1) % win;
    count++;
    return;
  }

  double old = buf[idx];
  sum   += x - old;
  sumsq += x*x - old*old;
  buf[idx] = x;
  idx = (idx + 1) % win;
}

double LotStep()
{
  double step = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
  if(step <= 0.0) step = 0.01;
  return step;
}

double MinLot()
{
  double v = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
  if(v <= 0.0) v = 0.01;
  return v;
}

double MaxLot()
{
  double v = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);
  if(v <= 0.0) v = 100.0;
  return v;
}

double RoundToStep(double vol)
{
  double step = LotStep();
  double mn = MinLot();
  double mx = MaxLot();
  vol = Clamp(vol, mn, mx);
  double k = MathFloor((vol - mn) / step + 0.5);
  return Clamp(mn + k * step, mn, mx);
}

double ComputeVolume()
{
  double mn = MinLot();
  double base = MathMax(InpStartLot, mn);
  base = RoundToStep(base);

  if(!InpUseEquitySteps) return base;

  static bool inited = false;
  static double anchorEquity = 0.0;
  if(!inited)
  {
    anchorEquity = AccountInfoDouble(ACCOUNT_EQUITY);
    inited = true;
  }

  double eq = AccountInfoDouble(ACCOUNT_EQUITY);
  double diff = eq - anchorEquity;

  if(!InpScaleDown && diff < 0.0) diff = 0.0;

  double steps = (InpEquityStepUSD > 0.0) ? MathFloor(diff / InpEquityStepUSD) : 0.0;
  double vol = base + steps * LotStep();
  return RoundToStep(vol);
}

// --- Regime proxies (swap to exact harness versions if you paste them) ---

// =====================
// FIX: arrays by reference
// =====================

// =====================
// Exact ATR + regimes (from your engine.py)
// =====================

double g_atr = 0.0;
int    g_trCount = 0;      // number of TR observations processed
bool   g_atrValid = false; // true once trCount >= atr_n

// vol_regime[96] needs rolling quantiles on ATR with min_periods=96
double g_atrRing96[];
int    g_atrRingIdx96 = 0;
int    g_atrRingCount96 = 0;
string g_volRegime96 = ""; // "", "low","mid","high"

// market_regime[48] needs rolling mean of flips with min_periods=48
double g_flipSum48 = 0.0;
double g_flipSumSq48 = 0.0; // unused but keeps RollPush signature simple
double g_flipRing48[];
int    g_flipIdx48 = 0;
int    g_flipCount48 = 0;
string g_marketRegime48 = ""; // "", "trend","neutral","whipsaw"

double QuantileLinearSorted(const double &sorted[], int n, double p)
{
  if(n <= 0) return 0.0;
  if(n == 1) return sorted[0];

  if(p <= 0.0) return sorted[0];
  if(p >= 1.0) return sorted[n-1];

  double pos = (n - 1) * p;
  int lo = (int)MathFloor(pos);
  int hi = (int)MathCeil(pos);
  if(hi >= n) hi = n - 1;
  double frac = pos - lo;
  return sorted[lo] + frac * (sorted[hi] - sorted[lo]);
}

void CopyRingToArray(const double &ring[], int win, int idx, int count, double &out[])
{
  ArrayResize(out, count);
  // oldest element is at (idx - count) in ring coordinates
  for(int i=0;i<count;i++)
  {
    int pos = idx - count + i;
    while(pos < 0) pos += win;
    pos %= win;
    out[i] = ring[pos];
  }
}

double RingQuantile(const double &ring[], int win, int idx, int count, double p)
{
  if(count <= 0) return 0.0;
  double tmp[];
  CopyRingToArray(ring, win, idx, count, tmp);
  ArraySort(tmp);
  return QuantileLinearSorted(tmp, count, p);
}

void AtrAndRegimeUpdateOnClosedBar(int atr_n, int vol_w, int mr_w)
{
  // === ATR update on bar 1 (just closed), exact TR definition ===
  double hi1 = iHigh(_Symbol, _Period, 1);
  double lo1 = iLow(_Symbol, _Period, 1);
  double pc  = iClose(_Symbol, _Period, 2); // prev close of bar 1
  if(pc <= 0.0) return;

  double tr1 = MathMax(MathAbs(hi1 - lo1),
               MathMax(MathAbs(hi1 - pc), MathAbs(lo1 - pc)));

  double alpha = 1.0 / (double)MathMax(1, atr_n);
  if(g_trCount == 0)
    g_atr = tr1;
  else
    g_atr = alpha * tr1 + (1.0 - alpha) * g_atr;

  g_trCount++;
  g_atrValid = (g_trCount >= atr_n);

  // === vol_regime[96] based on rolling quantiles of ATR, min_periods=96 ===
  // Python uses pd.Series(atr_arr) which has NaNs until min_periods=atr_n.
  // We mimic that by only pushing ATR once it's valid.
  if(g_atrValid)
  {
    if(ArraySize(g_atrRing96) != vol_w) ArrayResize(g_atrRing96, vol_w);
    g_atrRing96[g_atrRingIdx96] = g_atr;
    g_atrRingIdx96 = (g_atrRingIdx96 + 1) % vol_w;
    if(g_atrRingCount96 < vol_w) g_atrRingCount96++;

    if(g_atrRingCount96 >= vol_w)
    {
      double q1 = RingQuantile(g_atrRing96, vol_w, g_atrRingIdx96, g_atrRingCount96, 1.0/3.0);
      double q2 = RingQuantile(g_atrRing96, vol_w, g_atrRingIdx96, g_atrRingCount96, 2.0/3.0);

      if(g_atr <= q1) g_volRegime96 = "low";
      else if(g_atr <= q2) g_volRegime96 = "mid";
      else g_volRegime96 = "high";
    }
    else
    {
      g_volRegime96 = ""; // min_periods not met
    }
  }
  else
  {
    g_volRegime96 = "";
  }

  // === market_regime[48] based on flip rate of sign(ret), ret = close - prev_close ===
  double c1 = iClose(_Symbol, _Period, 1);
  double c2 = iClose(_Symbol, _Period, 2);
  double c3 = iClose(_Symbol, _Period, 3);
  if(c1 <= 0.0 || c2 <= 0.0 || c3 <= 0.0)
    return;

  double r1 = c1 - c2;
  double r2 = c2 - c3;

  int s1 = (r1 > 0.0) ? 1 : (r1 < 0.0 ? -1 : 0);
  int s2 = (r2 > 0.0) ? 1 : (r2 < 0.0 ? -1 : 0);

  double flip = ((s1 * s2) < 0) ? 1.0 : 0.0; // matches (s * roll(s,1) < 0)

  // rolling mean with min_periods=mr_w
  if(mr_w < 2) mr_w = 2;
  // use RollPush just to maintain sum easily
  if(ArraySize(g_flipRing48) != mr_w) ArrayResize(g_flipRing48, mr_w);

  // manual rolling sum (simpler than RollPush because we only need sum)
  if(g_flipCount48 < mr_w)
  {
    g_flipRing48[g_flipIdx48] = flip;
    g_flipSum48 += flip;
    g_flipIdx48 = (g_flipIdx48 + 1) % mr_w;
    g_flipCount48++;
  }
  else
  {
    double old = g_flipRing48[g_flipIdx48];
    g_flipSum48 += flip - old;
    g_flipRing48[g_flipIdx48] = flip;
    g_flipIdx48 = (g_flipIdx48 + 1) % mr_w;
  }

  if(g_flipCount48 >= mr_w)
  {
    double fr = g_flipSum48 / (double)mr_w;
    if(fr >= 0.60) g_marketRegime48 = "whipsaw";
    else if(fr <= 0.45) g_marketRegime48 = "trend";
    else g_marketRegime48 = "neutral";
  }
  else
  {
    g_marketRegime48 = "";
  }
}

// Replace your old gate functions with:
bool Gate_HVOL_STALL8()
{
  if(g_barsSinceNewMFE < InpAgeBarsSinceMFE_GTE) return false;
  return (g_volRegime96 == "high");
}

bool Gate_WHIP()
{
  return (g_marketRegime48 == "whipsaw");
}

// Replace any GetATR/iATR usage in stops with this:
bool GetAtrForStop(double &atr_out)
{
  if(!g_atrValid) return false;
  atr_out = g_atr;
  return (atr_out > 0.0);
}
// --- Detectors ---

double RetZ_ShockCum_Z(int cumHorizon, int rollWin, double &outCum)
{
  outCum = 0.0;
  if(cumHorizon < 1) return 0.0;

  double c0 = iClose(_Symbol, _Period, 1);
  double cH = iClose(_Symbol, _Period, 1 + cumHorizon);
  if(c0 <= 0.0 || cH <= 0.0) return 0.0;

  // raw cumulative return
  double cumRet = (c0 / cH) - 1.0;
  outCum = cumRet;

  // z uses rolling std of cumRet series (maintained externally)
  return 0.0;
}

bool PositionIsOpen(int &dir, double &vol)
{
  if(!PositionSelect(_Symbol)) return false;
  long type = PositionGetInteger(POSITION_TYPE);
  dir = (type == POSITION_TYPE_BUY) ? +1 : -1;
  vol = PositionGetDouble(POSITION_VOLUME);
  return true;
}

void ResetRunState()
{
  g_armed = false;
  g_entryPrice = 0.0;
  g_Rdist = 0.0;
  g_bestFavorable = 0.0;
  g_barsSinceNewMFE = 0;

  ArrayResize(g_pvBuf, 0);
  g_pvIdx=0; g_pvCount=0; g_pvSum=0; g_pvSumSq=0;
}



void UpdateMFEStateOnClosedBar()
{
  int dir; double vol;
  if(!PositionIsOpen(dir, vol)) return;

  double hi = iHigh(_Symbol, _Period, 1);
  double lo = iLow(_Symbol, _Period, 1);

  bool improved = false;

  if(dir > 0)
  {
    if(hi > g_bestFavorable) { g_bestFavorable = hi; improved = true; }
  }
  else
  {
    if(lo < g_bestFavorable || g_bestFavorable == g_entryPrice) { g_bestFavorable = lo; improved = true; }
  }

  if(improved) g_barsSinceNewMFE = 0;
  else         g_barsSinceNewMFE++;

  double mfeR = 0.0;
  if(g_Rdist > 0.0)
  {
    mfeR = (dir > 0) ? (g_bestFavorable - g_entryPrice) / g_Rdist
                     : (g_entryPrice - g_bestFavorable) / g_Rdist;
  }
  if(!g_armed && mfeR >= InpArmMinPeakPnL_R) g_armed = true;
}


bool ExitSignal_A_or_C(int dir, int cumH, double k, double stdCum)
{
  // For long: adverse negative shock => z <= -k
  // For short: adverse positive shock => z >= +k
  if(stdCum <= 0.0) return false;

  double c0 = iClose(_Symbol, _Period, 1);
  double cH = iClose(_Symbol, _Period, 1 + cumH);
  if(c0 <= 0.0 || cH <= 0.0) return false;

  double cumRet = (c0 / cH) - 1.0;
  double z = cumRet / stdCum;

  if(dir > 0) return (z <= -k);
  else        return (z >=  k);
}

bool ExitSignal_B_PullVel(int dir, double k, double stdVel)
{
  if(stdVel <= 0.0) return false;
  if(!g_armed) return false;

  double c = iClose(_Symbol, _Period, 1);
  if(c <= 0.0) return false;

  double dd = 0.0;
  if(dir > 0) dd = (g_bestFavorable - c);
  else        dd = (c - g_bestFavorable);

  double denom = (double)MathMax(1, g_barsSinceNewMFE);
  double vel = dd / denom;
  double z = vel / stdVel;

  return (z >= k);
}

// --- Entries ---

bool CandleIsGreen(int shift)
{
  double o = iOpen(_Symbol, _Period, shift);
  double c = iClose(_Symbol, _Period, shift);
  return (c > o);
}

bool CandleIsRed(int shift)
{
  double o = iOpen(_Symbol, _Period, shift);
  double c = iClose(_Symbol, _Period, shift);
  return (c < o);
}

bool Entry_A_or_B_Long()
{
  // trend_up = close > EMA(slow_n)
  double emaSlow;
  if(!GetMA(hEmaSlow200, 1, emaSlow)) return false;
  double c1 = iClose(_Symbol, _Period, 1);
  bool trend_up = (c1 > emaSlow);

  // red_seq = red.shift(2).rolling(n_red).sum()==n_red
  // with n_red=3 => bars 3,4,5 are red
  bool red_seq = true;
  for(int k=0;k<InpA_NRed;k++)
    red_seq = red_seq && CandleIsRed(3 + k);

  // long_cond = trend_up & red_seq & green & green.shift(1)
  bool green1 = CandleIsGreen(1);
  bool green2 = CandleIsGreen(2);

  return (trend_up && red_seq && green1 && green2);
}

double EWMAStdUpdate(double x, int span)
{
  double alpha = 2.0 / (span + 1.0);

  if(!g_ewmInit)
  {
    g_ewmInit = true;
    g_ewmMean = x;
    g_ewmVar  = 0.0;
    return 0.0;
  }

  double prevMean = g_ewmMean;
  g_ewmMean = alpha * x + (1.0 - alpha) * prevMean;

  // EWMA variance (simple, stable)
  double diff = x - g_ewmMean;
  g_ewmVar = alpha * diff*diff + (1.0 - alpha) * g_ewmVar;

  if(g_ewmVar < 0.0) g_ewmVar = 0.0;
  return MathSqrt(g_ewmVar);
}

void UpdateEWMAResidualOnClosedBar()
{
  // residual = close - EMA(mid)
  double emaMid;
  if(!GetMA(hEmaMid48, 1, emaMid)) return;

  double c1 = iClose(_Symbol, _Period, 1);
  double resid = c1 - emaMid;
  EWMAStdUpdate(resid, InpC_VolSpan);
}

bool Entry_C_LongShort(int &dirOut)
{
  double emaMid;
  if(!GetMA(hEmaMid48, 1, emaMid)) return false;

  double c1 = iClose(_Symbol, _Period, 1);
  double resid = c1 - emaMid;

  double std = MathSqrt(g_ewmVar);
  if(std <= 0.0) return false;

  double z = resid / std;

  bool long_cond  = (z <= -InpC_ZEntry);
  bool short_cond = (z >=  InpC_ZEntry);

  // edge_trigger handled outside (separate prevs)
  if(long_cond)  { dirOut = +1; return true; }
  if(short_cond) { dirOut = -1; return true; }
  return false;
}

// --- Orders ---

bool EnsurePositionStop(int dir)
{
  if(!PositionSelect(_Symbol)) return false;

  double sl = PositionGetDouble(POSITION_SL);
  if(sl > 0.0) return true;

  double atr;
  if(!GetAtrForStop(atr)) return false;

  double dist = atr * InpStopATRMult;
  double minDist = MathMax((double)InpMinStopPoints * _Point, BrokerMinStopDistancePrice());
  if(dist < minDist) dist = minDist;

  double priceOpen = PositionGetDouble(POSITION_PRICE_OPEN);
  double newSL = (dir > 0) ? (priceOpen - dist) : (priceOpen + dist);
  newSL = NormalizeDouble(newSL, _Digits);

  trade.SetExpertMagicNumber(InpMagicBase + (ulong)InpRunMode);
  return trade.PositionModify(_Symbol, newSL, 0.0);
}

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
  return v;
}

double CashRiskPerLot(double stop_dist_price)
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

  return out;
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

  return out;
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

bool BuildOrderPrices(int dir, double &entry_price, double &stop_price, double &stop_dist_price)
{
  MqlTick tick;
  if(!SymbolInfoTick(_Symbol, tick))
    return false;

  double atr;
  if(!GetAtrForStop(atr))
    return false;

  double min_dist = MathMax((double)InpMinStopPoints * _Point, BrokerMinStopDistancePrice());
  stop_dist_price = MathMax(atr * InpStopATRMult, min_dist);

  entry_price = (dir > 0) ? tick.ask : tick.bid;
  if(entry_price <= 0.0) return false;

  stop_price = (dir > 0)
             ? (entry_price - stop_dist_price)
             : (entry_price + stop_dist_price);

  entry_price = NormalizeDouble(entry_price, _Digits);
  stop_price  = NormalizeDouble(stop_price,  _Digits);

  return true;
}

double ComputeVolumeByRiskProfile(double stop_dist_price, bool &allowed, double &target_risk_dollars)
{
  allowed = true;
  target_risk_dollars = 0.0;

  // Fixed lot
  if(InpRiskProfile == RP_FIXED_LOT)
    return NormalizeLotsClip(InpFixedLot);

  // Stepped lot
  if(InpRiskProfile == RP_STEPPED_LOT)
    return ComputeSteppedLotProfile();

  // Fractional profiles below
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
//////


bool ClosePosition()
{
  if(!PositionSelect(_Symbol)) return false;
  trade.SetExpertMagicNumber(InpMagicBase + (ulong)InpRunMode);
  bool ok = trade.PositionClose(_Symbol);
  if(ok) ResetRunState();
  return ok;
}

int OnInit()
{
  trade.SetTypeFillingBySymbol(_Symbol);

  hEmaSlow200 = iMA(_Symbol, _Period, InpA_SlowN, 0, MODE_EMA, PRICE_CLOSE);
  hEmaMid50   = iMA(_Symbol, _Period, InpA_MidN, 0, MODE_EMA, PRICE_CLOSE);
  hEmaFast20  = iMA(_Symbol, _Period, InpA_FastN, 0, MODE_EMA, PRICE_CLOSE);

  hEmaMid48   = iMA(_Symbol, _Period, InpC_MidN, 0, MODE_EMA, PRICE_CLOSE);
  hEmaSlow144 = iMA(_Symbol, _Period, InpC_SlowN, 0, MODE_EMA, PRICE_CLOSE);

  hATR        = iATR(_Symbol, _Period, InpATR_N);

  if(hEmaSlow200 == INVALID_HANDLE || hEmaMid48 == INVALID_HANDLE || hATR == INVALID_HANDLE)
    return INIT_FAILED;

  g_lastBarTime = (datetime)iTime(_Symbol, _Period, 0);
  g_peakEquity   = AccountInfoDouble(ACCOUNT_EQUITY);
  g_anchorEquity = g_peakEquity;
  return INIT_SUCCEEDED;
}

void OnDeinit(const int reason)
{
  if(hEmaSlow200 != INVALID_HANDLE) IndicatorRelease(hEmaSlow200);
  if(hEmaMid50   != INVALID_HANDLE) IndicatorRelease(hEmaMid50);
  if(hEmaFast20  != INVALID_HANDLE) IndicatorRelease(hEmaFast20);
  if(hEmaMid48   != INVALID_HANDLE) IndicatorRelease(hEmaMid48);
  if(hEmaSlow144 != INVALID_HANDLE) IndicatorRelease(hEmaSlow144);
  if(hATR        != INVALID_HANDLE) IndicatorRelease(hATR);
}

void UpdateRegimeStatsOnClosedBar()
{
  // 1-bar raw return (close-close)
  double c1 = iClose(_Symbol, _Period, 1);
  double c2 = iClose(_Symbol, _Period, 2);
  if(c1 <= 0.0 || c2 <= 0.0) return;

  double r = (c1 / c2) - 1.0;

  RollPush(g_r96Sum,  g_r96SumSq,  g_r96Buf,  g_r96Idx,  g_r96Count,  InpVolRegimeWindow, r);
  RollPush(g_r384Sum, g_r384SumSq, g_r384Buf, g_r384Idx, g_r384Count, InpVolRegimeWindow*4, r);
}

void UpdateRetZStatsOnClosedBar()
{
  // A: cum horizon 2
  {
    int h = InpRetZ_CumHorizon_A;
    double c0 = iClose(_Symbol, _Period, 1);
    double cH = iClose(_Symbol, _Period, 1 + h);
    if(c0 > 0.0 && cH > 0.0)
    {
      double cumRet = (c0 / cH) - 1.0;
      RollPush(g_retSumA, g_retSumSqA, g_retBufA, g_retIdxA, g_retCountA, InpRetZ_Window, cumRet);
    }
  }
  // C: cum horizon 3
  {
    int h = InpRetZ_CumHorizon_C;
    double c0 = iClose(_Symbol, _Period, 1);
    double cH = iClose(_Symbol, _Period, 1 + h);
    if(c0 > 0.0 && cH > 0.0)
    {
      double cumRet = (c0 / cH) - 1.0;
      RollPush(g_retSumC, g_retSumSqC, g_retBufC, g_retIdxC, g_retCountC, InpRetZ_Window, cumRet);
    }
  }
}

void UpdatePullVelStatsOnClosedBar()
{
  int dir; double vol;
  if(!PositionIsOpen(dir, vol)) return;

  if(!g_armed) return;

  double c = iClose(_Symbol, _Period, 1);
  double dd = (dir > 0) ? (g_bestFavorable - c) : (c - g_bestFavorable);
  double denom = (double)MathMax(1, g_barsSinceNewMFE);
  double vel = dd / denom;

  RollPush(g_pvSum, g_pvSumSq, g_pvBuf, g_pvIdx, g_pvCount, (int)InpPullVelWindow, vel);
}

void EvaluateAndQueueEntry()
{
  int dir; double vol;
  if(PositionIsOpen(dir, vol)) return; // netting-style: one position per symbol

  if(InpRunMode == RUN_A || InpRunMode == RUN_B)
  {
    bool cond = Entry_A_or_B_Long();

    bool prev = (InpRunMode == RUN_A) ? g_prevSigA : g_prevSigB;
    bool edge = cond && !prev;

    if(InpRunMode == RUN_A) g_prevSigA = cond;
    else                    g_prevSigB = cond;

    if(edge)
    {
      g_pendingDir = +1;
      g_pending = true;
      return;
    }
  }

  if(InpRunMode == RUN_C)
  {
    int dirNow = 0;
    bool condAny = Entry_C_LongShort(dirNow);

    // edge per direction
    bool long_cond  = (dirNow == +1);
    bool short_cond = (dirNow == -1);

    bool edge = false;
    if(long_cond)
    {
      edge = long_cond && !g_prevSigC_L;
      g_prevSigC_L = long_cond;
      g_prevSigC_S = false;
    }
    else if(short_cond)
    {
      edge = short_cond && !g_prevSigC_S;
      g_prevSigC_S = short_cond;
      g_prevSigC_L = false;
    }
    else
    {
      g_prevSigC_L = false;
      g_prevSigC_S = false;
    }

    if(condAny && edge)
    {
      g_pendingDir = dirNow;
      g_pending = true;
      return;
    }
  }
}

void EvaluateExitAndExecute()
{
  int dir; double vol;
  if(!PositionIsOpen(dir, vol)) return;

  // ensure SL exists
  EnsurePositionStop(dir);

  // update run-state was already updated on bar close
  if(!g_armed) return;

  // overlay_scope = runner_only_after_arm
  // Apply gating + detector
  bool doExit = false;

  if(InpRunMode == RUN_A)
  {
    if(Gate_HVOL_STALL8())
    {
      double stdA = StdFromSums(g_retSumA, g_retSumSqA, g_retCountA);
      doExit = ExitSignal_A_or_C(dir, InpRetZ_CumHorizon_A, InpRetZ_K, stdA);
    }
  }
  else if(InpRunMode == RUN_B)
  {
    if(Gate_HVOL_STALL8())
    {
      double stdPV = StdFromSums(g_pvSum, g_pvSumSq, g_pvCount);
      doExit = ExitSignal_B_PullVel(dir, InpPullVelK, stdPV);
    }
  }
  else if(InpRunMode == RUN_C)
  {
    if(Gate_WHIP())
    {
      double stdC = StdFromSums(g_retSumC, g_retSumSqC, g_retCountC);
      doExit = ExitSignal_A_or_C(dir, InpRetZ_CumHorizon_C, InpRetZ_K, stdC);
    }
  }

  if(doExit)
    ClosePosition();
}

void ExecutePendingIfAny()
{
  if(!g_pending) return;

  // execute next bar open tick
  OpenPosition(g_pendingDir);

  g_pending = false;
  g_pendingDir = 0;
}

void SyncRunStateFromPosition()
{
  int dir;
  double vol;

  if(!PositionIsOpen(dir, vol))
  {
    ResetRunState();
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
    double atr = 0.0;
    if(GetAtrForStop(atr))
      g_Rdist = atr * InpStopATRMult;
    else
      g_Rdist = _Point * MathMax(1.0, InpMinStopPoints);

    double minDist = _Point * InpMinStopPoints;
    if(g_Rdist < minDist)
      g_Rdist = minDist;
  }

  g_bestFavorable = g_entryPrice;
  g_barsSinceNewMFE = 0;
  g_armed = false;

  ArrayResize(g_pvBuf, 0);
  g_pvIdx = 0;
  g_pvCount = 0;
  g_pvSum = 0.0;
  g_pvSumSq = 0.0;
}

bool InitRunStateFromLivePosition()
{
  int dir;
  double vol;

  if(!PositionIsOpen(dir, vol))
    return false;

  g_entryPrice = PositionGetDouble(POSITION_PRICE_OPEN);

  double sl = PositionGetDouble(POSITION_SL);
  if(sl <= 0.0)
    return false;

  g_Rdist = MathAbs(g_entryPrice - sl);

  double minDist = _Point * InpMinStopPoints;
  if(g_Rdist < minDist)
    g_Rdist = minDist;

  g_bestFavorable = g_entryPrice;
  g_barsSinceNewMFE = 0;
  g_armed = false;

  ArrayResize(g_pvBuf, 0);
  g_pvIdx = 0;
  g_pvCount = 0;
  g_pvSum = 0.0;
  g_pvSumSq = 0.0;

  return true;
}

bool OpenPosition(int dir)
{
  double entry_price, stop_price, stop_dist_price;
  if(!BuildOrderPrices(dir, entry_price, stop_price, stop_dist_price))
    return false;

  bool allowed = false;
  double target_risk_dollars = 0.0;
  double vol = ComputeVolumeByRiskProfile(stop_dist_price, allowed, target_risk_dollars);

  if(!allowed || vol <= 0.0)
  {
    PrintFormat("ENTRY BLOCKED | profile=%d | targetRisk=%.2f | stopDist=%.5f",
                (int)InpRiskProfile, target_risk_dollars, stop_dist_price);
    return false;
  }

  trade.SetExpertMagicNumber(InpMagicBase + (ulong)InpRunMode);

  bool ok = false;
  if(dir > 0)
    ok = trade.Buy(vol, _Symbol, 0.0, stop_price, 0.0);
  else
    ok = trade.Sell(vol, _Symbol, 0.0, stop_price, 0.0);

  if(!ok)
  {
    PrintFormat("ORDER FAILED | dir=%d | vol=%.2f | sl=%.5f | err=%d",
                dir, vol, stop_price, GetLastError());
    return false;
  }

  g_entryPrice = 0.0;
  SyncRunStateFromPosition();

  PrintFormat("ENTRY OK | profile=%d | dir=%d | vol=%.2f | targetRisk=%.2f | stopDist=%.5f | DD=%.2f%% | openRisk=%.2f",
              (int)InpRiskProfile,
              dir,
              vol,
              target_risk_dollars,
              stop_dist_price,
              100.0 * CurrentDrawdownFrac(),
              CurrentOpenRiskDollars());

  return true;
}

void EvaluateEntriesAndExecuteNow()
{
  int dir;
  double vol;

  if(PositionIsOpen(dir, vol))
    return;

  if(InpRunMode == RUN_A || InpRunMode == RUN_B)
  {
    bool cond = Entry_A_or_B_Long();
    bool prev = (InpRunMode == RUN_A) ? g_prevSigA : g_prevSigB;
    bool edge = cond && !prev;

    if(InpRunMode == RUN_A)
      g_prevSigA = cond;
    else
      g_prevSigB = cond;

    if(edge)
      OpenPosition(+1);

    return;
  }

  if(InpRunMode == RUN_C)
  {
    int dirNow = 0;
    bool condAny = Entry_C_LongShort(dirNow);

    bool long_cond  = (condAny && dirNow == +1);
    bool short_cond = (condAny && dirNow == -1);

    bool edge = false;

    if(long_cond)
    {
      edge = !g_prevSigC_L;
      g_prevSigC_L = true;
      g_prevSigC_S = false;
    }
    else if(short_cond)
    {
      edge = !g_prevSigC_S;
      g_prevSigC_S = true;
      g_prevSigC_L = false;
    }
    else
    {
      g_prevSigC_L = false;
      g_prevSigC_S = false;
    }

    if(edge)
      OpenPosition(dirNow);
  }
}

void OnTick()
{
  UpdatePeakEquity();

  if(!IsNewBar()) return;

  SyncRunStateFromPosition();

  AtrAndRegimeUpdateOnClosedBar(InpATR_N, InpVolRegimeWindow, InpWhipsawWindow);
  UpdateEWMAResidualOnClosedBar();
  UpdateRetZStatsOnClosedBar();
  UpdateMFEStateOnClosedBar();
  UpdatePullVelStatsOnClosedBar();

  EvaluateExitAndExecute();
  ExecutePendingIfAny();
  EvaluateAndQueueEntry();
}

