#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MT5 QUANT STARSHIP – EINSTEIN FULL STACK (V2.2 LIGHT MODE + HARD-CODED DEMO CREDENTIALS)
---------------------------------------------------------------------------
Ova verzija:
  - Više trejdova (olabavljeni filteri)
  - News filter samo HIGH impact (blackout pre/poslije objave)
  - Isključeni adaptive threshold & EV filter (više prolaza signala)
  - Uži SL i brži TP (brža realizacija)
  - Partial scaling + smooth trailing
  - ML ensemble (per-symbol SGD + global + LightGBM + stacking + calibracija)
  - Offline pretrain + inkrementalni online update
  - Backtest modul + News integracija
  - Hard-coded DEMO MT5 kredencijali (na tvoj zahtjev)
  - Soft profit target (opcija – default off)
  - Weekend trading shutdown (Saturday/Sunday) - use --allow-weekend-trading to override
---------------------------------------------------------------------------
UPOZORENJE:
  Opušteni risk filteri povećavaju varijansu i potencijalni drawdown.
  Preporuka: testiraj na DEMO i skaliraj risk_per_trade prema balansu.
  Trading je automatski onemogućen vikendom (subota/nedjelja) osim ako se koristi --allow-weekend-trading.

POKRETANJE (LIVE DEMO):
  python mt5_quant_starship_einstein_full.py --mode live --einstein

BACKTEST PRIMJER:
  python mt5_quant_starship_einstein_full.py --mode backtest --data-dir ./btdata --bt-enable-ml --bt-enable-news --news-file news.json --bt-write-trades

Ako želiš kasnije security: ukloni hard-coded kredencijale i prebaci na env varijable.
"""

import os, sys, time, math, signal, argparse, logging, gzip, pickle, traceback, json, random
from datetime import datetime, timedelta
from collections import defaultdict, deque
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import MetaTrader5 as mt5
except ImportError:
    raise SystemExit("Nedostaje MetaTrader5 modul. Instaliraj: pip install MetaTrader5")

from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression
from sklearn.cluster import KMeans
from scipy.stats import ks_2samp

try:
    import lightgbm as lgb
except ImportError:
    lgb = None
try:
    from catboost import CatBoostClassifier
except ImportError:
    CatBoostClassifier = None

# ================== HARD-CODED DEMO CREDENTIALS (po tvom zahtjevu) ==================
MT5_CREDENTIALS = {
    "login": 125553342,
    "password": "pN36RkgDg*!M",
    "server": "TradeQuo"
}

# ================== CONFIG (Light Mode) ==================
BASE_CONFIG = {
    "general": {
        "symbols": ["EURUSD","GBPUSD","USDJPY","XAUUSD","BTCUSD","AUDUSD"],
        "timeframes": ["M1","M5","M15","M30","H1","H4"],
        "lookback_bars": 1700,
        "poll_interval_seconds": 5,
        "min_bars_start": 300
    },
    "risk": {
        "account_risk_percent_per_trade": 1.0,
        "max_parallel_trades": 40,
        "max_symbol_trades": 6,
        "sl_atr_multiplier": 1.6,
        "tp_atr_multiplier": 2.3,
        "breakeven_atr_threshold": 1.0,
        "trail_atr_multiplier": 1.35,
        "daily_loss_limit_pct": 999.0,
        "max_drawdown_pct": 30.0,
        "max_consecutive_losses": 15,
        "equity_kill_switch": False,
        "volatility_atr_spike_factor": 3.5,
        "cooldown_seconds_strategy": 30,
        "cooldown_seconds_symbol": 15,
        "partial_scale": {
            "enabled": True,
            "levels": [1.8, 3.0],
            "scale_percents": [0.5, 0.5]
        },
        "disable_daily_loss_limit": True,
        "soft_profit_target_enabled": False,
        "soft_profit_target_usd": 100.0,
        "soft_profit_risk_scale": 0.4
    },
    "deal_filters": {
        "max_spread_points": 60,
        "allowed_hours": list(range(0,24)),
        "disable_friday_hour_after": 21,
        "pause_minutes_before_midnight": 3
    },
    "ml": {
        "enabled": True,
        "retrain_every_trades": 40,
        "min_samples_before_train": 250,
        "model_path": "ml_model.pkl.gz",
        "scaler_path": "ml_scaler.pkl.gz",
        "features": ["rsi","atr","volatility","ema_fast","ema_slow","bb_width","keltner_diff","session_hour","ha_trend"]
    },
    "einstein": {
        "enabled": True,
        "primary_timeframe": "M15",
        "train_bars": 2200,
        "forward_horizon": 10,
        "label_threshold": 0.0003,
        "per_symbol_models": True,
        "global_model_mix": [0.33, 0.33, 0.17, 0.17],  # sym, global, ensemble, strategy weight share
        "max_buffer_rows": 12000,
        "extra_features": True,
        "model_dir": "einstein_models",
        "min_symbol_samples": 400,
        "min_global_samples": 1200
    },
    "ml_ext": {
        "dynamic_label": True,
        "label_upper_quantile": 0.62,
        "label_lower_quantile": 0.38,
        "neutral_class": True,
        "future_horizons": [8,10],
        "use_lightgbm": True,
        "use_catboost": False,
        "stacking_enabled": True,
        "calibration": True,
        "adaptive_threshold": False,
        "expected_value_filter": False,
        "beta_scoring": True,
        "drift_detection": False,
        "regime_clustering": True,
        "regime_clusters": 4,
        "regime_models": False,
        "decay_alpha": 0.05,
        "lot_size_scaling": True,
        "lot_scale_range": [0.7, 1.8],
        "prob_degrade_highbin_loss_count": 4,
        "prob_highbin_floor": 0.72,
        "challenger_period_trades": 250,
        "challenger_promote_winrate_delta": 0.02,
        "challenger_promote_pf_delta": 0.15
    },
    "strategies": {
        "enabled": [
            "TrendFollowingATR","MeanReversionBands","BreakoutBox","MomentumRSI",
            "VWAPReversion","MultiTimeframeAlignment","VolatilityCompressionExpansion",
            "SessionRangeFade","PullbackEMA","SwingStructure","AdaptiveKalmanTrend",
            "CorrelationDivergence","OrderFlowProxy","PivotPointsReversal",
            "KeltnerBreakout","HeikenAshiTrend","RangeScalperMicro"
        ],
        "weights": {
            "TrendFollowingATR":1.3,
            "MeanReversionBands":1.1,
            "BreakoutBox":1.15,
            "MomentumRSI":1.05,
            "VWAPReversion":1.05,
            "MultiTimeframeAlignment":1.4,
            "VolatilityCompressionExpansion":1.2,
            "SessionRangeFade":0.95,
            "PullbackEMA":1.2,
            "SwingStructure":1.25,
            "AdaptiveKalmanTrend":1.1,
            "CorrelationDivergence":1.05,
            "OrderFlowProxy":1.35,
            "PivotPointsReversal":1.0,
            "KeltnerBreakout":1.1,
            "HeikenAshiTrend":1.15,
            "RangeScalperMicro":0.9,
            "MLConsensusBooster":0.0
        },
        "max_signals_per_cycle": 40
    },
    "news": {
        "enabled": True,
        "min_impact_required": "high",
        "blackout_minutes_before": 25,
        "blackout_minutes_after": 10,
        "apply_to_crypto": True,
        "apply_to_metals": True,
        "impacts_map": {"low":0,"medium":1,"high":2}
    },
    "logging": {
        "level": "INFO",
        "stdout": True,
        "log_dir": "logs",
        "journal_csv": "trade_journal.csv",
        "performance_csv": "performance_metrics.csv",
        "ml_report_json": "ml_monitor.json"
    },
    "backtest": {
        "slippage_points": 4,
        "commission_per_lot": 7.0
    },
    "trailing": {
        "enabled": True,
        "initial_lock_r_multiple": 0.6,
        "dynamic_start_r_multiple": 0.9,
        "tighten_r_multiple": 1.8,
        "ratchet_r_multiple": 2.7,
        "base_atr_mult": 1.35,
        "tight_atr_mult": 0.95,
        "ratchet_atr_mult": 0.65,
        "aging_minutes": 40,
        "aging_multiplier": 0.82,
        "trail_move_threshold_fraction": 0.22,
        "max_trail_updates_per_min": 8
    }
}

# ================== ENV Overrides ==================
def apply_env_overrides(cfg: dict):
    if "RISK_PER_TRADE" in os.environ:
        cfg["risk"]["account_risk_percent_per_trade"] = float(os.environ["RISK_PER_TRADE"])

# ================== Logging ==================
def setup_logger(cfg):
    os.makedirs(cfg["logging"]["log_dir"], exist_ok=True)
    logger = logging.getLogger("STARSHIP")
    if logger.handlers:
        return logger
    lvl = getattr(logging, cfg["logging"]["level"].upper(), logging.INFO)
    logger.setLevel(lvl)
    fmt = logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s")
    fh = logging.FileHandler(os.path.join(cfg["logging"]["log_dir"], f"run_{datetime.utcnow().strftime('%Y%m%d')}.log"), encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    if cfg["logging"]["stdout"]:
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        logger.addHandler(sh)
    return logger

# ================== Helpers ==================
PIP_FACTORS = {
    "XAUUSD":0.1,"BTCUSD":1.0,"EURUSD":0.0001,"GBPUSD":0.0001,
    "AUDUSD":0.0001,"NZDUSD":0.0001,"USDJPY":0.01,"USDCHF":0.0001,"USDCAD":0.0001
}
def pip_factor(symbol): return PIP_FACTORS.get(symbol,0.0001)

def is_weekend():
    """Check if current UTC time is weekend (Saturday or Sunday)."""
    weekday = datetime.utcnow().weekday()
    return weekday in [5, 6]  # Saturday=5, Sunday=6
def dynamic_position_size(balance, risk_percent, sl_pips, pip_value_est=10.0):
    risk_amount = balance * (risk_percent/100)
    if sl_pips <= 0: return 0.0
    lots = risk_amount / (sl_pips * pip_value_est)
    return max(round(lots,2),0.01)
def timeframe_to_mt5(tf):
    mapping={
        "M1":mt5.TIMEFRAME_M1,"M5":mt5.TIMEFRAME_M5,"M15":mt5.TIMEFRAME_M15,
        "M30":mt5.TIMEFRAME_M30,"H1":mt5.TIMEFRAME_H1,"H4":mt5.TIMEFRAME_H4,"D1":mt5.TIMEFRAME_D1
    }
    return mapping[tf]
def safe_nan(v, default=0):
    if v is None: return default
    if isinstance(v,float) and (math.isnan(v) or math.isinf(v)):
        return default
    return v

# ================== News Manager ==================
class NewsManager:
    def __init__(self, cfg, path=None):
        self.cfg = cfg
        self.path = path
        self.events = []
        self.impacts_map = cfg["news"]["impacts_map"]
        self.min_required_rank = self.impacts_map.get(cfg["news"]["min_impact_required"], 1)
        if path: self._load()
    def _parse_time(self,t):
        if isinstance(t,(int,float)):
            return datetime.utcfromtimestamp(t)
        for fmt in ["%Y-%m-%dT%H:%M:%S%z","%Y-%m-%dT%H:%M:%S","%Y-%m-%d %H:%M:%S"]:
            try:
                dt=datetime.strptime(t,fmt)
                if "%z" in fmt:
                    return dt.astimezone(tz=timedelta(0)).replace(tzinfo=None)
                return dt
            except: continue
        return None
    def _load(self):
        if not os.path.exists(self.path): return
        if self.path.lower().endswith(".json"):
            import json
            with open(self.path,"r",encoding="utf-8") as f:
                raw=json.load(f)
            for r in raw:
                tt=self._parse_time(r.get("time"))
                if not tt: continue
                imp=r.get("impact","medium").lower()
                if self.impacts_map.get(imp,0)<self.min_required_rank: continue
                cur=r.get("currencies",[])
                if isinstance(cur,str): cur=[cur]
                self.events.append({"time":tt,"impact":imp,"currencies":[c.upper() for c in cur],"title":r.get("title","")})
        else:
            import csv
            with open(self.path,"r",encoding="utf-8") as f:
                reader=csv.DictReader(f)
                for row in reader:
                    tt=self._parse_time(row.get("time"))
                    if not tt: continue
                    imp=row.get("impact","medium").lower()
                    if self.impacts_map.get(imp,0)<self.min_required_rank: continue
                    cur=[c.strip().upper() for c in row.get("currencies","").split(",") if c.strip()]
                    self.events.append({"time":tt,"impact":imp,"currencies":cur,"title":row.get("title","")})
        self.events.sort(key=lambda x:x["time"])
    def symbol_affected(self, symbol, currencies):
        base=symbol[:3]; quote=symbol[3:]
        for c in currencies:
            if c==base or c==quote: return True
        return False
    def in_blackout(self, symbol, now_utc=None):
        if now_utc is None:
            now_utc=datetime.utcnow()
        pre=timedelta(minutes=self.cfg["news"]["blackout_minutes_before"])
        post=timedelta(minutes=self.cfg["news"]["blackout_minutes_after"])
        for ev in self.events:
            if now_utc < ev["time"] - pre:
                if (ev["time"]-pre - now_utc).total_seconds()>3600:
                    break
                continue
            if ev["time"]-pre <= now_utc <= ev["time"]+post:
                if self.symbol_affected(symbol, ev["currencies"]):
                    return True, ev
        return False, None

# ================== MT5 Connector (hard-coded creds) ==================
class MT5Connector:
    def __init__(self, max_retries=5, retry_delay=3):
        self.max_retries=max_retries
        self.retry_delay=retry_delay
    def initialize(self):
        login=MT5_CREDENTIALS["login"]
        password=MT5_CREDENTIALS["password"]
        server=MT5_CREDENTIALS["server"]
        for i in range(self.max_retries):
            ok=mt5.initialize(login=login,password=password,server=server)
            if ok:
                logger.info(f"MT5 konekcija spremna (server={server}, login={login}).")
                return True
            logger.warning(f"MT5 init fail {i+1}/{self.max_retries}")
            time.sleep(self.retry_delay)
        return False
    def shutdown(self):
        mt5.shutdown()
        logger.info("MT5 shutdown.")
    def get_rates(self, symbol, timeframe, count=1000):
        try:
            rates=mt5.copy_rates_from_pos(symbol,timeframe_to_mt5(timeframe),0,count)
        except:
            return None
        if rates is None: return None
        df=pd.DataFrame(rates)
        if df.empty: return df
        df['time']=pd.to_datetime(df['time'],unit='s')
        return df
    def account_info(self):
        try: return mt5.account_info()
        except: return None
    def symbol_spread_points(self, symbol):
        info=mt5.symbol_info(symbol)
        if not info: return None
        tick=mt5.symbol_info_tick(symbol)
        if not tick: return None
        return (tick.ask - tick.bid)/info.point if info.point else None
    def send_order(self, symbol, direction, volume, sl_price=None, tp_price=None, comment=""):
        tick=mt5.symbol_info_tick(symbol)
        if not tick:
            logger.warning(f"Nema tick za {symbol}")
            return None
        price = tick.ask if direction=="BUY" else tick.bid
        order_type = mt5.ORDER_TYPE_BUY if direction=="BUY" else mt5.ORDER_TYPE_SELL
        req={
            "action":mt5.TRADE_ACTION_DEAL,
            "symbol":symbol,
            "volume":volume,
            "type":order_type,
            "price":price,
            "sl":sl_price or 0.0,
            "tp":tp_price or 0.0,
            "deviation":40,
            "magic":987654321,
            "comment":comment,
            "type_time":mt5.ORDER_TIME_GTC,
            "type_filling":mt5.ORDER_FILLING_RETURN
        }
        r=mt5.order_send(req)
        if not r:
            logger.warning("order_send None")
            return None
        if r.retcode!=mt5.TRADE_RETCODE_DONE:
            logger.warning(f"Order FAIL {symbol} {direction} ret={r.retcode}")
        else:
            logger.info(f"OPEN {direction} {symbol} vol={volume} sl={sl_price} tp={tp_price} ({comment})")
        return r
    def positions(self):
        try: return mt5.positions_get()
        except: return []
    def close_position(self, ticket):
        pos=mt5.positions_get(ticket=ticket)
        if not pos: return None
        p=pos[0]
        tick=mt5.symbol_info_tick(p.symbol)
        if not tick: return None
        price=tick.bid if p.type==0 else tick.ask
        order_type=mt5.ORDER_TYPE_SELL if p.type==0 else mt5.ORDER_TYPE_BUY
        req={
            "action":mt5.TRADE_ACTION_DEAL,
            "symbol":p.symbol,
            "volume":p.volume,
            "type":order_type,
            "position":p.ticket,
            "price":price,
            "deviation":40,
            "magic":p.magic,
            "comment":"close_starship"
        }
        r=mt5.order_send(req)
        if r and r.retcode==mt5.TRADE_RETCODE_DONE:
            logger.info(f"Closed {ticket}")
        return r

# ================== Data Feed ==================
class DataFeed:
    def __init__(self, mt5c, cfg):
        self.mt5=mt5c
        self.cfg=cfg
        self.data={}
    def bootstrap(self):
        for s in self.cfg["general"]["symbols"]:
            for tf in self.cfg["general"]["timeframes"]:
                df=self.mt5.get_rates(s,tf,self.cfg["general"]["lookback_bars"])
                if df is not None:
                    self.data[(s,tf)]=df
        logger.info("Bootstrap feed OK.")
    def update_all(self):
        for (s,tf) in list(self.data.keys()):
            df_new=self.mt5.get_rates(s,tf,self.cfg["general"]["lookback_bars"])
            if df_new is not None and not df_new.empty:
                self.data[(s,tf)]=df_new
        return self.data

# ================== Indicator Engine ==================
class IndicatorEngine:
    def __init__(self,cfg):
        self.cfg=cfg
        self.cache={}
    def update(self,bars):
        for (sym,tf),df in bars.items():
            if df is None or df.empty: continue
            ind={}
            c=df['close']; h=df['high']; l=df['low']; o=df['open']
            tr=pd.concat([(h-l),(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1)
            atr=tr.rolling(14).mean()
            ind['atr']=atr
            ind['ema_fast']=c.ewm(span=12,min_periods=12).mean()
            ind['ema_slow']=c.ewm(span=26,min_periods=26).mean()
            delta=c.diff()
            gain=(delta.where(delta>0,0)).rolling(14).mean()
            loss=(-delta.where(delta<0,0)).rolling(14).mean()
            rs=gain/loss.replace(0,np.nan)
            rsi=100-(100/(1+rs))
            ind['rsi']=rsi
            ma20=c.rolling(20).mean()
            std20=c.rolling(20).std()
            ind['bb_mid']=ma20
            ind['bb_upper']=ma20+2*std20
            ind['bb_lower']=ma20-2*std20
            ind['bb_width']=(ind['bb_upper']-ind['bb_lower'])/ma20
            ema20=c.ewm(span=20,min_periods=20).mean()
            ind['keltner_mid']=ema20
            ind['keltner_upper']=ema20+2*atr
            ind['keltner_lower']=ema20-2*atr
            ind['keltner_diff']=(ind['keltner_upper']-ind['keltner_lower'])/ema20
            ha_close=(o+h+l+c)/4
            ha_open=ha_close.copy()
            for i in range(1,len(df)):
                ha_open.iat[i]=(ha_open.iat[i-1]+ha_close.iat[i-1])/2
            ha_trend=np.where(ha_close>ha_open,1,-1)
            ind['ha_trend']=pd.Series(ha_trend,index=df.index)
            ind['volatility']=c.pct_change().rolling(30).std()
            vol=df['tick_volume']
            cum_vol=vol.cumsum()
            cum_pv=(c*vol).cumsum()
            ind['vwap']=cum_pv/cum_vol.replace(0,np.nan)
            if tf in ["M1","M5","M15","M30","H1"]:
                day=df['time'].dt.date
                daily=df.groupby(day).agg({'high':'max','low':'min','close':'last'})
                piv=[]
                for d in df['time'].dt.date:
                    if d in daily.index:
                        H=daily.loc[d,'high']; L=daily.loc[d,'low']; C=daily.loc[d,'close']
                        piv.append((H+L+C)/3)
                    else: piv.append(np.nan)
                ind['pivot']=pd.Series(piv,index=df.index)
            else:
                ind['pivot']=pd.Series(np.nan,index=df.index)
            self.cache[(sym,tf)]=ind
    def get_last(self, symbol, tf, name):
        d=self.cache.get((symbol,tf),{})
        s=d.get(name)
        if s is None or len(s)==0: return None
        try: return s.iloc[-1]
        except: return None
    def series(self, symbol, tf, name):
        d=self.cache.get((symbol,tf),{})
        return d.get(name)
    def features_snapshot(self, symbol, tf):
        d=self.cache.get((symbol,tf),{})
        snap={}
        for k,v in d.items():
            try: snap[k]=v.iloc[-1]
            except: snap[k]=None
        return snap

# ================== Strategy Bayesian Score ==================
class StrategyBayesScore:
    def __init__(self, alpha_init=2, beta_init=2):
        self.params=defaultdict(lambda:[alpha_init,beta_init])
    def update(self,strategy,profit):
        a,b=self.params[strategy]
        if profit>0: a+=1
        else: b+=1
        self.params[strategy]=[a,b]
    def score(self,strategy):
        a,b=self.params[strategy]; return a/(a+b)

# ================== High Prob Control ==================
class HighProbControl:
    def __init__(self, prob_floor=0.72, max_losses=4):
        self.prob_floor=prob_floor; self.max_losses=max_losses; self.loss_streak=0
    def update(self,prob,profit):
        if prob>=self.prob_floor:
            if profit<0: self.loss_streak+=1
            else: self.loss_streak=0
    def adjust_confidence(self,conf):
        if self.loss_streak>=self.max_losses:
            return conf*0.85
        return conf

# ================== EV / Threshold Helpers ==================
def compute_expectancy(prob, avg_win, avg_loss, cost=0.0):
    return prob * avg_win - (1 - prob) * avg_loss - cost
def adaptive_probability_threshold(history_probs, history_labels, grid=None):
    if grid is None: grid=np.linspace(0.45,0.78,14)
    best_thr=0.55; best_ev=-1e9
    probs=np.array(history_probs); labels=np.array(history_labels)
    for thr in grid:
        mask=probs>=thr
        if mask.sum()<30: continue
        wins=(labels[mask]==1).sum()
        losses=(labels[mask]==0).sum()
        if wins+losses==0: continue
        win_rate=wins/(wins+losses)
        ev=win_rate-(1-win_rate)
        if ev>best_ev:
            best_ev=ev; best_thr=thr
    return best_thr
def scale_lot(base_lot, prob, lo=0.7, hi=1.8):
    p_clamped=min(0.82,max(0.45,prob))
    ratio=(p_clamped-0.45)/0.37
    return round(base_lot*(lo+(hi-lo)*ratio),2)

# ================== Einstein Dataset Build ==================
def dynamic_label_assign(future_returns, upper_q=0.62, lower_q=0.38, neutral=True):
    uq=future_returns.quantile(upper_q)
    lq=future_returns.quantile(lower_q)
    labels=[]
    for fr in future_returns:
        if fr>uq: labels.append(1)
        elif fr<lq: labels.append(0)
        else: labels.append(-1 if neutral else None)
    return labels,uq,lq

def add_advanced_offline_features(df):
    c=df['close']; o=df['open']; h=df['high']; l=df['low']
    upper_wick=(h - df[['open','close']].max(axis=1))
    lower_wick=(df[['open','close']].min(axis=1) - l)
    rng=(h-l).replace(0,np.nan)
    body=(c-o).abs()
    body_to_range=body/rng
    df['upper_wick']=upper_wick
    df['lower_wick']=lower_wick
    df['body']=body
    df['body_to_range']=body_to_range
    df['log_return_1']=np.log(c/c.shift(1))
    df['log_return_5']=np.log(c/c.shift(5))
    df['log_return_15']=np.log(c/c.shift(15))
    for n in [10,20]:
        if len(df)>n:
            slopes=[np.nan]*len(df)
            for i in range(n,len(df)):
                sub=c.iloc[i-n:i]
                if sub.isna().any(): continue
                x=np.arange(n)
                A=np.vstack([x,np.ones(len(x))]).T
                m,_=np.linalg.lstsq(A,sub.values,rcond=None)[0]
                slopes[i]=m
            df[f"mom_slope_{n}"]=slopes
        else:
            df[f"mom_slope_{n}"]=np.nan
    return df

def build_einstein_dataset(mt5c, cfg, indicator_engine, symbols):
    ein=cfg["einstein"]; ext=cfg["ml_ext"]
    tf=ein["primary_timeframe"]
    horizons=ext["future_horizons"]
    bars_needed=ein["train_bars"]+max(horizons)+60
    rows=[]
    logger.info(f"[EINSTEIN] Offline dataset build tf={tf} bars={bars_needed}")
    for sym in symbols:
        df=mt5c.get_rates(sym,tf,bars_needed)
        if df is None or df.empty or len(df)<max(horizons)+50:
            logger.warning(f"[EINSTEIN] Not enough data {sym}")
            continue
        df=add_advanced_offline_features(df)
        tmp={(sym,tf):df.copy()}
        indicator_engine.update(tmp)
        cache=indicator_engine.cache.get((sym,tf))
        if not cache: continue
        feat_df=pd.DataFrame({
            "time":df['time'],
            "open":df['open'],
            "high":df['high'],
            "low":df['low'],
            "close":df['close'],
            "tick_volume":df['tick_volume'],
            "rsi":cache['rsi'],
            "atr":cache['atr'],
            "volatility":cache['volatility'],
            "ema_fast":cache['ema_fast'],
            "ema_slow":cache['ema_slow'],
            "bb_width":cache['bb_width'],
            "keltner_diff":cache['keltner_diff'],
            "ha_trend":cache['ha_trend'],
            "vwap":cache['vwap']
        })
        for col in ['upper_wick','lower_wick','body','body_to_range','log_return_1','log_return_5','log_return_15','mom_slope_10','mom_slope_20']:
            feat_df[col]=df[col]
        future_ret_blend=pd.Series([np.nan]*len(feat_df))
        for H in horizons:
            fr=np.log(feat_df['close'].shift(-H)/feat_df['close'])
            future_ret_blend=future_ret_blend.combine(fr,
                lambda a,b:(safe_nan(a,0)+safe_nan(b,0))/2 if (not np.isnan(a) and not np.isnan(b)) else (b if np.isnan(a) else a))
        if ext["dynamic_label"]:
            valid=future_ret_blend.dropna()
            if len(valid)<150:
                thr=ein["label_threshold"]
                labels=[]
                for i in range(len(feat_df)):
                    if i>=len(feat_df)-max(horizons): labels.append(None); continue
                    fr=future_ret_blend.iloc[i]
                    if pd.isna(fr): labels.append(None); continue
                    if fr>thr: labels.append(1)
                    elif fr<-thr: labels.append(0)
                    else: labels.append(-1 if ext["neutral_class"] else None)
            else:
                lbls,uq,lq=dynamic_label_assign(valid, upper_q=ext["label_upper_quantile"],
                                                lower_q=ext["label_lower_quantile"],
                                                neutral=ext["neutral_class"])
                label_map={}
                idxs=valid.index.tolist()
                for k,irow in enumerate(idxs): label_map[irow]=lbls[k]
                labels=[label_map.get(i,None) for i in range(len(feat_df))]
        else:
            thr=ein["label_threshold"]
            labels=[]
            for i in range(len(feat_df)):
                if i>=len(feat_df)-max(horizons): labels.append(None); continue
                fr=future_ret_blend.iloc[i]
                if pd.isna(fr): labels.append(None); continue
                if fr>thr: labels.append(1)
                elif fr<-thr: labels.append(0)
                else: labels.append(-1 if ext["neutral_class"] else None)
        feat_df['label']=labels
        feat_df['symbol']=sym
        feat_df['session_hour']=feat_df['time'].dt.hour
        feat_df['atr_norm']=feat_df['atr']/feat_df['close']
        feat_df['distance_from_vwap']=(feat_df['close']-feat_df['vwap'])/feat_df['close']
        bb_std=(cache['bb_upper'] - cache['bb_lower'])/4
        feat_df['bb_zscore']=(feat_df['close'] - cache['bb_mid'])/bb_std.replace(0,np.nan)
        feat_df['ema_ratio']=feat_df['ema_fast']/feat_df['ema_slow']
        feat_df['volume_rel']=feat_df['tick_volume']/feat_df['tick_volume'].rolling(100).median()
        rows.append(feat_df)
    if not rows:
        logger.warning("[EINSTEIN] Empty dataset")
        return pd.DataFrame()
    df_all=pd.concat(rows,ignore_index=True)
    df_all=df_all.dropna(subset=['label'])
    logger.info(f"[EINSTEIN] Dataset rows={len(df_all)}")
    return df_all

# ================== Einstein Ensemble ==================
class EinsteinEnsemble:
    def __init__(self,cfg):
        self.cfg=cfg
        self.base_feats=cfg["ml"]["features"]
        self.extra_enabled=cfg["einstein"]["extra_features"]
        self.ext=cfg["ml_ext"]
        self.ein=cfg["einstein"]
        self.model_dir=cfg["einstein"]["model_dir"]
        os.makedirs(self.model_dir,exist_ok=True)
        self.symbol_models={}
        self.symbol_scalers={}
        self.symbol_trained=set()
        self.global_model=SGDClassifier(loss="log_loss")
        self.global_scaler=StandardScaler()
        self.global_trained=False
        self.lgbm_model=None
        self.cat_model=None
        self.stack_meta=None
        self.calibrator=None
        self.regime_kmeans=None
        self.regime_models={}
        self.buffer_sym=defaultdict(list)
        self.buffer_sym_y=defaultdict(list)
        self.buffer_global=[]
        self.buffer_global_y=[]
        self.trade_counter=0
        self.prob_history=[]
        self.label_history=[]
        self.dynamic_threshold=0.55
        self.mix_weights=self.ein["global_model_mix"]
        self._load_existing()
    def _feature_order(self):
        feats=self.base_feats.copy()
        if self.extra_enabled:
            feats+=["log_return_1","log_return_5","log_return_15","ema_ratio",
                    "distance_from_vwap","bb_zscore","atr_norm","regime_vol",
                    "upper_wick","lower_wick","body","body_to_range",
                    "mom_slope_10","mom_slope_20","volume_rel"]
        return list(dict.fromkeys(feats))
    def _load_pickle(self,path):
        with gzip.open(path,'rb') as f: return pickle.load(f)
    def _save_pickle(self,obj,path):
        with gzip.open(path,'wb') as f: pickle.dump(obj,f)
    def _load_existing(self):
        gp=os.path.join(self.model_dir,"global_model.pkl.gz")
        gs=os.path.join(self.model_dir,"global_scaler.pkl.gz")
        if os.path.exists(gp) and os.path.exists(gs):
            try:
                self.global_model=self._load_pickle(gp)
                self.global_scaler=self._load_pickle(gs)
                self.global_trained=True
                logger.info("[EINSTEIN] Global model loaded.")
            except Exception as e:
                logger.warning(f"[EINSTEIN] Load global fail: {e}")
    def _persist_global(self):
        try:
            self._save_pickle(self.global_model, os.path.join(self.model_dir,"global_model.pkl.gz"))
            self._save_pickle(self.global_scaler, os.path.join(self.model_dir,"global_scaler.pkl.gz"))
        except Exception as e:
            logger.warning(f"[EINSTEIN] Persist global fail: {e}")
    def _ensure_symbol_model(self,symbol):
        if symbol in self.symbol_models: return
        self.symbol_models[symbol]=SGDClassifier(loss="log_loss")
        self.symbol_scalers[symbol]=StandardScaler()
    def _persist_symbol(self,symbol):
        try:
            self._save_pickle(self.symbol_models[symbol], os.path.join(self.model_dir,f"{symbol}_model.pkl.gz"))
            self._save_pickle(self.symbol_scalers[symbol], os.path.join(self.model_dir,f"{symbol}_scaler.pkl.gz"))
        except Exception as e:
            logger.warning(f"[EINSTEIN] Persist {symbol} fail: {e}")
    def offline_pretrain(self, df):
        if df.empty:
            logger.warning("[EINSTEIN] Pretrain skipped (empty).")
            return
        feats=self._feature_order()
        if self.ext["regime_clustering"]:
            try:
                source=df[['atr','volatility','bb_width']].fillna(0)
                self.regime_kmeans=KMeans(n_clusters=self.ext["regime_clusters"],n_init=10,random_state=42)
                df['regime_id']=self.regime_kmeans.fit_predict(source)
            except Exception as e:
                logger.warning(f"[EINSTEIN] Regime cluster fail: {e}")
                df['regime_id']=1
        else:
            df['regime_id']=1
        df_bin=df[df['label']!=-1].copy()
        if df_bin.empty:
            logger.warning("[EINSTEIN] No binary rows after neutral filter.")
            return
        if self.ein["per_symbol_models"]:
            for sym, dfg in df_bin.groupby("symbol"):
                X=dfg[feats].values; y=dfg['label'].values
                if len(X)<self.ein["min_symbol_samples"]: continue
                self._ensure_symbol_model(sym)
                Xs=self.symbol_scalers[sym].fit_transform(X)
                try:
                    self.symbol_models[sym].partial_fit(Xs,y,classes=[0,1])
                    self.symbol_trained.add(sym)
                    self._persist_symbol(sym)
                    logger.info(f"[EINSTEIN] Symbol pretrain {sym} rows={len(X)}")
                except Exception as e:
                    logger.warning(f"[EINSTEIN] Symbol train fail {sym}: {e}")
        if len(df_bin)>=self.ein["min_global_samples"]:
            Xg=df_bin[feats].values; yg=df_bin['label'].values
            Xgs=self.global_scaler.fit_transform(Xg)
            self.global_model.partial_fit(Xgs,yg,classes=[0,1])
            self.global_trained=True
            self._persist_global()
            logger.info(f"[EINSTEIN] Global pretrain rows={len(df_bin)}")
        if self.ext["use_lightgbm"] and lgb:
            try:
                lgbm=lgb.LGBMClassifier(n_estimators=300,learning_rate=0.04,
                                        num_leaves=32,subsample=0.8,colsample_bytree=0.8,random_state=42)
                lgbm.fit(df_bin[feats],df_bin['label'])
                self.lgbm_model=lgbm
                logger.info("[EINSTEIN] LGBM trained.")
            except Exception as e:
                logger.warning(f"[EINSTEIN] LGBM fail: {e}")
        if self.ext["use_catboost"] and CatBoostClassifier:
            try:
                cat=CatBoostClassifier(depth=6,learning_rate=0.05,iterations=300,
                                       verbose=False,random_state=42)
                cat.fit(df_bin[feats],df_bin['label'])
                self.cat_model=cat
                logger.info("[EINSTEIN] CatBoost trained.")
            except Exception as e:
                logger.warning(f"[EINSTEIN] CatBoost fail: {e}")
        if self.ext["stacking_enabled"] and (self.lgbm_model or self.cat_model):
            try:
                base_preds=[]
                if self.lgbm_model: base_preds.append(self.lgbm_model.predict_proba(df_bin[feats])[:,1])
                if self.cat_model: base_preds.append(self.cat_model.predict_proba(df_bin[feats])[:,1])
                if base_preds:
                    blend=np.vstack(base_preds).T
                    meta=LogisticRegression()
                    meta.fit(blend,df_bin['label'])
                    self.stack_meta=meta
                    if self.ext["calibration"]:
                        raw=meta.predict_proba(blend)[:,1]
                        iso=IsotonicRegression(out_of_bounds='clip')
                        iso.fit(raw,df_bin['label'])
                        self.calibrator=iso
                        logger.info("[EINSTEIN] Isotonic calibration done.")
            except Exception as e:
                logger.warning(f"[EINSTEIN] Stacking fail: {e}")
    def add_online_example(self, symbol, feature_dict, label):
        feats=self._feature_order()
        x=[]
        for f in feats:
            v=feature_dict.get(f,0)
            if v is None or (isinstance(v,float) and (math.isnan(v) or math.isinf(v))): v=0
            x.append(v)
        self.buffer_sym[symbol].append(x)
        self.buffer_sym_y[symbol].append(label)
        self.buffer_global.append(x)
        self.buffer_global_y.append(label)
        self.trade_counter+=1
        max_rows=self.ein["max_buffer_rows"]
        if len(self.buffer_global)>max_rows:
            self.buffer_global=self.buffer_global[-max_rows:]
            self.buffer_global_y=self.buffer_global_y[-max_rows:]
        if len(self.buffer_sym[symbol])>max_rows:
            self.buffer_sym[symbol]=self.buffer_sym[symbol][-max_rows:]
            self.buffer_sym_y[symbol]=self.buffer_sym_y[symbol][-max_rows:]
        self.prob_history.append(feature_dict.get("_prob_snapshot",0.5))
        self.label_history.append(label)
        if len(self.prob_history)>5000:
            self.prob_history=self.prob_history[-5000:]
            self.label_history=self.label_history[-5000:]
    def maybe_retrain(self):
        if self.trade_counter % max(5,self.cfg["ml"]["retrain_every_trades"]) !=0:
            return
        feats=self._feature_order()
        if self.ein["per_symbol_models"]:
            for sym,Xbuf in self.buffer_sym.items():
                if len(Xbuf)<self.ein["min_symbol_samples"]: continue
                ybuf=self.buffer_sym_y[sym]
                self._ensure_symbol_model(sym)
                try:
                    Xs=self.symbol_scalers[sym].fit_transform(np.array(Xbuf))
                    self.symbol_models[sym].partial_fit(Xs,np.array(ybuf),classes=[0,1])
                    self.symbol_trained.add(sym)
                    self._persist_symbol(sym)
                except Exception as e:
                    logger.warning(f"[EINSTEIN] Retrain sym fail {sym}: {e}")
        if len(self.buffer_global)>=self.ein["min_global_samples"]:
            try:
                Xg=self.global_scaler.fit_transform(np.array(self.buffer_global))
                yg=np.array(self.buffer_global_y)
                self.global_model.partial_fit(Xg,yg,classes=[0,1] if not self.global_trained else None)
                self.global_trained=True
                self._persist_global()
            except Exception as e:
                logger.warning(f"[EINSTEIN] Retrain global fail: {e}")
    def _predict_base(self, symbol, feats_row:dict):
        feats=self._feature_order()
        x=[]
        for f in feats:
            v=feats_row.get(f,0)
            if v is None or (isinstance(v,float) and (math.isnan(v) or math.isinf(v))): v=0
            x.append(v)
        X=np.array(x).reshape(1,-1)
        p_sym=None
        if self.ein["per_symbol_models"] and symbol in self.symbol_models and symbol in self.symbol_trained:
            try:
                Xs=self.symbol_scalers[symbol].transform(X)
                p_sym=self.symbol_models[symbol].predict_proba(Xs)[0][1]
            except: p_sym=None
        p_glob=None
        if self.global_trained:
            try:
                Xg=self.global_scaler.transform(X)
                p_glob=self.global_model.predict_proba(Xg)[0][1]
            except: p_glob=None
        p_ens=None
        base_preds=[]
        if self.lgbm_model:
            try: base_preds.append(self.lgbm_model.predict_proba(X)[:,1][0])
            except: pass
        if self.cat_model:
            try: base_preds.append(self.cat_model.predict_proba(X)[0][1])
            except: pass
        if base_preds:
            bmean=np.mean(base_preds)
            if self.stack_meta:
                try:
                    meta_in=np.array(base_preds).reshape(1,-1)
                    mp=self.stack_meta.predict_proba(meta_in)[0][1]
                    if self.calibrator: mp=self.calibrator.transform([mp])[0]
                    p_ens=mp
                except:
                    p_ens=bmean
            else: p_ens=bmean
        parts=[]; weights=[]
        w_sym,w_glob,w_extra,w_strategy=self.mix_weights
        if p_sym is not None: parts.append(p_sym); weights.append(w_sym)
        if p_glob is not None: parts.append(p_glob); weights.append(w_glob)
        if p_ens is not None: parts.append(p_ens); weights.append(w_extra)
        if not parts: return 0.5
        prob=sum(p*w for p,w in zip(parts,weights))/sum(weights)
        return max(0.0005,min(0.9995,prob))
    def annotate_signals(self, signals, indicator_engine):
        for s in signals:
            symbol=s['symbol']; tf=s.get("timeframe","M15")
            feat=indicator_engine.features_snapshot(symbol,tf)
            feat["session_hour"]=datetime.utcnow().hour
            feat["ha_trend"]=feat.get("ha_trend",0)
            feat["ema_ratio"]=(feat.get("ema_fast",0)/feat.get("ema_slow",1)) if feat.get("ema_slow") else 1
            feat["distance_from_vwap"]=0
            if feat.get("vwap") not in (0,None):
                ref=feat.get("close", feat.get("ema_fast",0))
                feat["distance_from_vwap"]=(ref - feat["vwap"])/max(1e-9,ref)
            feat["atr_norm"]=(feat.get("atr",0)/feat.get("close",1)) if feat.get("close") else 0
            feat["bb_zscore"]=0
            if all(feat.get(x) is not None for x in ["bb_upper","bb_lower","bb_mid"]):
                std_approx=(feat["bb_upper"]-feat["bb_lower"])/4
                if std_approx:
                    ref=feat.get("close", feat.get("ema_fast",0))
                    feat["bb_zscore"]=(ref - feat["bb_mid"])/std_approx
            for lr in ["log_return_1","log_return_5","log_return_15","mom_slope_10","mom_slope_20","upper_wick","lower_wick","body","body_to_range","volume_rel"]:
                feat.setdefault(lr,0)
            vol_val=feat.get("volatility",0)
            if vol_val is None or np.isnan(vol_val): feat["regime_vol"]=1
            elif vol_val<0.0005: feat["regime_vol"]=0
            elif vol_val>0.002: feat["regime_vol"]=2
            else: feat["regime_vol"]=1
            prob=self._predict_base(symbol, feat)
            feat["_prob_snapshot"]=prob
            s["ml_prob"]=prob
            base=s.get("confidence",0.5)
            strategy_component=self.mix_weights[3]
            model_component=1-strategy_component
            s["confidence"]= base*strategy_component + prob*model_component
            if prob<0.35: s["confidence"]*=0.75
            elif prob>0.7: s["confidence"]*=1.05

# ================== Risk Manager ==================
class RiskManager:
    def __init__(self,cfg, mt5c):
        self.cfg=cfg; self.mt5=mt5c
        self.daily_start_equity=None
        self.max_equity_seen=None
        self.consecutive_losses=0
        self.closed_pnl_today=0.0
        self.kill_switch=False
        self.dynamic_risk_scale=1.0
    def _equity(self):
        ai=self.mt5.account_info()
        if not ai: return None
        return ai.equity
    def daily_reset_if_needed(self):
        now=datetime.utcnow()
        if now.hour==0 and now.minute<5:
            eq=self._equity()
            if eq and (self.daily_start_equity is None or now.minute<2):
                self.daily_start_equity=eq
                self.closed_pnl_today=0.0
                self.consecutive_losses=0
                self.kill_switch=False
                self.dynamic_risk_scale=1.0
    def update_from_closed(self, profit):
        self.closed_pnl_today += profit
        if profit<0: self.consecutive_losses+=1
        else: self.consecutive_losses=0
        if self.cfg["risk"].get("soft_profit_target_enabled"):
            target=self.cfg["risk"]["soft_profit_target_usd"]
            if self.closed_pnl_today>=target and self.dynamic_risk_scale==1.0:
                self.dynamic_risk_scale=self.cfg["risk"]["soft_profit_risk_scale"]
                logging.info(f"[RISK] Soft profit target dosegnut (+{self.closed_pnl_today:.2f}). Risk scale -> {self.dynamic_risk_scale}")
    def evaluate_equity_guards(self):
        if self.kill_switch: return False
        eq=self._equity()
        if eq is None: return True
        if self.daily_start_equity is None: self.daily_start_equity=eq
        if self.max_equity_seen is None or eq>self.max_equity_seen:
            self.max_equity_seen=eq
        if not self.cfg["risk"].get("disable_daily_loss_limit",False):
            daily_drop=(self.daily_start_equity - eq)/self.daily_start_equity*100 if self.daily_start_equity>0 else 0
            if daily_drop>self.cfg["risk"]["daily_loss_limit_pct"]:
                logging.error(f"[RISK] Daily loss limit hit {daily_drop:.2f}%")
                self.kill_switch=True
                return False
        overall_dd=(self.max_equity_seen - eq)/self.max_equity_seen*100 if self.max_equity_seen else 0
        if overall_dd>self.cfg["risk"]["max_drawdown_pct"]:
            logging.error(f"[RISK] Max DD limit hit {overall_dd:.2f}%")
            self.kill_switch=True
            return False
        if self.consecutive_losses>=self.cfg["risk"]["max_consecutive_losses"]:
            logging.error(f"[RISK] Consecutive losses limit {self.consecutive_losses}")
            self.kill_switch=True
            return False
        return True
    def effective_risk_percent(self):
        base=self.cfg["risk"]["account_risk_percent_per_trade"]
        return base * self.dynamic_risk_scale
    def can_open(self, symbol):
        if not self.evaluate_equity_guards(): return False
        pos=mt5.positions_get()
        if pos and len(pos)>=self.cfg["risk"]["max_parallel_trades"]:
            return False
        if pos:
            same=sum(1 for p in pos if p.symbol==symbol)
            if same>=self.cfg["risk"]["max_symbol_trades"]:
                return False
        return True

# ================== Portfolio Tracker ==================
class PortfolioTracker:
    def __init__(self,cfg, mt5c, ml_module:Optional[EinsteinEnsemble], bayes_score:StrategyBayesScore, high_prob_ctrl:HighProbControl):
        self.cfg=cfg; self.mt5=mt5c; self.ml=ml_module
        self.open_meta={}
        self.journal_path=cfg["logging"]["journal_csv"]
        self.performance_path=cfg["logging"]["performance_csv"]
        self.ml_report_path=cfg["logging"]["ml_report_json"]
        if not os.path.exists(self.journal_path):
            with open(self.journal_path,'w',encoding='utf-8') as f:
                f.write("close_time,ticket,symbol,strategy,dir,volume,open_price,close_price,profit,ml_prob,label,prob_snapshot\n")
        self.strategy_perf=defaultdict(lambda: {"wins":0,"losses":0,"profit":0.0,"pips":0.0,"avg_win":1.0,"avg_loss":1.0})
        self.last_history_sync=0
        self.last_perf_flush=0
        self.bayes=bayes_score
        self.high_prob=high_prob_ctrl
        self.decay_alpha=self.cfg["ml_ext"]["decay_alpha"]
        self.history_probs=[]
        self.history_labels=[]
    def register_open(self, ticket, meta):
        self.open_meta[ticket]=meta
        logger.info(f"[OPEN] t={ticket} {meta['symbol']} {meta['direction']} strat={meta['strategy']} conf={meta['confidence']:.2f}")
    def sync_closed(self, risk_manager:RiskManager):
        now=time.time()
        if now - self.last_history_sync < 15: return
        self.last_history_sync=now
        from_time=datetime.utcnow()-timedelta(days=2)
        deals=mt5.history_deals_get(from_time, datetime.utcnow())
        if not deals: return
        for d in deals:
            if d.entry not in (1,3): continue
            ticket=d.position_id
            if ticket in self.open_meta:
                meta=self.open_meta.pop(ticket)
                profit=d.profit
                prob_snapshot=meta.get('ml_prob',0.5)
                label=1 if profit>0 else 0
                if self.ml:
                    fsnap=meta.get("features_snapshot",{}).copy()
                    fsnap["_prob_snapshot"]=prob_snapshot
                    self.ml.add_online_example(meta['symbol'], fsnap, label)
                self._journal(d, meta, profit, label, prob_snapshot)
                risk_manager.update_from_closed(profit)
                self._perf(meta, profit)
                self.bayes.update(meta['strategy'], profit)
                self.high_prob.update(prob_snapshot, profit)
                self.history_probs.append(prob_snapshot)
                self.history_labels.append(label)
                if len(self.history_probs)>6000:
                    self.history_probs=self.history_probs[-6000:]
                    self.history_labels=self.history_labels[-6000:]
                logger.info(f"[CLOSED] t={ticket} {meta['symbol']} p={profit:.2f} strat={meta['strategy']}")
    def _journal(self, deal, meta, profit, label, prob_snapshot):
        with open(self.journal_path,'a',encoding='utf-8') as f:
            f.write(f"{datetime.utcfromtimestamp(deal.time)},{deal.position_id},{meta['symbol']},{meta['strategy']},{meta['direction']},"
                    f"{meta['volume']},{meta['open_price']},{deal.price},{profit},{meta.get('ml_prob',0)},{label},{prob_snapshot}\n")
    def _perf(self, meta, profit):
        rec=self.strategy_perf[meta['strategy']]
        if profit>0:
            rec["wins"]+=1
            rec["avg_win"]=(1-self.decay_alpha)*rec["avg_win"]+self.decay_alpha*profit
        else:
            rec["losses"]+=1
            rec["avg_loss"]=(1-self.decay_alpha)*rec["avg_loss"]+self.decay_alpha*abs(profit)
        rec["profit"]+=profit
        pf=pip_factor(meta['symbol'])
        approx_pips=(profit/(meta['volume']*10)) if pf>0 else 0
        rec["pips"]+=approx_pips
    def flush_performance(self):
        now=time.time()
        if now - self.last_perf_flush < 120: return
        self.last_perf_flush=now
        with open(self.performance_path,'w',encoding='utf-8') as f:
            f.write("strategy,wins,losses,profit,pips,avg_win,avg_loss\n")
            for s,v in self.strategy_perf.items():
                f.write(f"{s},{v['wins']},{v['losses']},{v['profit']},{v['pips']},{v['avg_win']},{v['avg_loss']}\n")
        if self.ml:
            rep={
                "adaptive_threshold_enabled": self.cfg["ml_ext"]["adaptive_threshold"],
                "timestamp": datetime.utcnow().isoformat()
            }
            try:
                with open(self.ml_report_path,'w',encoding='utf-8') as j: json.dump(rep,j,indent=2)
            except: pass
    def periodic_log(self):
        open_counts=defaultdict(int)
        for m in self.open_meta.values():
            open_counts[m['strategy']]+=1
        if open_counts:
            logger.info(f"Aktivne po strategiji: {dict(open_counts)}")
    def avg_win_loss(self, symbol):
        wins=[]; losses=[]
        for s,v in self.strategy_perf.items():
            wins.append(v["avg_win"]); losses.append(v["avg_loss"])
        avg_win=np.nanmean(wins) if wins else 1.0
        avg_loss=np.nanmean(losses) if losses else 1.0
        return max(0.0001,avg_win), max(0.0001,avg_loss)

# ================== Smooth Trailing Engine ==================
class SmoothTrailingEngine:
    def __init__(self,cfg):
        self.cfg=cfg["trailing"]
        self.last_trail_mod_time={}
        self.mod_counter_minute=defaultdict(int)
    def compute_r_multiple(self, position):
        if position.type==0:
            risk=(position.price_open - position.sl) if position.sl>0 else position.price_open*0.01
            if risk<=0: risk=position.price_open*0.01
            return (position.price_current - position.price_open)/risk
        else:
            risk=(position.sl - position.price_open) if position.sl>0 else position.price_open*0.01
            if risk<=0: risk=position.price_open*0.01
            return (position.price_open - position.price_current)/risk
    def should_update(self, ticket, new_sl, old_sl, side):
        if old_sl==0: return True
        if side=="BUY" and new_sl<=old_sl: return False
        if side=="SELL" and new_sl>=old_sl: return False
        now_min=int(time.time()/60)
        key=(ticket,now_min)
        self.mod_counter_minute[key]+=1
        if self.mod_counter_minute[key]>self.cfg["max_trail_updates_per_min"]:
            return False
        return True
    def calculate_new_sl(self, position, atr, r_mult, aging_minutes):
        c=self.cfg
        if r_mult < c["initial_lock_r_multiple"]:
            return None
        if r_mult >= c["ratchet_r_multiple"]:
            dist=atr*c["ratchet_atr_mult"]
        elif r_mult >= c["tighten_r_multiple"]:
            dist=atr*c["tight_atr_mult"]
        elif r_mult >= c["dynamic_start_r_multiple"]:
            dist=atr*c["base_atr_mult"]
        else:
            dist=None
        if dist and aging_minutes>=c["aging_minutes"]:
            dist*=c["aging_multiplier"]
        side="BUY" if position.type==0 else "SELL"
        if dist is None:
            buffer=atr*0.12
            return position.price_open + buffer if side=="BUY" else position.price_open - buffer
        if side=="BUY":
            candidate=position.price_current - dist
            candidate=max(candidate, position.price_open + atr*0.05)
        else:
            candidate=position.price_current + dist
            candidate=min(candidate, position.price_open - atr*0.05)
        return candidate

# ================== Order Manager ==================
class OrderManager:
    def __init__(self,cfg, mt5c, risk_manager, portfolio, ml_module:EinsteinEnsemble,
                 bayes_score:StrategyBayesScore, high_prob_ctrl:HighProbControl, news_manager=None):
        self.cfg=cfg
        self.mt5=mt5c
        self.risk=risk_manager
        self.portfolio=portfolio
        self.ml=ml_module
        self.last_sig_time_strategy={}
        self.last_sig_time_symbol={}
        self.bayes=bayes_score
        self.high_prob=high_prob_ctrl
        self.news_manager=news_manager
        self.trailing_engine=SmoothTrailingEngine(cfg)
    def process_signals(self, signals, indicator_engine):
        if not signals: return
        acct=self.mt5.account_info()
        balance=acct.balance if acct else 10000
        signals=sorted(signals,key=lambda x:x.get("confidence",0), reverse=True)
        chosen=[]
        for s in signals:
            if len(chosen)>=self.cfg["strategies"]["max_signals_per_cycle"]: break
            if not self._passes_cooldown(s): continue
            if not self._passes_filters(s): continue
            chosen.append(s)
        for sig in chosen:
            sym=sig['symbol']; direction=sig['direction']; tf=sig.get("timeframe","M15")
            prob=sig.get("ml_prob",0.5)
            if not self.risk.can_open(sym): continue
            atr=indicator_engine.get_last(sym, tf, "atr")
            if atr is None or atr<=0: continue
            atr_series=indicator_engine.series(sym, tf, "atr")
            if atr_series is not None and len(atr_series.dropna())>70:
                avg=atr_series.dropna().iloc[-70:].mean()
                if avg>0 and atr>avg*self.cfg["risk"]["volatility_atr_spike_factor"]:
                    continue
            pf=pip_factor(sym)
            atr_pips=atr/pf
            sl_pips=atr_pips*self.cfg["risk"]["sl_atr_multiplier"]
            tp_pips=atr_pips*self.cfg["risk"]["tp_atr_multiplier"]
            lots=dynamic_position_size(balance, self.risk.effective_risk_percent(), sl_pips)
            if lots<=0: continue
            if self.cfg["ml_ext"]["lot_size_scaling"]:
                lots=scale_lot(lots, prob, lo=self.cfg["ml_ext"]["lot_scale_range"][0], hi=self.cfg["ml_ext"]["lot_scale_range"][1])
            tick=mt5.symbol_info_tick(sym)
            if not tick: continue
            price=tick.ask if direction=="BUY" else tick.bid
            if direction=="BUY":
                sl=price - sl_pips*pf
                tp=price + tp_pips*pf
            else:
                sl=price + sl_pips*pf
                tp=price - tp_pips*pf
            comment=f"{sig['strategy']}|{tf}|c={sig.get('confidence',0):.2f}|p={prob:.2f}"
            r=self.mt5.send_order(sym, direction, lots, sl, tp, comment)
            if r and r.retcode==mt5.TRADE_RETCODE_DONE:
                ticket=r.order
                snapshot=indicator_engine.features_snapshot(sym, tf)
                meta={
                    "symbol":sym,"direction":direction,"volume":lots,"sl":sl,"tp":tp,
                    "strategy":sig['strategy'],"confidence":sig.get("confidence",0),
                    "timeframe":tf,"open_time":datetime.utcnow(),"open_price":price,
                    "ml_prob":prob,"features_snapshot":snapshot,
                    "partial_scaled":False,"initial_sl":sl
                }
                self.portfolio.register_open(ticket, meta)
                self._mark_cooldown(sig)
        self._update_active_positions(indicator_engine)
    def _passes_cooldown(self,sig):
        now=time.time()
        sname=sig['strategy']; sym=sig['symbol']
        if now - self.last_sig_time_strategy.get(sname,0) < self.cfg["risk"]["cooldown_seconds_strategy"]: return False
        if now - self.last_sig_time_symbol.get(sym,0) < self.cfg["risk"]["cooldown_seconds_symbol"]: return False
        return True
    def _mark_cooldown(self,sig):
        now=time.time()
        self.last_sig_time_strategy[sig['strategy']]=now
        self.last_sig_time_symbol[sig['symbol']]=now
    def _passes_filters(self,sig):
        sym=sig['symbol']
        if self.news_manager and self.cfg["news"]["enabled"]:
            in_bw,_=self.news_manager.in_blackout(sym)
            if in_bw: return False
        spread=self.mt5.symbol_spread_points(sym)
        if spread is not None and spread>self.cfg["deal_filters"]["max_spread_points"]:
            return False
        hour=datetime.utcnow().hour
        if hour not in self.cfg["deal_filters"]["allowed_hours"]: return False
        wd=datetime.utcnow().weekday()
        if wd==4 and hour>=self.cfg["deal_filters"]["disable_friday_hour_after"]: return False
        if hour==23 and datetime.utcnow().minute >= (60 - self.cfg["deal_filters"]["pause_minutes_before_midnight"]):
            return False
        return True
    def _partial_scale_out(self, position):
        cfg_ps=self.cfg["risk"]["partial_scale"]
        if not cfg_ps["enabled"]: return
        meta=self.portfolio.open_meta.get(position.ticket)
        if not meta or meta.get("partial_scaled",False): return
        entry=position.price_open
        init_sl=meta.get("initial_sl", position.sl)
        if position.type==0:
            risk=entry - init_sl if init_sl>0 else entry*0.01
            if risk<=0: risk=entry*0.01
            r_mult=(position.price_current - entry)/risk
        else:
            risk=init_sl - entry if init_sl>0 else entry*0.01
            if risk<=0: risk=entry*0.01
            r_mult=(entry - position.price_current)/risk
        levels=cfg_ps["levels"]; pct=cfg_ps["scale_percents"]
        if len(levels)!=len(pct): return
        if r_mult>=levels[0]:
            vol=position.volume
            first=vol*pct[0]
            if first>0 and position.volume-first>=0.01:
                self._close_partial(position, first, "partial_scale_1")
            if r_mult>=levels[-1]:
                second=(position.volume - first)*pct[-1]
                if second>0 and position.volume-second>=0.01:
                    self._close_partial(position, second, "partial_scale_2")
                    meta["partial_scaled"]=True
    def _close_partial(self, position, volume, comment="partial"):
        tick=mt5.symbol_info_tick(position.symbol)
        if not tick: return
        price=tick.bid if position.type==0 else tick.ask
        order_type=mt5.ORDER_TYPE_SELL if position.type==0 else mt5.ORDER_TYPE_BUY
        req={
            "action":mt5.TRADE_ACTION_DEAL,
            "symbol":position.symbol,
            "volume":volume,
            "type":order_type,
            "position":position.ticket,
            "price":price,
            "deviation":40,
            "magic":position.magic,
            "comment":comment
        }
        r=mt5.order_send(req)
        if r and r.retcode==mt5.TRADE_RETCODE_DONE:
            logger.info(f"Partial close {comment} vol={volume} ticket={position.ticket}")
    def _update_active_positions(self, indicator_engine):
        positions=self.mt5.positions()
        if not positions: return
        central_tf="M15" if "M15" in self.cfg["general"]["timeframes"] else self.cfg["general"]["timeframes"][0]
        for p in positions:
            symbol=p.symbol
            atr=indicator_engine.get_last(symbol, central_tf, "atr")
            if atr is None or atr<=0: continue
            self._partial_scale_out(p)
            r_mult=self.trailing_engine.compute_r_multiple(p)
            meta=self.portfolio.open_meta.get(p.ticket,{})
            aging_minutes=(datetime.utcnow() - meta.get("open_time",datetime.utcnow())).total_seconds()/60.0
            new_sl=self.trailing_engine.calculate_new_sl(p,atr,r_mult,aging_minutes)
            if new_sl is None: continue
            side="BUY" if p.type==0 else "SELL"
            old_sl=p.sl
            if not self.trailing_engine.should_update(p.ticket,new_sl,old_sl,side):
                continue
            if p.tp>0:
                if side=="BUY" and new_sl>=p.tp: new_sl=p.tp - atr*0.2
                if side=="SELL" and new_sl<=p.tp: new_sl=p.tp + atr*0.2
            self._modify_sl(p.ticket,new_sl,p.tp,symbol)
    def _modify_sl(self,ticket,new_sl,tp,symbol):
        req={"action":mt5.TRADE_ACTION_SLTP,"position":ticket,"symbol":symbol,"sl":new_sl,"tp":tp}
        r=mt5.order_send(req)
        if r and r.retcode==mt5.TRADE_RETCODE_DONE:
            logger.info(f"SL update ticket={ticket} new_sl={new_sl}")

# ================== STRATEGIES ==================
def _atr_slope(ind,symbol,tf,period=10):
    atr_series=ind.series(symbol,tf,"atr")
    if atr_series is None or len(atr_series.dropna())<period+1: return 0
    return (atr_series.iloc[-1]-atr_series.iloc[-period])/period

def s_trend_following_atr(bars, ind, cfg):
    out=[]
    for sym in cfg["general"]["symbols"]:
        tf="H1"; cache=ind.cache.get((sym,tf))
        if not cache: continue
        ef=cache['ema_fast'].iloc[-1]; es=cache['ema_slow'].iloc[-1]; atr_sl=_atr_slope(ind,sym,tf)
        if None in [ef,es]: continue
        if ef>es and atr_sl>=0: out.append({"symbol":sym,"direction":"BUY","confidence":0.58,"timeframe":tf})
        elif ef<es and atr_sl<=0: out.append({"symbol":sym,"direction":"SELL","confidence":0.58,"timeframe":tf})
    return out

def s_mean_reversion_bands(bars, ind, cfg):
    out=[]
    for sym in cfg["general"]["symbols"]:
        tf="M15"; cache=ind.cache.get((sym,tf))
        if not cache: continue
        df=bars.get((sym,tf)); 
        if df is None: continue
        price=df['close'].iloc[-1]
        up=cache['bb_upper'].iloc[-1]; lo=cache['bb_lower'].iloc[-1]; rsi=cache['rsi'].iloc[-1]
        if None in [up,lo,rsi]: continue
        if price>up and rsi<80: out.append({"symbol":sym,"direction":"SELL","confidence":0.6,"timeframe":tf})
        elif price<lo and rsi>20: out.append({"symbol":sym,"direction":"BUY","confidence":0.6,"timeframe":tf})
    return out

def s_breakout_box(bars, ind, cfg):
    out=[]; N=24
    for sym in cfg["general"]["symbols"]:
        tf="M30"; df=bars.get((sym,tf)); cache=ind.cache.get((sym,tf))
        if df is None or cache is None or len(df)<N+10: continue
        pre_range=df.iloc[-N-10:-10]
        compression=pre_range['high'].max()-pre_range['low'].min()
        lastN=df.iloc[-N:]
        box_high=lastN['high'].max(); box_low=lastN['low'].min()
        price=df['close'].iloc[-1]; bbw=cache['bb_width'].iloc[-1]
        if compression/(pre_range['close'].mean())<0.0045 and bbw<0.007:
            if price>box_high: out.append({"symbol":sym,"direction":"BUY","confidence":0.62,"timeframe":tf})
            elif price<box_low: out.append({"symbol":sym,"direction":"SELL","confidence":0.62,"timeframe":tf})
    return out

def s_momentum_rsi(bars, ind, cfg):
    out=[]
    for sym in cfg["general"]["symbols"]:
        tf="M15"; cache=ind.cache.get((sym,tf))
        if not cache: continue
        rsi=cache['rsi'].iloc[-1]; ef=cache['ema_fast'].iloc[-1]; es=cache['ema_slow'].iloc[-1]
        if None in [rsi,ef,es]: continue
        if rsi>70 and ef<es: out.append({"symbol":sym,"direction":"SELL","confidence":0.54,"timeframe":tf})
        elif rsi<30 and ef>es: out.append({"symbol":sym,"direction":"BUY","confidence":0.54,"timeframe":tf})
    return out

def s_vwap_reversion(bars, ind, cfg):
    out=[]
    for sym in cfg["general"]["symbols"]:
        tf="M5"; cache=ind.cache.get((sym,tf))
        if not cache: continue
        df=bars.get((sym,tf)); 
        if df is None: continue
        price=df['close'].iloc[-1]; vwap=cache['vwap'].iloc[-1]
        if vwap is None: continue
        diff=(price-vwap)/vwap
        vol=cache['volatility'].iloc[-1]
        if abs(diff)>0.0022 and (vol is None or vol<0.0045):
            if diff>0: out.append({"symbol":sym,"direction":"SELL","confidence":0.53,"timeframe":tf})
            else: out.append({"symbol":sym,"direction":"BUY","confidence":0.53,"timeframe":tf})
    return out

def s_multi_timeframe_alignment(bars, ind, cfg):
    out=[]
    for sym in cfg["general"]["symbols"]:
        tf1="M15"; tf2="H1"
        i1=ind.cache.get((sym,tf1)); i2=ind.cache.get((sym,tf2))
        if not i1 or not i2: continue
        f1=i1['ema_fast'].iloc[-1]; s1=i1['ema_slow'].iloc[-1]
        f2=i2['ema_fast'].iloc[-1]; s2=i2['ema_slow'].iloc[-1]
        ha=i2['ha_trend'].iloc[-1]
        if None in [f1,s1,f2,s2,ha]: continue
        if f1>s1 and f2>s2 and ha==1:
            out.append({"symbol":sym,"direction":"BUY","confidence":0.68,"timeframe":tf1})
        elif f1<s1 and f2<s2 and ha==-1:
            out.append({"symbol":sym,"direction":"SELL","confidence":0.68,"timeframe":tf1})
    return out

def s_volatility_compression_expansion(bars, ind, cfg):
    out=[]
    for sym in cfg["general"]["symbols"]:
        tf="M30"; cache=ind.cache.get((sym,tf))
        if not cache: continue
        bbw_hist=cache['bb_width'].iloc[-30:].dropna()
        if len(bbw_hist)<10: continue
        current=cache['bb_width'].iloc[-1]
        price=bars[(sym,tf)]['close'].iloc[-1]; mid=cache['bb_mid'].iloc[-1]
        if current<bbw_hist.quantile(0.28):
            if price>mid: out.append({"symbol":sym,"direction":"BUY","confidence":0.59,"timeframe":tf})
            else: out.append({"symbol":sym,"direction":"SELL","confidence":0.59,"timeframe":tf})
    return out

def s_session_range_fade(bars, ind, cfg):
    out=[]
    for sym in cfg["general"]["symbols"]:
        tf="M5"; df=bars.get((sym,tf))
        if df is None or len(df)<50: continue
        last=df.iloc[-1]; hour=last['time'].hour
        if hour not in [7,8,9,13,14,15]: continue
        recent=df.iloc[-36:]
        hi=recent['high'].max(); lo=recent['low'].min(); price=last['close']
        if price>=hi: out.append({"symbol":sym,"direction":"SELL","confidence":0.55,"timeframe":tf})
        elif price<=lo: out.append({"symbol":sym,"direction":"BUY","confidence":0.55,"timeframe":tf})
    return out

def s_pullback_ema(bars, ind, cfg):
    out=[]
    for sym in cfg["general"]["symbols"]:
        tf="M15"; cache=ind.cache.get((sym,tf))
        if not cache: continue
        ef=cache['ema_fast'].iloc[-1]; es=cache['ema_slow'].iloc[-1]; rsi=cache['rsi'].iloc[-1]
        price=bars[(sym,tf)]['close'].iloc[-1]
        if None in [ef,es,rsi]: continue
        if 38<=rsi<=62:
            if ef>es and price<=ef*1.0004:
                out.append({"symbol":sym,"direction":"BUY","confidence":0.6,"timeframe":tf})
            elif ef<es and price>=ef*0.9996:
                out.append({"symbol":sym,"direction":"SELL","confidence":0.6,"timeframe":tf})
    return out

def s_swing_structure(bars, ind, cfg):
    out=[]
    for sym in cfg["general"]["symbols"]:
        tf="H4"; df=bars.get((sym,tf))
        if df is None or len(df)<60: continue
        recent=df.tail(14)
        hi=recent['high'].max(); lo=recent['low'].min(); price=recent['close'].iloc[-1]
        if price>=hi*0.9985: out.append({"symbol":sym,"direction":"SELL","confidence":0.63,"timeframe":tf})
        elif price<=lo*1.0015: out.append({"symbol":sym,"direction":"BUY","confidence":0.63,"timeframe":tf})
    return out

def s_adaptive_kalman_trend(bars, ind, cfg):
    out=[]
    for sym in cfg["general"]["symbols"]:
        tf="M30"; cache=ind.cache.get((sym,tf))
        if not cache: continue
        ema=cache['ema_fast']
        if len(ema)<7: continue
        slope=(ema.iloc[-1]-ema.iloc[-7])/7
        if slope>0: out.append({"symbol":sym,"direction":"BUY","confidence":0.54,"timeframe":tf})
        elif slope<0: out.append({"symbol":sym,"direction":"SELL","confidence":0.54,"timeframe":tf})
    return out

def s_correlation_divergence(bars, ind, cfg):
    out=[]
    if "EURUSD" not in cfg["general"]["symbols"] or "GBPUSD" not in cfg["general"]["symbols"]:
        return out
    tf="M15"
    e=bars.get(("EURUSD",tf)); g=bars.get(("GBPUSD",tf))
    if e is None or g is None or len(e)<12 or len(g)<12: return out
    e_ret=e['close'].pct_change().iloc[-6:].sum()
    g_ret=g['close'].pct_change().iloc[-6:].sum()
    diff=e_ret-g_ret
    out_th=0.0035
    if diff>out_th:
        out+=[{"symbol":"EURUSD","direction":"SELL","confidence":0.56,"timeframe":tf},
              {"symbol":"GBPUSD","direction":"BUY","confidence":0.56,"timeframe":tf}]
    elif diff<-out_th:
        out+=[{"symbol":"EURUSD","direction":"BUY","confidence":0.56,"timeframe":tf},
              {"symbol":"GBPUSD","direction":"SELL","confidence":0.56,"timeframe":tf}]
    return out

def s_orderflow_proxy(bars, ind, cfg):
    out=[]
    for sym in cfg["general"]["symbols"]:
        tf="M1"; df=bars.get((sym,tf))
        if df is None or len(df)<35: continue
        vol_avg=df['tick_volume'].iloc[-12:].mean()
        vol_prev=df['tick_volume'].iloc[-24:-12].mean()
        price=df['close'].iloc[-1]; prev=df['close'].iloc[-2]
        micro_range=(df['high'].iloc[-6:].max()-df['low'].iloc[-6:].min())/price
        if vol_prev and vol_avg>vol_prev*1.5 and micro_range<0.0022:
            if price>prev: out.append({"symbol":sym,"direction":"BUY","confidence":0.62,"timeframe":tf})
            else: out.append({"symbol":sym,"direction":"SELL","confidence":0.62,"timeframe":tf})
    return out

def s_pivot_points_reversal(bars, ind, cfg):
    out=[]
    for sym in cfg["general"]["symbols"]:
        tf="M15"; cache=ind.cache.get((sym,tf))
        if not cache: continue
        pivot=cache['pivot'].iloc[-1]
        price=bars[(sym,tf)]['close'].iloc[-1]
        if pivot is None or (isinstance(pivot,float) and math.isnan(pivot)): continue
        if price>pivot*1.0042:
            out.append({"symbol":sym,"direction":"SELL","confidence":0.53,"timeframe":tf})
        elif price<pivot*0.9958:
            out.append({"symbol":sym,"direction":"BUY","confidence":0.53,"timeframe":tf})
    return out

def s_keltner_breakout(bars, ind, cfg):
    out=[]
    for sym in cfg["general"]["symbols"]:
        tf="M30"; cache=ind.cache.get((sym,tf))
        if not cache: continue
        price=bars[(sym,tf)]['close'].iloc[-1]
        ku=cache['keltner_upper'].iloc[-1]; kl=cache['keltner_lower'].iloc[-1]
        atr=cache['atr'].iloc[-1]
        atr_hist=cache['atr'].iloc[-50:].mean() if cache['atr'].iloc[-50:].notna().sum()>10 else atr
        if atr_hist and atr and atr<atr_hist*0.55: continue
        if price>ku: out.append({"symbol":sym,"direction":"BUY","confidence":0.58,"timeframe":tf})
        elif price<kl: out.append({"symbol":sym,"direction":"SELL","confidence":0.58,"timeframe":tf})
    return out

def s_heiken_ashi_trend(bars, ind, cfg):
    out=[]
    for sym in cfg["general"]["symbols"]:
        tf="H1"; cache=ind.cache.get((sym,tf))
        if not cache: continue
        ha=cache['ha_trend'].iloc[-3:]
        ef=cache['ema_fast'].iloc[-1]; es=cache['ema_slow'].iloc[-1]
        if len(ha)<3 or None in [ef,es]: continue
        if all(v==1 for v in ha) and ef>es:
            out.append({"symbol":sym,"direction":"BUY","confidence":0.57,"timeframe":tf})
        elif all(v==-1 for v in ha) and ef<es:
            out.append({"symbol":sym,"direction":"SELL","confidence":0.57,"timeframe":tf})
    return out

def s_range_scalper_micro(bars, ind, cfg):
    out=[]
    active_hours=set(range(6,11))|set(range(13,18))
    for sym in cfg["general"]["symbols"]:
        tf="M1"; df=bars.get((sym,tf))
        if df is None or len(df)<40: continue
        hour=df['time'].iloc[-1].hour
        if hour not in active_hours: continue
        r=df.iloc[-25:]; hi=r['high'].max(); lo=r['low'].min(); price=r['close'].iloc[-1]
        rng=hi-lo
        if rng==0: continue
        if rng/price<0.0013:
            if price>(hi - rng*0.12):
                out.append({"symbol":sym,"direction":"SELL","confidence":0.5,"timeframe":tf})
            elif price<(lo + rng*0.12):
                out.append({"symbol":sym,"direction":"BUY","confidence":0.5,"timeframe":tf})
    return out

STRATEGY_FUNCS={
    "TrendFollowingATR":s_trend_following_atr,
    "MeanReversionBands":s_mean_reversion_bands,
    "BreakoutBox":s_breakout_box,
    "MomentumRSI":s_momentum_rsi,
    "VWAPReversion":s_vwap_reversion,
    "MultiTimeframeAlignment":s_multi_timeframe_alignment,
    "VolatilityCompressionExpansion":s_volatility_compression_expansion,
    "SessionRangeFade":s_session_range_fade,
    "PullbackEMA":s_pullback_ema,
    "SwingStructure":s_swing_structure,
    "AdaptiveKalmanTrend":s_adaptive_kalman_trend,
    "CorrelationDivergence":s_correlation_divergence,
    "OrderFlowProxy":s_orderflow_proxy,
    "PivotPointsReversal":s_pivot_points_reversal,
    "KeltnerBreakout":s_keltner_breakout,
    "HeikenAshiTrend":s_heiken_ashi_trend,
    "RangeScalperMicro":s_range_scalper_micro
}

# ================== Strategy Engine ==================
class StrategyEngine:
    def __init__(self,cfg,indicator_engine,bayes_score):
        self.cfg=cfg
        self.indicators=indicator_engine
        self.bayes=bayes_score
    def generate_signals(self,bars):
        signals=[]
        weights=self.cfg["strategies"]["weights"]
        for name in self.cfg["strategies"]["enabled"]:
            fn=STRATEGY_FUNCS.get(name)
            if not fn: continue
            try:
                res=fn(bars,self.indicators,self.cfg)
                if not res: continue
                for r in res:
                    base=r.get("confidence",0.5)
                    w=weights.get(name,1.0)
                    r["confidence"]=base*w
                    r["strategy"]=name
                    signals.append(r)
            except Exception as e:
                logger.warning(f"Strategija {name} greška: {e}")
        filtered={}
        for s in signals:
            k=(s['symbol'], s['direction'])
            if k not in filtered or s['confidence']>filtered[k]['confidence']:
                filtered[k]=s
        if self.cfg["ml_ext"]["beta_scoring"]:
            for s in filtered.values():
                q=self.bayes.score(s['strategy'])
                s['confidence']=0.8*s['confidence'] + 0.2*q
        return list(filtered.values())

# ================== Advanced Backtester ==================
class AdvancedBacktester:
    def __init__(self,cfg,data_dir,news_manager=None,enable_ml=True,enable_news=True,
                 initial_balance=10000.0,commission_per_lot=None,slippage_points=None,
                 write_trades=True):
        self.cfg=cfg
        self.data_dir=data_dir
        self.news_manager=news_manager if enable_news else None
        self.enable_ml=enable_ml
        self.enable_news=enable_news
        self.balance=initial_balance
        self.equity=initial_balance
        self.commission_per_lot=commission_per_lot or cfg["backtest"]["commission_per_lot"]
        self.slippage_points=slippage_points or cfg["backtest"]["slippage_points"]
        self.write_trades=write_trades
        self.trades=[]
        self.open_trades=[]
        self.symbol_data={}
        self.indicator_engine=IndicatorEngine(cfg)
        self.strategy_engine=None
        self.ml=None
        self.bayes=StrategyBayesScore()
        self.high_prob=HighProbControl(prob_floor=cfg["ml_ext"]["prob_highbin_floor"],
                                       max_losses=cfg["ml_ext"]["prob_degrade_highbin_loss_count"])
        self.output_file="bt_trades.csv"
        self._load_data()
    def _load_data(self):
        for sym in self.cfg["general"]["symbols"]:
            for tf in self.cfg["general"]["timeframes"]:
                path=os.path.join(self.data_dir,f"{sym}_{tf}.csv")
                if not os.path.exists(path): continue
                df=pd.read_csv(path)
                if 'time' in df.columns:
                    try:
                        if np.issubdtype(df['time'].dtype,np.number):
                            df['time']=pd.to_datetime(df['time'],unit='s',utc=True).dt.tz_convert(None)
                        else:
                            df['time']=pd.to_datetime(df['time'],utc=True).dt.tz_convert(None)
                    except: pass
                df=df[['time','open','high','low','close','tick_volume']].sort_values('time').reset_index(drop=True)
                self.symbol_data[(sym,tf)]=df
        logger.info(f"[BT] Učitano dataset parova={len(self.symbol_data)}")
    def build_common_index(self, primary_tf):
        key_syms=[(s,primary_tf) for s in self.cfg["general"]["symbols"] if (s,primary_tf) in self.symbol_data]
        if not key_syms: raise RuntimeError("Nema primary timeframe podataka.")
        times=set()
        for ks in key_syms:
            times.update(self.symbol_data[ks]['time'].tolist())
        return sorted(list(times))
    def init_components(self):
        self.strategy_engine=StrategyEngine(self.cfg,self.indicator_engine,self.bayes)
        if self.enable_ml and self.cfg["einstein"]["enabled"]:
            self.ml=EinsteinEnsemble(self.cfg)
            focus=[s for s in self.cfg["general"]["symbols"] if (s,self.cfg["einstein"]["primary_timeframe"]) in self.symbol_data]
            offline_df=self._offline_dataset_for_bt(focus)
            self.ml.offline_pretrain(offline_df)
    def _offline_dataset_for_bt(self, symbols):
        class DummyMT5:
            def __init__(self,outer): self.outer=outer
            def get_rates(self,symbol,timeframe,count=2000):
                df=self.outer.symbol_data.get((symbol,timeframe))
                if df is None: return None
                return df.tail(count).copy()
        return build_einstein_dataset(DummyMT5(self), self.cfg, self.indicator_engine, symbols)
    def _simulate_bar_updates(self,current_time):
        snap={}
        for k,df in self.symbol_data.items():
            sub=df[df['time']<=current_time]
            if len(sub)>0: snap[k]=sub.tail(self.cfg["general"]["lookback_bars"])
        self.indicator_engine.update(snap)
        return snap
    def _open_trade(self,symbol,direction,price,atr,pf,strategy,confidence,open_time):
        atr_pips=atr/pf if pf>0 else 0
        sl_pips=atr_pips*self.cfg["risk"]["sl_atr_multiplier"]
        tp_pips=atr_pips*self.cfg["risk"]["tp_atr_multiplier"]
        lots=dynamic_position_size(self.balance,self.cfg["risk"]["account_risk_percent_per_trade"],sl_pips)
        if self.cfg["ml_ext"]["lot_size_scaling"] and self.ml:
            lots=scale_lot(lots,confidence,lo=self.cfg["ml_ext"]["lot_scale_range"][0],hi=self.cfg["ml_ext"]["lot_scale_range"][1])
        if direction=="BUY":
            sl=price - sl_pips*pf
            tp=price + tp_pips*pf
        else:
            sl=price + sl_pips*pf
            tp=price - tp_pips*pf
        self.open_trades.append({
            "symbol":symbol,"dir":direction,"open_time":open_time,"open_price":price,
            "sl":sl,"tp":tp,"volume":lots,"strategy":strategy,"confidence":confidence,
            "ml_prob":confidence if self.ml else None,"active":True,"initial_sl":sl
        })
    def _update_open_trades_on_bar_close(self, bar_dict, current_time):
        closed=[]
        for tr in self.open_trades:
            if not tr["active"]: continue
            key=(tr["symbol"], self.cfg["einstein"]["primary_timeframe"])
            df=bar_dict.get(key)
            if df is None or df.empty: continue
            row=df.iloc[-1]
            high=row['high']; low=row['low']; close=row['close']
            if tr["dir"]=="BUY":
                hit_sl=low<=tr["sl"]; hit_tp=high>=tr["tp"]
                if hit_sl and hit_tp: exit_price=tr["sl"]
                elif hit_sl: exit_price=tr["sl"]
                elif hit_tp: exit_price=tr["tp"]
                else: exit_price=close
                profit=(exit_price - tr["open_price"])*(tr["volume"]*100000 if tr["symbol"].endswith("USD") else tr["volume"]*1000)
            else:
                hit_sl=high>=tr["sl"]; hit_tp=low<=tr["tp"]
                if hit_sl and hit_tp: exit_price=tr["sl"]
                elif hit_sl: exit_price=tr["sl"]
                elif hit_tp: exit_price=tr["tp"]
                else: exit_price=close
                profit=(tr["open_price"] - exit_price)*(tr["volume"]*100000 if tr["symbol"].endswith("USD") else tr["volume"]*1000)
            tr["close_time"]=current_time
            tr["close_price"]=exit_price
            tr["profit"]=profit - self.commission_per_lot*tr["volume"]
            tr["active"]=False
            closed.append(tr)
        self.open_trades=[t for t in self.open_trades if t["active"]]
        for c in closed:
            self.trades.append(c)
            self.balance+=c["profit"]
            self.equity=self.balance
        return closed
    def run(self):
        primary_tf=self.cfg["einstein"]["primary_timeframe"]
        index=self.build_common_index(primary_tf)
        logger.info(f"[BT] Start bars={len(index)} tf={primary_tf}")
        for i,current_time in enumerate(index):
            bar_dict=self._simulate_bar_updates(current_time)
            signals=self.strategy_engine.generate_signals(bar_dict)
            if self.ml and self.enable_ml:
                self.ml.annotate_signals(signals,self.indicator_engine)
                for s in signals:
                    s['confidence']=self.high_prob.adjust_confidence(s['confidence'])
            exec_signals=[]
            for s in signals:
                if self.news_manager:
                    in_bw,_=self.news_manager.in_blackout(s['symbol'], current_time)
                    if in_bw: continue
                exec_signals.append(s)
            for s in sorted(exec_signals,key=lambda x:x.get("confidence",0),
                            reverse=True)[:self.cfg["strategies"]["max_signals_per_cycle"]]:
                atr=self.indicator_engine.get_last(s['symbol'], s.get("timeframe","M15"), "atr")
                price=bar_dict.get((s['symbol'],primary_tf), pd.DataFrame()).tail(1)['close']
                if atr is None or price.empty: continue
                pf=pip_factor(s['symbol'])
                self._open_trade(s['symbol'], s['direction'], float(price.values[0]), atr, pf, s['strategy'], s['confidence'], current_time)
            closed=self._update_open_trades_on_bar_close(bar_dict,current_time)
            if i%250==0:
                logger.info(f"[BT] progress {i}/{len(index)} equity={self.equity:.2f} open={len(self.open_trades)}")
        logger.info(f"[BT] Done trades={len(self.trades)} final_balance={self.balance:.2f}")
        if self.write_trades: self._write_results()
        return self.trades
    def _write_results(self):
        if not self.trades: return
        import csv
        with open(self.output_file,"w",newline="",encoding="utf-8") as f:
            w=csv.writer(f)
            w.writerow(["symbol","dir","open_time","open_price","close_time","close_price","volume","strategy","confidence","ml_prob","profit"])
            for t in self.trades:
                w.writerow([t["symbol"],t["dir"],t["open_time"],t["open_price"],t["close_time"],t["close_price"],t["volume"],t["strategy"],t.get("confidence"),t.get("ml_prob"),t.get("profit")])
        logger.info(f"[BT] Rezultati zapisani u {self.output_file}")

# ================== MAIN LOOP CONTROL ==================
RUNNING=True
def _sig_handler(sig, frame):
    global RUNNING
    logger.warning("Signal primljen – shutdown...")
    RUNNING=False
signal.signal(signal.SIGINT, _sig_handler)
signal.signal(signal.SIGTERM, _sig_handler)

# ================== MAIN ==================
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--mode",default="live",choices=["live","demo","backtest"])
    parser.add_argument("--symbols",default="")
    parser.add_argument("--risk-per-trade",type=float,default=None)
    parser.add_argument("--max-daily-loss-pct",type=float,default=None)
    parser.add_argument("--bt-start",default="")
    parser.add_argument("--bt-end",default="")
    parser.add_argument("--einstein",action="store_true")
    parser.add_argument("--news-file",default="")
    parser.add_argument("--news-blackout-min-before",type=int,default=None)
    parser.add_argument("--news-blackout-min-after",type=int,default=None)
    parser.add_argument("--data-dir",default="data")
    parser.add_argument("--bt-cash",type=float,default=10000.0)
    parser.add_argument("--bt-commission-per-lot",type=float,default=None)
    parser.add_argument("--bt-slippage-points",type=int,default=None)
    parser.add_argument("--bt-enable-news",action="store_true")
    parser.add_argument("--bt-enable-ml",action="store_true")
    parser.add_argument("--bt-write-trades",action="store_true")
    parser.add_argument("--allow-weekend-trading",action="store_true",help="Allow trading during weekends (Saturday/Sunday)")
    args=parser.parse_args()

    cfg=BASE_CONFIG
    apply_env_overrides(cfg)
    if args.symbols:
        cfg["general"]["symbols"]=args.symbols.split(",")
    if args.risk_per_trade is not None:
        cfg["risk"]["account_risk_percent_per_trade"]=args.risk_per_trade
    if args.max_daily_loss_pct is not None:
        cfg["risk"]["daily_loss_limit_pct"]=args.max_daily_loss_pct
        cfg["risk"]["disable_daily_loss_limit"]=False
    if args.einstein:
        cfg["einstein"]["enabled"]=True
    if args.news_blackout_min_before is not None:
        cfg["news"]["blackout_minutes_before"]=args.news_blackout_min_before
    if args.news_blackout_min_after is not None:
        cfg["news"]["blackout_minutes_after"]=args.news_blackout_min_after

    global logger
    logger=setup_logger(cfg)
    logger.info("=== STARSHIP EINSTEIN FULL V2.2 LIGHT (DEMO CREDS) INIT ===")

    # Weekend trading shutdown check (live/demo modes only)
    if args.mode in ["live", "demo"] and is_weekend() and not args.allow_weekend_trading:
        logger.warning("Weekend trading disabled (Saturday/Sunday). Use --allow-weekend-trading to override.")
        weekend_shutdown_marker = os.path.join(cfg["logging"]["log_dir"], "weekend_shutdown.marker")
        os.makedirs(cfg["logging"]["log_dir"], exist_ok=True)
        with open(weekend_shutdown_marker, "w") as f:
            f.write(f"Weekend shutdown: {datetime.utcnow().isoformat()}\n")
            f.write(f"Mode: {args.mode}\n")
            f.write("Use --allow-weekend-trading to override weekend shutdown.\n")
        logger.info(f"Weekend shutdown marker created: {weekend_shutdown_marker}")
        return

    if args.mode=="backtest":
        news_manager=None
        if args.bt_enable_news and args.news_file:
            news_manager=NewsManager(cfg,path=args.news_file)
            logger.info(f"[BT] Učitano news eventa: {len(news_manager.events)}")
        bt=AdvancedBacktester(
            cfg=cfg,
            data_dir=args.data_dir,
            news_manager=news_manager,
            enable_ml=args.bt_enable_ml,
            enable_news=args.bt_enable_news,
            initial_balance=args.bt_cash,
            commission_per_lot=args.bt_commission_per_lot,
            slippage_points=args.bt_slippage_points,
            write_trades=args.bt_write_trades
        )
        bt.init_components()
        bt.run()
        return

    mt5c=MT5Connector()
    if not mt5c.initialize():
        logger.error("MT5 inicijalizacija nije uspjela.")
        return
    data_feed=DataFeed(mt5c,cfg)
    data_feed.bootstrap()
    indicator_engine=IndicatorEngine(cfg)
    bayes_score=StrategyBayesScore()
    high_prob_ctrl=HighProbControl(prob_floor=cfg["ml_ext"]["prob_highbin_floor"],
                                   max_losses=cfg["ml_ext"]["prob_degrade_highbin_loss_count"])
    news_manager=None
    if cfg["news"]["enabled"] and args.news_file:
        news_manager=NewsManager(cfg,path=args.news_file)
        logger.info(f"[LIVE] Učitano news eventa: {len(news_manager.events)}")

    einstein_ml=None
    if cfg["ml"]["enabled"] and cfg["einstein"]["enabled"]:
        einstein_ml=EinsteinEnsemble(cfg)
        focus=[s for s in cfg["general"]["symbols"] if s in ["EURUSD","GBPUSD","USDJPY","XAUUSD","BTCUSD","AUDUSD"]]
        offline_df=build_einstein_dataset(mt5c,cfg,indicator_engine,focus)
        einstein_ml.offline_pretrain(offline_df)
    elif cfg["ml"]["enabled"]:
        einstein_ml=EinsteinEnsemble(cfg)
        logger.info("[EINSTEIN] Fallback global only.")

    strategy_engine=StrategyEngine(cfg,indicator_engine,bayes_score)
    risk_manager=RiskManager(cfg,mt5c)
    portfolio=PortfolioTracker(cfg,mt5c,einstein_ml,bayes_score,high_prob_ctrl)
    order_manager=OrderManager(cfg, mt5c, risk_manager, portfolio, einstein_ml, bayes_score, high_prob_ctrl, news_manager=news_manager)

    poll=cfg["general"]["poll_interval_seconds"]
    logger.info("Launch OK. Live loop start (Light Mode).")
    cycle=0
    while RUNNING:
        started=time.time()
        try:
            risk_manager.daily_reset_if_needed()
            bars=data_feed.update_all()
            indicator_engine.update(bars)
            portfolio.sync_closed(risk_manager)
            if einstein_ml: einstein_ml.maybe_retrain()
            signals=strategy_engine.generate_signals(bars)
            if einstein_ml:
                einstein_ml.annotate_signals(signals,indicator_engine)
                for s in signals:
                    s['confidence']=high_prob_ctrl.adjust_confidence(s['confidence'])
            order_manager.process_signals(signals,indicator_engine)
            portfolio.periodic_log()
            portfolio.flush_performance()
            cycle+=1
            elapsed=time.time()-started
            if elapsed<poll:
                time.sleep(poll-elapsed)
            else:
                logger.warning(f"Cycle lag {elapsed:.2f}s > poll {poll}s")
        except Exception as e:
            logger.error(f"Loop error: {e}\n{traceback.format_exc()}")
            time.sleep(2)

    logger.info("Shutdown sekvenca...")
    portfolio.flush_performance()
    if einstein_ml:
        einstein_ml._persist_global()
        for sym in einstein_ml.symbol_models.keys():
            einstein_ml._persist_symbol(sym)
    mt5c.shutdown()
    logger.info("Gotovo. Starship Einstein Full V2.2 Light (Demo) se spustio.")

if __name__=="__main__":
    logger=setup_logger(BASE_CONFIG)
    main()