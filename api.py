"""FastAPI service exposing selected stock analytics and AI endpoints.

Run (PowerShell):
  .venv\Scripts\python.exe -m uvicorn api:app --reload --port 8000

OpenAPI docs:
  http://127.0.0.1:8000/docs
  http://127.0.0.1:8000/redoc
"""
from __future__ import annotations
import os
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
import pandas as pd
import yfinance as yf

# Reuse logic by importing specific functions from main without triggering Streamlit app.
# We guard the import to only pull helper functions.
from main import get_stock_data, build_trend_snapshot, calculate_rsi  # type: ignore

app = FastAPI(title="Predictor Stock API", version="0.1.0", description="Stock data + trend snapshot + minimal AI placeholder.")

DEFAULT_UNIVERSE = [
    "YESBANK.NS", "SUZLON.NS", "PNB.NS", "IDEA.NS", "RPOWER.NS", "JPPOWER.NS",
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "ICICIBANK.NS"
]

class Bar(BaseModel):
    date: str
    open: float
    high: float
    low: float
    close: float
    volume: int

class StockSeries(BaseModel):
    symbol: str
    bars: List[Bar]

class TrendRecord(BaseModel):
    symbol: str
    last: float
    d1: Optional[float]
    d5: Optional[float]
    d20: Optional[float]
    rsi14: float
    ma_state: str
    vol_ratio: Optional[float]
    stale: bool

class TrendSnapshot(BaseModel):
    generated_at: str
    summary_lines: List[str]
    tickers: List[TrendRecord]

@app.get("/health", tags=["system"])
async def health():
    return {"status": "ok"}

@app.get("/stock/{symbol}", response_model=StockSeries, tags=["data"])
async def get_stock(symbol: str, period: str = Query("6mo", regex="^(1mo|3mo|6mo|1y|2y)$")):
    df = get_stock_data(symbol, period=period)
    if df.empty:
        raise HTTPException(status_code=404, detail="Symbol not found or no data")
    rows = []
    for _, r in df.tail(400).iterrows():
        rows.append(Bar(
            date=str(r.get("Date")),
            open=float(r.get("Open")),
            high=float(r.get("High")),
            low=float(r.get("Low")),
            close=float(r.get("Close")),
            volume=int(r.get("Volume") or 0)
        ))
    return StockSeries(symbol=symbol, bars=rows)

@app.get("/trends", response_model=TrendSnapshot, tags=["trends"])
async def trends(symbols: Optional[str] = Query(None, description="Comma-separated list of symbols")):
    if symbols:
        syms = [s.strip() for s in symbols.split(",") if s.strip()]
    else:
        syms = DEFAULT_UNIVERSE
    snap = build_trend_snapshot(syms)
    # Convert dict to response_model
    return TrendSnapshot(
        generated_at=snap['generated_at'],
        summary_lines=snap['summary_lines'],
        tickers=[TrendRecord(**t) for t in snap['tickers']]
    )

@app.get("/trends/simple", tags=["trends"], summary="Plaintext quick trend lines")
async def trends_plain(symbols: Optional[str] = None):
    if symbols:
        syms = [s.strip() for s in symbols.split(",") if s.strip()]
    else:
        syms = DEFAULT_UNIVERSE
    snap = build_trend_snapshot(syms)
    lines = [f"Generated: {snap['generated_at']}"] + snap['summary_lines']
    for rec in snap['tickers'][:25]:
        lines.append(
            f"{rec['symbol']}: Last {rec['last']:.2f} | 1d {rec['d1']:+.2f}% | 5d {rec['d5']:+.2f}% | 20d {rec['d20']:+.2f}% | RSI14 {rec['rsi14']:.1f} | {rec['ma_state']}"
        )
    return {"text": "\n".join(lines)}

# Placeholder AI endpoint (does not call OpenAI/Ollama here to keep API stateless by default)
class AskRequest(BaseModel):
    question: str
    symbols: Optional[List[str]] = None

@app.post("/ask", tags=["ai"])
async def ask(req: AskRequest):
    syms = req.symbols or DEFAULT_UNIVERSE
    snap = build_trend_snapshot(syms)
    # Very naive heuristic summary (no external LLM call to keep this endpoint safe)
    gains = sorted([t for t in snap['tickers'] if t['d1'] is not None], key=lambda x: x['d1'], reverse=True)[:3]
    losses = sorted([t for t in snap['tickers'] if t['d1'] is not None], key=lambda x: x['d1'])[:3]
    answer_lines = ["(Local heuristic summary — no AI model invoked)"]
    answer_lines.append("Top intraday gainers: " + ", ".join(f"{g['symbol']} {g['d1']:+.2f}%" for g in gains))
    answer_lines.append("Top intraday losers: " + ", ".join(f"{l['symbol']} {l['d1']:+.2f}%" for l in losses))
    answer_lines.append(f"Your question: {req.question}")
    return {"answer": "\n".join(answer_lines), "snapshot_generated_at": snap['generated_at']}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)
