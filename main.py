#!/usr/bin/env python3
# coding: utf-8

import argparse
import sys
import os
import psutil
from typing import List, Dict, Any, Optional
from typing_extensions import Annotated
from datetime import datetime
from contextlib import asynccontextmanager

import baostock as bs
from fastmcp import FastMCP
from fastmcp.server.http import create_sse_app
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

_kline_cache: Dict[str, List[Dict[str, Any]]] = {}

def normalize_code_for_baostock(code: str) -> str:
    c = code.strip().lower()
    if "." in c and (c.startswith("sz.") or c.startswith("sh.")):
        return c
    if len(c) == 6 and c.isdigit():
        pre3 = c[:3]
        if pre3 in {"600","601","603","605","688","689","900"}:
            return f"sh.{c}"
        if pre3 in {"000","001","002","003","004","200","300","301","302"}:
            return f"sz.{c}"
        return f"sz.{c}"
    return c

def get_5min_kline(
    stock_code: str,
    start_date: str = "2024-01-01",
    end_date: str = "2025-12-31"
) -> List[Dict[str, Any]]:
    key = f"{stock_code}|{start_date}|{end_date}"
    if key in _kline_cache:
        return _kline_cache[key]

    lg = bs.login()
    if lg.error_code != '0':
        raise RuntimeError(f"Baostock 登录失败: {lg.error_msg}")

    fields = "date,code,time,open,high,low,close,volume,amount,adjustflag"
    rs = bs.query_history_k_data_plus(
        stock_code,
        fields,
        start_date=start_date,
        end_date=end_date,
        frequency="5",
        adjustflag="3"
    )
    if rs.error_code != '0':
        bs.logout()
        raise RuntimeError(f"Baostock 查询失败: {rs.error_msg}")

    data: List[Dict[str, Any]] = []
    while rs.error_code == '0' and rs.next():
        row = rs.get_row_data()
        date_str, code, time_str, open_, high, low, close, volume, amount, _ = row
        datetime_str = f"{date_str} {time_str}"
        try:
            open_f = float(open_)
            high_f = float(high)
            low_f = float(low)
            close_f = float(close)
            volume_i = int(float(volume))
            amount_f = float(amount)
        except Exception as ex:
            print(f"⚠️ 跳过无效数据: {row} -> {ex}", file=sys.stderr)
            continue

        data.append({
            "code": code,
            "datetime": datetime_str,
            "time": time_str,
            "open": open_f,
            "high": high_f,
            "low": low_f,
            "close": close_f,
            "volume": volume_i,
            "amount": amount_f
        })

    bs.logout()
    _kline_cache[key] = data
    return data

def sim_grid(
    initial_capital: float,
    init_shares: float,
    grid: float,
    shares_per_trade: int,
    kdata: List[Dict[str, Any]],
    fee_rate: float = 0.0003,
    fixed_fee: float = 5.0
) -> Dict[str, Any]:
    cash = float(initial_capital)
    shares = float(init_shares)
    baseline = kdata[0]['open']

    total_fee = 0.0
    trades = 0
    trade_records: List[Dict[str, Any]] = []

    for bar in kdata:
        high_price = bar.get('high')
        low_price = bar.get('low')
        up_grid = baseline + grid
        down_grid = baseline - grid
        dt = bar.get('datetime')

        if low_price < up_grid < high_price:
            if shares >= shares_per_trade:
                price = up_grid
                trade_amount = price * shares_per_trade
                fee = trade_amount * fee_rate + fixed_fee
                cash += (trade_amount - fee)
                shares -= shares_per_trade
                total_fee += fee
                trades += 1
                trade_records.append({
                    "datetime": dt,
                    "action": "SELL",
                    "price": price,
                    "shares": shares_per_trade,
                    "fee": round(fee,4),
                    "cash_after": round(cash,4),
                    "shares_after": shares
                })
                baseline = up_grid

        elif low_price < down_grid < high_price:
            price = down_grid
            trade_amount = price * shares_per_trade
            fee = trade_amount * fee_rate + fixed_fee
            if cash >= (trade_amount + fee):
                cash -= (trade_amount + fee)
                shares += shares_per_trade
                total_fee += fee
                trades += 1
                trade_records.append({
                    "datetime": dt,
                    "action": "BUY",
                    "price": price,
                    "shares": shares_per_trade,
                    "fee": round(fee,4),
                    "cash_after": round(cash,4),
                    "shares_after": shares
                })
                baseline = down_grid

    final_price = kdata[-1]['close']
    final_capital = cash + shares * final_price
    first_open = kdata[0]['open']
    initial_cost = init_shares * first_open
    total_profit = final_capital - initial_capital - initial_cost

    return {
        "final_capital": round(final_capital,4),
        "total_profit": round(total_profit,4),
        "total_fee": round(total_fee,4),
        "trades": int(trades),
        "remaining_shares": float(shares),
        "cash": round(cash,4),
        "trade_records": trade_records,
        "initial_cost_shares": float(init_shares),
        "initial_baseline": float(first_open),
        "final_price": float(final_price)
    }

# 创建两个 MCP 服务实例：HTTP 和 SSE
mcp_http = FastMCP("GridTradeSim_HTTP")
mcp_sse  = FastMCP("GridTradeSim_SSE")

# HTTP 模式工具
@mcp_http.tool()
def simulate_grid_tool_http(
    code: Annotated[str, "股票代码，如 '600000' 或 'sh.600000'"],
    start: Annotated[str, "起始日期，格式 YYYY-MM-DD"] = "2024-01-01",
    end: Annotated[str, "结束日期，格式 YYYY-MM-DD"] = "2025-12-31",
    capital: Annotated[float, "初始资金总额（元）"] = 100000.0,
    base_ratio: Annotated[float, "初次买入占资金比例（0-1之间）"] = 0.30,
    grid: Annotated[float, "网格价格间隔（元）"] = 0.2,
    trade_size: Annotated[int, "每次交易份额（股数）"] = 2000,
    fee_rate: Annotated[float, "交易费率（如0.0003）"] = 0.0003,
    fixed_fee: Annotated[float, "固定交易费用（元）"] = 5.0
) -> Dict[str, Any]:
    std_code = normalize_code_for_baostock(code)
    kdata = get_5min_kline(std_code, start, end)
    if not kdata:
        raise RuntimeError(f"{std_code} 在区间 {start} ~ {end} 无有效 K 线数据")
    first_open = kdata[0]['open']
    init_shares_float = (capital * base_ratio) / first_open
    init_shares = int(init_shares_float // trade_size * trade_size)
    init_cash = capital - init_shares * first_open
    result = sim_grid(
        initial_capital=init_cash,
        init_shares=init_shares,
        grid=grid,
        shares_per_trade=trade_size,
        kdata=kdata,
        fee_rate=fee_rate,
        fixed_fee=fixed_fee
    )
    return {
        "input": {
            "code": code,
            "normalized_code": std_code,
            "start": start,
            "end": end,
            "capital": capital,
            "base_ratio": base_ratio,
            "grid": grid,
            "trade_size": trade_size,
            "fee_rate": fee_rate,
            "fixed_fee": fixed_fee,
            "first_open": round(first_open,4),
            "init_shares": init_shares,
            "init_cash": round(init_cash,4),
        },
        "result": result
    }

# SSE 模式工具
@mcp_sse.tool()
def simulate_grid_tool_sse(
    code: Annotated[str, "股票代码，如 '600000' 或 'sh.600000'"],
    start: Annotated[str, "起始日期，格式 YYYY-MM-DD"] = "2024-01-01",
    end: Annotated[str, "结束日期，格式 YYYY-MM-DD"] = "2025-12-31",
    capital: Annotated[float, "初始资金总额（元）"] = 100000.0,
    base_ratio: Annotated[float, "初次买入占资金比例（0‐1之间）"] = 0.30,
    grid: Annotated[float, "网格价格间隔（元）"] = 0.2,
    trade_size: Annotated[int, "每次交易份额（股数）"] = 2000,
    fee_rate: Annotated[float, "交易费率（如0.0003）"] = 0.0003,
    fixed_fee: Annotated[float, "固定交易费用（元）"] = 5.0
) -> Dict[str, Any]:
    std_code = normalize_code_for_baostock(code)
    kdata = get_5min_kline(std_code, start, end)
    if not kdata:
        raise RuntimeError(f"{std_code} 在区间 {start} ~ {end} 无有效 K 线数据")
    first_open = kdata[0]['open']
    init_shares_float = (capital * base_ratio) / first_open
    init_shares = int(init_shares_float // trade_size * trade_size)
    init_cash = capital - init_shares * first_open
    result = sim_grid(
        initial_capital=init_cash,
        init_shares=init_shares,
        grid=grid,
        shares_per_trade=trade_size,
        kdata=kdata,
        fee_rate=fee_rate,
        fixed_fee=fixed_fee
    )
    return {
        "input": {
            "code": code,
            "normalized_code": std_code,
            "start": start,
            "end": end,
            "capital": capital,
            "base_ratio": base_ratio,
            "grid": grid,
            "trade_size": trade_size,
            "fee_rate": fee_rate,
            "fixed_fee": fixed_fee,
            "first_open": round(first_open,4),
            "init_shares": init_shares,
            "init_cash": round(init_cash,4),
        },
        "result": result
    }

# 还可以加更多工具 …
@mcp_http.tool()
def get_current_time_http() -> str:
    return datetime.utcnow().isoformat() + "Z"

@mcp_sse.tool()
def get_current_time_sse() -> str:
    return datetime.utcnow().isoformat() + "Z"

def main():
    parser = argparse.ArgumentParser(
        description="GridTradeSim MCP Server – HTTP (/mcp) 和 SSE (/sse) 双挂载。",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--host", default="0.0.0.0", help="监听主机 (默认: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=9898, help="监听端口 (默认: 9898)")
    parser.add_argument("--stdio", action="store_true", help="以 stdio 模式运行（不启动 HTTP/SSE 服务）")
    args = parser.parse_args()

    if args.stdio:
        # stdio 模式，可选择一个实例运行
        mcp_http.run()
    else:
        # 创建 HTTP 子应用与 SSE 子应用
        http_sub_app = mcp_http.http_app(path="/", middleware=None)
        sse_sub_app  = create_sse_app(server=mcp_sse, message_path="/message", sse_path="/", middleware=None)

        # 合并 lifespan：确保两个子应用的生命周期都执行
        @asynccontextmanager
        async def combined_lifespan(app: FastAPI):
            async with http_sub_app.lifespan(http_sub_app):
                async with sse_sub_app.lifespan(sse_sub_app):
                    yield

        app = FastAPI(lifespan=combined_lifespan)

        # CORS 中间件
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=False,
            allow_methods=["*"],
            allow_headers=["*"],
            expose_headers=["*"],
        )

        # 挂载两个路径
        app.mount("/mcp", http_sub_app)
        app.mount("/sse", sse_sub_app)

        import uvicorn
        uvicorn.run(app, host=args.host, port=args.port, log_level="info", access_log=True)

if __name__ == "__main__":
    main()
