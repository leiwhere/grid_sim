#!/usr/bin/env python3
# coding: utf-8
"""
MCP Server (FastMCP) with CORS (跨域) + SSE 预检 OPTIONS 支持
提供：
- 工具 simulate_grid：网格交易模拟（含费率 + 固定费用叠加）
- 工具 get_current_time：获取当前 UTC 时间（ISO8601）
- 工具 cache_status_full：查看内存缓存 + 当前进程 & 系统内存使用
- 工具 clear_cache：清理内存缓存
启动方式支持命令行参数：
  --transport (stdio|http|sse)、--host、--port
  --no-auto-clear （关闭自动清理缓存，默认开启）
  --auto-clear-proc-mb (进程内存 MB 阈值)、--auto-clear-sys-avail-mb (系统可用内存 MB 阈值)
加入跨域支持：当使用 HTTP 或 SSE 模式时，允许所有来源访问，并在 SSE 模式下显式响应 OPTIONS 预检请求。
"""

import argparse
import sys
import os
import psutil
from typing import List, Dict, Any, Optional
from datetime import datetime

import baostock as bs
from fastmcp import FastMCP
from fastapi.responses import PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware

# ------------------------
# 内存缓存（运行期间有效）
_kline_cache: Dict[str, List[Dict[str, Any]]] = {}

_AUTO_CLEAR_ENABLED = True
_AUTO_CLEAR_PROC_MB_THRESHOLD = 300.0
_AUTO_CLEAR_SYS_AVAILABLE_MB_THRESHOLD = 200.0

def normalize_code_for_baostock(code: str) -> str:
    c = code.strip().lower()
    if "." in c and (c.startswith("sz.") or c.startswith("sh.")):
        return c
    if len(c) == 6 and c.isdigit():
        pre3 = c[:3]
        if pre3 in {"600", "601", "603", "605", "688", "689", "900"}:
            return f"sh.{c}"
        if pre3 in {"000", "001", "002", "003", "004", "200", "300", "301", "302"}:
            return f"sz.{c}"
        return f"sz.{c}"
    return c

def _maybe_auto_clear_cache() -> None:
    if not _AUTO_CLEAR_ENABLED:
        return
    proc = psutil.Process(os.getpid())
    proc_mem_mb = proc.memory_info().rss / (1024 * 1024)
    vmem = psutil.virtual_memory()
    sys_available_mb = vmem.available / (1024 * 1024)
    if proc_mem_mb > _AUTO_CLEAR_PROC_MB_THRESHOLD or sys_available_mb < _AUTO_CLEAR_SYS_AVAILABLE_MB_THRESHOLD:
        print(f"自动清理触发：proc_mem_mb={proc_mem_mb:.2f} MB, sys_available_mb={sys_available_mb:.2f} MB", file=sys.stderr)
        _kline_cache.clear()

def get_5min_kline(
    stock_code: str,
    start_date: str = "2024-01-01",
    end_date: str = "2025-12-31"
) -> List[Dict[str, Any]]:
    key = f"{stock_code}|{start_date}|{end_date}"
    if key in _kline_cache:
        return _kline_cache[key]

    _maybe_auto_clear_cache()

    lg = bs.login()
    if lg.error_code != '0':
        raise RuntimeError(f"Baostock login failed: {lg.error_msg}")

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
        raise RuntimeError(f"Baostock query failed: {rs.error_msg}")

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
            print(f"⚠️ 跳过一条数据（转换失败）: {row} -> {ex}", file=sys.stderr)
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
                    "cash_after": cash,
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
                    "cash_after": cash,
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

# ------------------------
mcp = FastMCP("GridTradeSim")

@mcp.tool()
def simulate_grid_tool(
    code: str,
    start: str = "2024-01-01",
    end: str = "2025-12-31",
    capital: float = 100000.0,
    base_ratio: float = 0.30,
    grid: float = 0.2,
    trade_size: int = 2000,
    fee_rate: float = 0.0003,
    fixed_fee: float = 5.0
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

@mcp.tool()
def get_current_time_tool() -> str:
    return datetime.utcnow().isoformat() + "Z"

@mcp.tool()
def cache_status_full_tool() -> Dict[str, Any]:
    status = { key: len(_kline_cache[key]) for key in _kline_cache }
    proc = psutil.Process(os.getpid())
    proc_mem_bytes = proc.memory_info().rss
    proc_mem_mb = round(proc_mem_bytes / (1024 * 1024), 4)

    vmem = psutil.virtual_memory()
    sys_total_bytes = vmem.total
    sys_total_mb = round(sys_total_bytes / (1024 * 1024), 4)
    sys_available_bytes = vmem.available
    sys_available_mb = round(sys_available_bytes / (1024 * 1024), 4)

    return {
        "cached_keys": list(_kline_cache.keys()),
        "counts": status,
        "total_keys": len(_kline_cache),
        "proc_mem_bytes": proc_mem_bytes,
        "proc_mem_mb": proc_mem_mb,
        "sys_total_bytes": sys_total_bytes,
        "sys_total_mb": sys_total_mb,
        "sys_available_bytes": sys_available_bytes,
        "sys_available_mb": sys_available_mb
    }

@mcp.tool()
def clear_cache_tool(key: Optional[str] = None) -> str:
    if key:
        if key in _kline_cache:
            del _kline_cache[key]
            return f"已清除缓存项：{key}"
        else:
            return f"未找到缓存项：{key}"
    else:
        _kline_cache.clear()
        return "已清除全部缓存"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GridTradeSim MCP Server with CORS + SSE OPTIONS 支持")
    parser.add_argument(
        "--transport", choices=["stdio","http","sse"], default="http",
        help="传输方式（stdio | http | sse），默认 http"
    )
    parser.add_argument(
        "--host", default="0.0.0.0",
        help="监听主机（http/sse 有效），默认 0.0.0.0"
    )
    parser.add_argument(
        "--port", type=int, default=9898,
        help="监听端口（http/sse 有效），默认 9898"
    )
    parser.add_argument(
        "--no-auto-clear", action="store_true",
        help="关闭自动清理缓存（默认开启）"
    )
    parser.add_argument(
        "--auto-clear-proc-mb", type=float,
        help=f"进程占用内存触发自动清理的阈值（MB），默认 {_AUTO_CLEAR_PROC_MB_THRESHOLD}"
    )
    parser.add_argument(
        "--auto-clear-sys-available-mb", type=float,
        help=f"系统可用内存触发自动清理的阈值（MB），默认 {_AUTO_CLEAR_SYS_AVAILABLE_MB_THRESHOLD}"
    )

    args = parser.parse_args()

    if args.no_auto_clear:
        _AUTO_CLEAR_ENABLED = False
    if args.auto_clear_proc_mb is not None:
        _AUTO_CLEAR_PROC_MB_THRESHOLD = args.auto_clear_proc_mb
    if args.auto_clear_sys_available_mb is not None:
        _AUTO_CLEAR_SYS_AVAILABLE_MB_THRESHOLD = args.auto_clear_sys_available_mb

    # **不带凭据，允许任意来源**
    if args.transport in ("http","sse"):
        try:
            mcp.app.add_middleware(
                CORSMiddleware,
                allow_origins=["*"],        # 允许所有来源
                allow_credentials=False,    # 🟢 关键改动：关闭凭据
                allow_methods=["*"],
                allow_headers=["*"],
                expose_headers=["*"],
                max_age=86400,              # 可选：预检缓存时间（秒）
            )
        except Exception as e:
            print(f"⚠️ 未能为 FastMCP 添加 CORS: {e}", file=sys.stderr)

        # 如果是 SSE 模式（或你专门想处理 OPTIONS），你也可以保留下面这一段：
        if args.transport == "http":
            @mcp.app.options("/mcp")
            async def _sse_preflight_options():
                return PlainTextResponse("OK", status_code=200)

    # 启动 MCP 服务
    mcp.run(transport=args.transport, host=args.host, port=args.port)
