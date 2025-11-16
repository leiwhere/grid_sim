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
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.routing import Mount

_kline_cache: Dict[str, List[Dict[str, Any]]] = {}

def normalize_code_for_baostock(code: str) -> str:
    """标准化股票代码格式为 baostock 要求的格式"""
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

def get_5min_kline(
    stock_code: str,
    start_date: str = "2024-01-01",
    end_date: str = "2025-12-31"
) -> List[Dict[str, Any]]:
    """从 baostock 获取 5 分钟 K 线数据"""
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
    """执行网格交易模拟核心逻辑"""
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

        # 价格触及上轨，卖出
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
                    "fee": round(fee, 4),
                    "cash_after": round(cash, 4),
                    "shares_after": shares
                })
                baseline = up_grid

        # 价格触及下轨，买入
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
                    "fee": round(fee, 4),
                    "cash_after": round(cash, 4),
                    "shares_after": shares
                })
                baseline = down_grid

    final_price = kdata[-1]['close']
    final_capital = cash + shares * final_price
    first_open = kdata[0]['open']
    initial_cost = init_shares * first_open
    total_profit = final_capital - initial_capital - initial_cost

    return {
        "final_capital": round(final_capital, 4),
        "total_profit": round(total_profit, 4),
        "total_fee": round(total_fee, 4),
        "trades": int(trades),
        "remaining_shares": float(shares),
        "cash": round(cash, 4),
        "trade_records": trade_records,
        "initial_cost_shares": float(init_shares),
        "initial_baseline": float(first_open),
        "final_price": float(final_price)
    }

# 创建 MCP 实例
mcp = FastMCP("GridTradeSim")

@mcp.tool()
def simulate_grid_tool(
    code: Annotated[str, "股票代码，如 '600000' 或 'sh.600000'"],
    start: Annotated[str, "起始日期，格式 YYYY-MM-DD"] = "2024-01-01",
    end: Annotated[str, "结束日期，格式 YYYY-MM-DD"] = "2025-12-31",
    capital: Annotated[float, "初始资金总额（元）"] = 100000.0,
    base_ratio: Annotated[float, "初次买入占资金比例（0-1之间）"] = 0.30,
    grid: Annotated[float, "网格价格间隔（元）"] = 0.2,
    trade_size: Annotated[int, "每次交易份额（股数）"] = 2000,
    fee_rate: Annotated[float, "交易费率（如 0.0003 表示万分之三）"] = 0.0003,
    fixed_fee: Annotated[float, "固定交易费用（元）"] = 5.0
) -> Dict[str, Any]:
    """
    执行5分钟K线网格交易模拟
    
    基于指定股票代码和时间范围，使用网格交易策略进行回测模拟。
    返回详细的交易记录和收益统计。
    """
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
            "first_open": round(first_open, 4),
            "init_shares": init_shares,
            "init_cash": round(init_cash, 4),
        },
        "result": result
    }

@mcp.tool()
def get_current_time_tool() -> str:
    """获取当前 UTC 时间（ISO8601 格式）"""
    return datetime.utcnow().isoformat() + "Z"

@mcp.tool()
def cache_status_full_tool() -> Dict[str, Any]:
    """查询缓存状态、进程内存和系统内存统计信息"""
    status = {key: len(_kline_cache[key]) for key in _kline_cache}
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
def clear_cache_tool(
    key: Annotated[Optional[str], "指定要清除的缓存键，留空则清除全部"] = None
) -> str:
    """清理 K 线数据缓存"""
    if key:
        if key in _kline_cache:
            del _kline_cache[key]
            return f"已清除缓存项：{key}"
        else:
            return f"未找到缓存项：{key}"
    else:
        count = len(_kline_cache)
        _kline_cache.clear()
        return f"已清除全部缓存（共 {count} 项）"

def create_dual_mode_app():
    """创建同时支持 HTTP 和 SSE 的组合应用"""
    middleware = [
        Middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["*"],
            allow_headers=["*"],
            expose_headers=["*"],
            allow_credentials=True
        )
    ]
    
    # 创建独立的 HTTP 和 SSE 应用
    # HTTP 应用处理 /mcp 路径
    http_sub_app = mcp.http_app(path="/", middleware=middleware)
    
    # SSE 应用：需要提供 message_path 和 sse_path 参数
    # 因为会被挂载到 /sse 路径，所以 sse_path 使用 "/"，message_path 使用 "/message"
    sse_sub_app = create_sse_app(
        mcp, 
        message_path="/message", 
        sse_path="/", 
        middleware=middleware
    )
    
    # 创建组合 lifespan
    @asynccontextmanager
    async def combined_lifespan(app):
        async with http_sub_app.lifespan(http_sub_app):
            async with sse_sub_app.lifespan(sse_sub_app):
                yield
    
    # 创建组合应用
    combined_app = Starlette(
        routes=[
            Mount("/mcp", http_sub_app, name="http"),
            Mount("/sse", sse_sub_app, name="sse"),
        ],
        middleware=middleware,
        lifespan=combined_lifespan
    )
    
    return combined_app

def main():
    parser = argparse.ArgumentParser(
        description="GridTradeSim MCP Server - 同时支持 HTTP 和 SSE 双模式",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--host", default="0.0.0.0",
        help="监听主机 (默认: 0.0.0.0)"
    )
    parser.add_argument(
        "--port", type=int, default=9898,
        help="监听端口 (默认: 9898)"
    )
    parser.add_argument(
        "--stdio", action="store_true",
        help="以 stdio 模式运行（不启动 HTTP/SSE 服务）"
    )

    args = parser.parse_args()

    if args.stdio:
        # stdio 模式
        mcp.run()
    else:
        # 双模式 HTTP + SSE
        app = create_dual_mode_app()
        import uvicorn
        uvicorn.run(
            app,
            host=args.host,
            port=args.port,
            log_level="info",
            access_log=True
        )

if __name__ == "__main__":
    main()
