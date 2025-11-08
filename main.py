#!/usr/bin/env python3
# coding: utf-8
"""
MCP Server (FastMCP) — 提供：
- 工具 simulate_grid：网格交易模拟（含费率 + 固定费用叠加）
- 工具 get_current_time：获取当前 UTC 时间（ISO8601）
启动方式支持命令行参数：--transport、--host、--port
"""

import argparse
import sys
import baostock as bs
from typing import List, Dict, Any
from fastmcp import FastMCP
from datetime import datetime

# ------------------------
# 代码规范化：转为 sz./sh.
# ------------------------
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

# ------------------------
# 获取 5 分钟 K 线数据
# ------------------------
def get_5min_kline(
    stock_code: str,
    start_date: str = "2024-01-01",
    end_date: str = "2025-12-31"
) -> List[Dict[str, Any]]:
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
    return data

# ------------------------
# 网格交易模拟
# ------------------------
def sim_grid(
    initial_capital: float,
    init_shares: float,
    grid: float,
    shares_per_trade: int,
    kdata: List[Dict[str, Any]],
    fee_rate: float = 0.0003,     # 默认 0.03%
    fixed_fee: float = 5.0        # 默认每笔固定 5 元
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

        # 卖出
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

        # 买入
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
        "initial_baseline": float(kdata[0]['open']),
        "final_price": float(final_price)
    }

# ------------------------
# MCP Server & 工具注册
# ------------------------
mcp = FastMCP("GridTradeSim")

@mcp.tool()
def simulate_grid(
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
    """
    网格交易模拟工具：
      - code：股票代码（6位或带 sz./sh. 前缀）
      - start, end：日期范围 YYYY-MM-DD
      - capital：初始资金
      - base_ratio：底仓比例
      - grid：网格距离（元）
      - trade_size：每次交易股数
      - fee_rate：比例费率（如 0.0003 表示 0.03%）
      - fixed_fee：每笔固定费用（元）
    返回 JSON 包含 input 与 result。
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
            "first_open": round(first_open,4),
            "init_shares": init_shares,
            "init_cash": round(init_cash,4),
        },
        "result": result
    }

@mcp.tool()
def get_current_time() -> str:
    """
    获取当前 UTC 时间（ISO8601 格式）。
    """
    return datetime.utcnow().isoformat() + "Z"

# ------------------------
# 启动入口：命令行指定 transport / host / port
# ------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GridTradeSim MCP Server")
    parser.add_argument(
        "--transport", choices=["stdio", "http", "sse"], default="sse",
        help="传输方式（stdio | http | sse），默认 sse"
    )
    parser.add_argument(
        "--host", default="0.0.0.0",
        help="监听主机（http/sse 有效），默认 0.0.0.0"
    )
    parser.add_argument(
        "--port", type=int, default=9898,
        help="监听端口（http/sse 有效），默认 9898"
    )
    args = parser.parse_args()

    mcp.run(transport=args.transport, host=args.host, port=args.port)
