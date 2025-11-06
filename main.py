import argparse
from fastmcp import FastMCP
import baostock as bs
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

mcp = FastMCP(
    name="BaoStock KData & Dividend Server",
    instructions="提供 A 股 K 线数据获取、统计分析、缓存管理、网格交易模拟（含分红）与除权除息数据查询工具"
)

# 全局缓存：K 线数据
cache = {}
# 全局缓存：股息／除权除息数据
divid_cache = {}

@mcp.tool()
def get_kdata(
    code: str,
    start_date: str = "",
    end_date:   str = "",
    frequency:  str = "5",
    adjustflag: str = "3"
) -> dict:
    """
    获取指定股票的 K 线数据，并返回统计指标。默认周期为 5 分钟，默认起始两年前，默认到今天。
    参数：
        code        : 股票代码，如 "sh.600000" 或 "sz.000001"
        start_date  : 起始日期 "YYYY-MM-DD"，默认为空（表示两年前）
        end_date    : 结束日期 "YYYY-MM-DD"，默认为空（表示今天）
        frequency   : K线周期，默认 "5"（5 分钟）；也可 "15","30","60","d"（日线）等
        adjustflag  : 复权类型，默认 "3"（不复权）；"1" 后复权、"2" 前复权
    返回：
        { "stats": { … 多种统计指标 … } }
    缓存机制：
        将原始数据（DataFrame）、统计结果和调用时间存入 global `cache`，键为 code。
    """
    if end_date == "":
        end_date = datetime.now().strftime("%Y-%m-%d")
    if start_date == "":
        start_date = (datetime.now() - timedelta(days=365*2)).strftime("%Y-%m-%d")

    lg = bs.login()
    if lg.error_code != '0':
        return {"error": f"登录失败: {lg.error_msg}"}

    fields = "date,time,code,open,high,low,close,volume,amount,adjustflag"
    rs = bs.query_history_k_data_plus(
        code=code,
        fields=fields,
        start_date=start_date,
        end_date=end_date,
        frequency=frequency,
        adjustflag=adjustflag
    )

    data_list = []
    while (rs.error_code == '0') & rs.next():
        data_list.append(rs.get_row_data())
    bs.logout()

    if not data_list:
        return {"error": "未获取到数据"}

    df = pd.DataFrame(data_list, columns=rs.fields)
    df[['open','high','low','close','volume','amount']] = \
        df[['open','high','low','close','volume','amount']].astype(float)

    start_price = float(df.iloc[0]['close'])
    end_price   = float(df.iloc[-1]['close'])
    closes = df['close']
    mean_price = float(np.mean(closes))
    std0 = float(np.std(closes, ddof=0))
    std1 = float(np.std(closes, ddof=1))
    ratio0 = std0 / mean_price if mean_price != 0 else None
    ratio1 = std1 / mean_price if mean_price != 0 else None
    try:
        freq_minutes = float(frequency) if frequency not in ("d","w","m") else 5.0
    except:
        freq_minutes = 5.0
    periods_per_day = (60.0 * 6.5) / freq_minutes
    annualised = ratio0 * np.sqrt(periods_per_day * 252) if ratio0 is not None else None

    stats = {
        "code":        code,
        "start_date":  df.iloc[0]['date'],
        "end_date":    df.iloc[-1]['date'],
        "start_price": start_price,
        "end_price":   end_price,
        "max_price":   float(df['high'].max()),
        "min_price":   float(df['low'].min()),
        "avg_price":   mean_price,
        "count":       int(len(df)),
        "change_percent": (end_price - start_price) / start_price * 100 if start_price != 0 else None,
        "volatility_std0":      std0,
        "volatility_std1":      std1,
        "volatility_ratio0":    ratio0,
        "volatility_ratio1":    ratio1,
        "volatility_annualised": annualised,
    }

    cache[code] = {
        "raw":       df,
        "stats":     stats,
        "timestamp": datetime.now().isoformat()
    }

    return {"stats": stats}

@mcp.tool()
def list_cache() -> dict:
    """
    列出当前缓存cache 中所有股票 K 线数据条目。
    在其他操作前，检查cache。
    返回：
        {
          "codes": [ … 股票代码列表 … ],
          "details": {
             "<code>": {
                "timestamp": "...",
                "count": <条目数>,
                "start_date": "...",
                "end_date": "..."}
             …}
        }
    """
    details = {}
    for code, entry in cache.items():
        stats = entry.get("stats", {})
        details[code] = {
            "timestamp":   entry.get("timestamp"),
            "count":       stats.get("count"),
            "start_date":   stats.get("start_date"),
            "end_date":     stats.get("end_date"),
        }
    return {"codes": list(cache.keys()), "details": details}

@mcp.tool()
def delete_cache(code: str) -> dict:
    """
    删除缓存cache中某个股票 K 线数据条目。
    参数：
        code : 股票代码，要删除的缓存条目 key。
    返回：
        {
          "deleted": <code>,
          "success": <true|false>,
          "error": <若失败则说明>
        }
    """
    if code in cache:
        del cache[code]
        return {"deleted": code, "success": True}
    else:
        return {"deleted": code, "success": False, "error": "code not found in cache"}

@mcp.tool()
def simulate_grid(
    code:           str,
    grid_size:      float,
    trade_size:     float,
    baseline:       float | None = None,
    total_capital:  float | None = None,
    base_position:  float = 0.0,
    fee_rate:       float = 0.0003,
    fixed_fee:      float = 5.0,
    grid_size_up:   float | None = None,
    grid_size_down: float | None = None
) -> dict:
    """
    模拟网格交易（含分红／送股处理，含交易费用）。
    使用缓存cache中已有的 K 线数据及除权除息数据。
    新增功能：向上突破与向下突破可分别指定网格间距 grid_size_up / grid_size_down。
    参数：
      code             : 股票代码
      grid_size        : 网格间距（单位同价格），若未指定 grid_size_up/down，则向上、向下都用此值
      trade_size       : 每次交易数量（股数或份额）
      baseline         : 基线价（起始价），若空则用缓存数据首条收盘价
      total_capital    : 总资金（限制买入），若空则不限制
      base_position    : 初始持仓量，默认为 0.0
      fee_rate         : 每次交易按成交额比例收取费用，默认 0.03%
      fixed_fee        : 每次交易额外固定手续费，默认 5 元
      grid_size_up     : 向上突破时使用的网格宽度；若为空，则用 grid_size
      grid_size_down   : 向下突破／下穿时使用的网格宽度；若为空，则用 grid_size
    返回：
      {
        "code": code,
        "baseline": baseline,
        "grid_size_up": grid_size_up_used,
        "grid_size_down": grid_size_down_used,
        "trade_size": trade_size,
        "base_position": base_position,
        "total_capital": total_capital,
        "fee_rate": fee_rate,
        "fixed_fee": fixed_fee,
        "trades": [ … 成交记录 … ],
        "stats": { … 模拟结果 … }
      }
    """
    # 校正网格宽度
    if grid_size_up is None:
        grid_size_up = grid_size
    if grid_size_down is None:
        grid_size_down = grid_size

    if code not in cache:
        return {"error": f"代码 {code} 未在缓存中，请先调用 get_kdata 并缓存数据。"}
    entry = cache[code]
    df = entry["raw"].copy().reset_index(drop=True)
    if "close" not in df.columns:
        return {"error": "缓存数据中缺少 close 列。"}

    # 分红数据（如果有）
    div_records = divid_cache.get(code, {}).get("records", [])
    div_by_date = { rec.get("dividOperateDate"): rec for rec in div_records if rec.get("dividOperateDate") }

    if baseline is None:
        baseline = float(df.iloc[0]["close"])
    used_capital = 0.0
    realised_profit = 0.0
    position = base_position
    avg_cost = baseline if base_position > 0 else None
    total_dividends = 0.0
    additional_shares = 0.0

    trades = []
    last_price = None

    for idx, row in df.iterrows():
        price = float(row["close"])
        date_str = row["date"]

        # 分红／送股
        if date_str in div_by_date:
            rec = div_by_date[date_str]
            cash_ps = float(rec.get("dividCashPsBeforeTax") or 0.0)
            stocks_ps = float(rec.get("dividStocksPs") or 0.0)
            if position > 0 and cash_ps > 0:
                dividend_cash = position * cash_ps
                total_dividends += dividend_cash
                realised_profit += dividend_cash
                trades.append({
                    "time": date_str,
                    "type": "dividend_cash",
                    "price": None,
                    "amount": position,
                    "dividend": dividend_cash
                })
            if position > 0 and stocks_ps > 0:
                bonus_shares = position * stocks_ps
                additional_shares += bonus_shares
                position += bonus_shares
                if avg_cost is not None:
                    avg_cost = (avg_cost * (position - bonus_shares) + price * bonus_shares) / position
                trades.append({
                    "time": date_str,
                    "type": "dividend_stock",
                    "price": None,
                    "amount": bonus_shares,
                    "bonus_shares": bonus_shares
                })

        if last_price is None:
            last_price = price
            continue

        # 价格下穿某网格价位 → 买入（使用 grid_size_down）
        grid_index_last_down = int(np.floor((last_price - baseline) / grid_size_down))
        grid_index_now_down  = int(np.floor((price - baseline) / grid_size_down))
        if grid_index_now_down < grid_index_last_down:
            target_buy_price = baseline + grid_index_now_down * grid_size_down
            amount = trade_size
            cost = amount * price
            if (total_capital is None) or (used_capital + cost <= total_capital):
                fee = cost * fee_rate + fixed_fee
                used_capital += cost + fee
                if position > 0 and avg_cost is not None:
                    avg_cost = (avg_cost * position + price * amount) / (position + amount)
                else:
                    avg_cost = price
                position += amount
                trades.append({
                    "time": date_str,
                    "type": "buy",
                    "price": price,
                    "amount": amount,
                    "cost": cost,
                    "fee": fee,
                    "avg_cost": avg_cost
                })

        # 价格上穿某网格价位 → 卖出（使用 grid_size_up）
        grid_index_last_up = int(np.floor((last_price - baseline) / grid_size_up))
        grid_index_now_up  = int(np.floor((price - baseline) / grid_size_up))
        if grid_index_now_up > grid_index_last_up:
            amount = min(trade_size, position)
            if amount > 0:
                revenue = amount * price
                profit = (price - avg_cost) * amount
                fee = revenue * fee_rate + fixed_fee
                realised_profit += profit - fee
                position -= amount
                trades.append({
                    "time": date_str,
                    "type": "sell",
                    "price": price,
                    "amount": amount,
                    "revenue": revenue,
                    "profit": profit,
                    "fee": fee,
                    "remaining_position": position
                })
                if position == 0:
                    avg_cost = None

        last_price = price

    ending_price = float(df.iloc[-1]["close"])
    unrealised_profit = 0.0
    if position > 0 and avg_cost is not None:
        unrealised_profit = (ending_price - avg_cost) * position
    bonus_value = additional_shares * ending_price
    total_profit = realised_profit + unrealised_profit + bonus_value
    capital_return_rate = (total_profit / used_capital) * 100 if used_capital > 0 else None

    stats = {
        "ending_position":        position,
        "used_capital":           used_capital,
        "realised_profit":        realised_profit,
        "total_dividends":        total_dividends,
        "additional_shares":      additional_shares,
        "bonus_value":            bonus_value,
        "unrealised_profit":      unrealised_profit,
        "total_profit":           total_profit,
        "capital_return_rate":    capital_return_rate
    }

    return {
        "code":           code,
        "baseline":       baseline,
        "grid_size_up":    grid_size_up,
        "grid_size_down":  grid_size_down,
        "trade_size":      trade_size,
        "base_position":   base_position,
        "total_capital":   total_capital,
        "fee_rate":        fee_rate,
        "fixed_fee":       fixed_fee,
        "trades":          trades,
        "stats":           stats
    }

@mcp.tool()
def query_dividends(
    code:       str,
    start_year: int,
    end_year:   int,
    yearType:   str = "report"
) -> dict:
    """
    查询股票的除权除息／股息分配信息。
    参数：
      code       : 股票代码，如 "sh.600000"
      start_year : 起始年份（含）
      end_year   : 结束年份（含）
      yearType   : 年份类别，字符串，默认 "report"
    返回：
      {
        "code": code,
        "years": [ … ],
        "data": [ … 每条记录 … ]
      }
    缓存机制：
      将结果保存至 global `divid_cache`。
    """
    lg = bs.login()
    if lg.error_code != '0':
        return {"error": f"登录失败: {lg.error_msg}"}

    all_rows = []
    for year in range(start_year, end_year + 1):
        rs = bs.query_dividend_data(code=code, year=str(year), yearType=yearType)
        if rs.error_code != '0':
            continue
        while (rs.error_code == '0') & rs.next():
            all_rows.append(rs.get_row_data())
    bs.logout()

    if not all_rows:
        return {"error": f"未获取到 {code} 在 {start_year}-{end_year} 年份的除权除息数据"}

    df = pd.DataFrame(all_rows, columns=rs.fields)
    records = df.to_dict(orient="records")

    divid_cache[code] = {
        "years":     list(range(start_year, end_year + 1)),
        "records":   records,
        "timestamp": datetime.now().isoformat()
    }

    return {"code": code, "years": list(range(start_year, end_year + 1)), "data": records}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="BaoStock KData & Dividend Server - FastMCP Wrapper"
    )
    parser.add_argument("--host", type=str, default="0.0.0.0", help="监听主机地址，默认 0.0.0.0")
    parser.add_argument("--port", type=int, default=9898, help="监听端口，默认 9898")
    parser.add_argument("--protocol", type=str, choices=["http", "sse", "stdio"], default="sse",
                        help="通信协议类型，可选 http / sse / stdio，默认 sse")
    args = parser.parse_args()

    print(f"[INFO] 启动 FastMCP 服务器：protocol={args.protocol}, host={args.host}, port={args.port}")
    mcp.run(transport=args.protocol, host=args.host, port=args.port)
