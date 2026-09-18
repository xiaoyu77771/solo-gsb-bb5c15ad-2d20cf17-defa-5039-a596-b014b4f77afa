"""IPv4 路由解析核心逻辑（纯标准库实现，无 Web 依赖）。

给定有序静态路由列表（``cidr`` / ``next_hop`` / ``metric``）与一组唯一的
目标 IPv4 地址，对每个目标按以下顺序解析：

1. **最长前缀匹配（LPM）**：在所有命中目标的路由中只保留前缀长度最大者；
2. **最低 metric 优先**：在 LPM 集合中再保留 metric 最小的全部路由；
3. 这些路由构成等价多路径（ECMP）结果，其 ``next_hop`` 按 **routes 输入
   顺序去重**后返回；未命中任何路由的目标标记为 ``unreachable``。

全程只做 32 位整数比较，**不逐地址展开网段**，可安全处理 ``0.0.0.0/0``。
"""
from __future__ import annotations

from collections import defaultdict
from ipaddress import IPv4Address, ip_address
from typing import Sequence

from .analyzer import parse_cidr_v4

#: 单次请求允许的最大路由条数
ROUTES_MAX = 1000

#: 单次请求允许的目标数区间
TARGETS_MIN = 1
TARGETS_MAX = 500

#: metric 取值范围
METRIC_MIN = 0
METRIC_MAX = 65535

REACHABLE = "reachable"
UNREACHABLE = "unreachable"


class InvalidIPv4AddressError(ValueError):
    """非法 IPv4 地址（格式错误、带前缀、IPv6 等）。"""


class DuplicateTargetsError(ValueError):
    """目标地址重复。

    ``duplicates`` 给出每个重复目标（按首次出现排序），形如
    ``{"target": "8.8.8.8", "indexes": [0, 3]}``。
    """

    def __init__(self, duplicates: list[dict]) -> None:
        self.duplicates = duplicates
        super().__init__(f"{len(duplicates)} 个重复目标地址")


def parse_address_v4(text: str) -> tuple[int, str]:
    """严格解析单个 IPv4 地址（**禁止** CIDR 前缀与 IPv6）。

    返回 ``(地址整数, 规范化字符串)``；与 CIDR 校验同样严格：
    不接受主机位/前缀写法（``8.8.8.8/32``）、前导零（``010.0.0.1``）
    及 IPv6（``::1``）。
    """
    if not isinstance(text, str):
        raise InvalidIPv4AddressError("IPv4 地址必须是字符串")
    raw = text.strip()
    if not raw:
        raise InvalidIPv4AddressError("IPv4 地址不能为空")
    if "/" in raw:
        raise InvalidIPv4AddressError(f"非法 IPv4 地址（不允许前缀写法）: {text!r}")
    try:
        addr = ip_address(raw)
    except ValueError as exc:
        raise InvalidIPv4AddressError(f"非法 IPv4 地址: {text!r}（{exc}）") from exc
    if not isinstance(addr, IPv4Address):
        raise InvalidIPv4AddressError(f"仅支持 IPv4，不支持 IPv6: {text!r}")
    return int(addr), str(addr)


class InvalidMetricError(ValueError):
    """metric 不是 0～65535 的整数。"""


def normalize_metric(metric: object) -> int:
    """校验 metric：非 bool 整数且在 0～65535 之间。"""
    if isinstance(metric, bool) or not isinstance(metric, int):
        raise InvalidMetricError(f"metric 必须是 0～{METRIC_MAX} 的整数: {metric!r}")
    if not METRIC_MIN <= metric <= METRIC_MAX:
        raise InvalidMetricError(
            f"metric 必须在 {METRIC_MIN}～{METRIC_MAX} 之间，实际 {metric}"
        )
    return metric


def resolve_routes(
    routes: Sequence[tuple[str, str, int]],
    targets: Sequence[str],
) -> dict:
    """解析一组目标地址的最长前缀、最低 metric 等价下一跳。

    ``routes`` 每项为 ``(cidr, next_hop, metric)``，允许为空（此时所有目标
    均不可达）；``targets`` 每项为单个严格 IPv4 地址，**必须唯一**。

    返回可直接 JSON 序列化的字典：``results`` 严格保持 targets 输入顺序；
    命中项给出 ``status="reachable"``、共同命中的规范化 ``matched_cidr``、
    ``metric`` 与按 routes 输入顺序去重的 ``next_hops``；未命中项为
    ``status="unreachable"``、其余字段为 ``None`` / ``[]``。
    ``summary.next_hop_counts`` 汇总每个 next_hop 命中的**目标数**（同一
    ECMP 结果的各下一跳分别计一次），按 next_hop 在 routes 中的首次出现
    位置排序。目标重复时抛出 :class:`DuplicateTargetsError`，非法 CIDR /
    地址 / metric 抛出对应 ValueError 子类。
    """
    if not 0 <= len(routes) <= ROUTES_MAX:
        raise ValueError(f"路由数必须在 0～{ROUTES_MAX} 之间，实际 {len(routes)}")
    if not TARGETS_MIN <= len(targets) <= TARGETS_MAX:
        raise ValueError(
            f"目标数必须在 {TARGETS_MIN}～{TARGETS_MAX} 之间，实际 {len(targets)}"
        )

    parsed_routes: list[dict] = []
    # 每个 next_hop 在 routes 中的最早输入序号，用于汇总排序
    hop_first_index: dict[int, int] = {}
    for index, (raw_cidr, raw_next_hop, raw_metric) in enumerate(routes):
        start, end, cidr = parse_cidr_v4(raw_cidr)
        hop_int, next_hop = parse_address_v4(raw_next_hop)
        metric = normalize_metric(raw_metric)
        parsed_routes.append(
            {
                "start": start,
                "end": end,
                "cidr": cidr,
                "prefixlen": int(cidr.rsplit("/", 1)[1]),
                "metric": metric,
                "hop_int": hop_int,
                "next_hop": next_hop,
            }
        )
        hop_first_index.setdefault(hop_int, index)

    parsed_targets: list[tuple[int, str]] = []
    positions: dict[int, list[int]] = defaultdict(list)
    first_seen: list[int] = []
    for index, raw_target in enumerate(targets):
        value, text = parse_address_v4(raw_target)
        parsed_targets.append((value, text))
        if value not in positions:
            first_seen.append(value)
        positions[value].append(index)

    duplicates = [
        {"target": parsed_targets[positions[value][0]][1],
         "indexes": positions[value]}
        for value in first_seen
        if len(positions[value]) > 1
    ]
    if duplicates:
        raise DuplicateTargetsError(duplicates)

    results: list[dict] = []
    # hop_int -> [规范化 next_hop, 命中目标数]；同一 ECMP 结果的去重下一跳
    # 只出现一次，但对每个命中目标各计一次
    hop_stats: dict[int, list] = {}
    unreachable_count = 0

    for value, text in parsed_targets:
        best_prefix = -1
        best_metric: int = 1 << 30  # 哨兵：尚无命中时大于任何合法 metric
        chosen: list[dict] = []
        # 线性扫描，仅做整数包含判断；同前缀长度且包含同一地址的 CIDR
        # 必然是同一网段，故 LPM 命中路由的 cidr 唯一。
        for route in parsed_routes:
            if not route["start"] <= value <= route["end"]:
                continue
            if route["prefixlen"] > best_prefix:
                best_prefix = route["prefixlen"]
                best_metric = route["metric"]
                chosen = [route]
            elif route["prefixlen"] == best_prefix and route["metric"] <= best_metric:
                if route["metric"] < best_metric:
                    best_metric = route["metric"]
                    chosen = [route]
                else:
                    chosen.append(route)

        if not chosen:
            unreachable_count += 1
            results.append(
                {
                    "target": text,
                    "status": UNREACHABLE,
                    "matched_cidr": None,
                    "metric": None,
                    "next_hops": [],
                }
            )
            continue

        # chosen 已按 routes 输入顺序排列；next_hop 去重并分别计数一次
        next_hops: list[str] = []
        counted_hops: set[int] = set()
        for route in chosen:
            hop_int = route["hop_int"]
            if hop_int not in counted_hops:
                counted_hops.add(hop_int)
                next_hops.append(route["next_hop"])
                stat = hop_stats.get(hop_int)
                if stat is None:
                    stat = [route["next_hop"], 0]
                    hop_stats[hop_int] = stat
                stat[1] += 1

        results.append(
            {
                "target": text,
                "status": REACHABLE,
                "matched_cidr": chosen[0]["cidr"],
                "metric": chosen[0]["metric"],
                "next_hops": next_hops,
            }
        )

    next_hop_counts = [
        {"next_hop": hop_stats[hop_int][0], "target_count": hop_stats[hop_int][1]}
        for hop_int in sorted(hop_stats, key=lambda hop: hop_first_index[hop])
    ]

    return {
        "route_count": len(parsed_routes),
        "target_count": len(parsed_targets),
        "results": results,
        "summary": {
            "reachable_count": len(results) - unreachable_count,
            "unreachable_count": unreachable_count,
            "next_hop_counts": next_hop_counts,
        },
    }
