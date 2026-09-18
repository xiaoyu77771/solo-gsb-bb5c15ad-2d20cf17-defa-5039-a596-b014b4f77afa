"""IPv4 路由解析核心逻辑（纯标准库实现，无 Web 依赖）。

给定有序静态路由列表（``cidr`` / ``next_hop`` / ``metric``）与一组唯一的
目标 IPv4 地址，对每个目标按以下顺序解析：

1. **最长前缀匹配（LPM）**：在所有命中目标的路由中只保留前缀长度最大者；
2. **最低 metric 优先**：在 LPM 集合中再保留 metric 最小的全部路由；
3. 这些路由构成等价多路径（ECMP）结果，其 ``next_hop`` 按 **routes 输入
   顺序去重**后返回；未命中任何路由的目标标记为 ``unreachable``。

:func:`resolve_failover` 在此之上模拟下一跳故障：先给出正常解析结果
（before），再从路由表中**整体移除**故障下一跳的**全部路由**并重新执行
LPM + 最低 metric + ECMP 选择（after）——不是只从原 ECMP 列表删值，因此
允许回退到较高 metric 的同前缀路由或更短前缀的路由。逐目标按
targets 输入顺序给出 before/after 与影响分类（unchanged / ecmp_reduced /
rerouted / unreachable）。

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

#: 单次请求允许的故障下一跳数区间（至少 1 个，上限与路由条数一致）
FAILED_NEXT_HOPS_MIN = 1
FAILED_NEXT_HOPS_MAX = ROUTES_MAX

#: metric 取值范围
METRIC_MIN = 0
METRIC_MAX = 65535

REACHABLE = "reachable"
UNREACHABLE = "unreachable"

#: 故障影响分类
IMPACT_UNCHANGED = "unchanged"
IMPACT_ECMP_REDUCED = "ecmp_reduced"
IMPACT_REROUTED = "rerouted"
IMPACT_UNREACHABLE = "unreachable"


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


class DuplicateFailedNextHopsError(ValueError):
    """故障下一跳地址重复。

    ``duplicates`` 给出每个重复地址（按首次出现排序），形如
    ``{"next_hop": "10.0.0.1", "indexes": [0, 2]}``。
    """

    def __init__(self, duplicates: list[dict]) -> None:
        self.duplicates = duplicates
        super().__init__(f"{len(duplicates)} 个重复的故障下一跳地址")


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


def _parse_route_entries(
    routes: Sequence[tuple[str, str, int]],
) -> tuple[list[dict], dict[int, int]]:
    """把 ``(cidr, next_hop, metric)`` 列表解析为整数化路由字典列表。

    返回 ``(parsed_routes, hop_first_index)``：后者记录每个 next_hop 在
    routes 中的最早输入序号，用于命中汇总排序。
    """
    parsed_routes: list[dict] = []
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
    return parsed_routes, hop_first_index


def _parse_address_list(
    values: Sequence[str],
) -> tuple[list[tuple[int, str]], dict[int, list[int]], list[int]]:
    """解析地址列表并记录每个地址的全部出现位置。

    返回 ``(parsed, positions, first_seen)``：``parsed`` 保持输入顺序，
    ``positions`` 为地址整数 -> 全部索引，``first_seen`` 为地址首次出现
    的顺序。
    """
    parsed: list[tuple[int, str]] = []
    positions: dict[int, list[int]] = defaultdict(list)
    first_seen: list[int] = []
    for index, raw in enumerate(values):
        value, text = parse_address_v4(raw)
        parsed.append((value, text))
        if value not in positions:
            first_seen.append(value)
        positions[value].append(index)
    return parsed, positions, first_seen


def _duplicate_entries(
    parsed: Sequence[tuple[int, str]],
    positions: dict[int, list[int]],
    first_seen: Sequence[int],
    key: str,
) -> list[dict]:
    """按首次出现顺序构造重复地址明细，每项形如 ``{key: 地址, indexes: [...]}``。"""
    return [
        {key: parsed[positions[value][0]][1], "indexes": positions[value]}
        for value in first_seen
        if len(positions[value]) > 1
    ]


def _select_routes(parsed_routes: Sequence[dict], target_value: int) -> list[dict]:
    """对单个目标执行 LPM + 最低 metric 选择，返回命中的全部并列路由。

    返回列表保持 routes 输入顺序；未命中任何路由时为空。线性扫描，仅做
    整数包含判断；同前缀长度且包含同一地址的 CIDR 必然是同一网段，故
    LPM 命中路由的 cidr 唯一。
    """
    best_prefix = -1
    best_metric: int = 1 << 30  # 哨兵：尚无命中时大于任何合法 metric
    chosen: list[dict] = []
    for route in parsed_routes:
        if not route["start"] <= target_value <= route["end"]:
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
    return chosen


def _build_item(
    target_text: str,
    chosen: Sequence[dict],
) -> tuple[dict, list[tuple[int, str]]]:
    """把命中路由列表组装为单个目标的解析结果项。

    返回 ``(item, chosen_hops)``：``chosen_hops`` 为去重后的命中下一跳
    ``(地址整数, 规范化地址)`` 列表（保持 routes 输入顺序），供命中计数
    复用。``chosen`` 为空时产出不可达项。
    """
    if not chosen:
        return (
            {
                "target": target_text,
                "status": UNREACHABLE,
                "matched_cidr": None,
                "metric": None,
                "next_hops": [],
            },
            [],
        )

    next_hops: list[str] = []
    chosen_hops: list[tuple[int, str]] = []
    seen: set[int] = set()
    for route in chosen:
        hop_int = route["hop_int"]
        if hop_int not in seen:
            seen.add(hop_int)
            chosen_hops.append((hop_int, route["next_hop"]))
            next_hops.append(route["next_hop"])
    return (
        {
            "target": target_text,
            "status": REACHABLE,
            "matched_cidr": chosen[0]["cidr"],
            "metric": chosen[0]["metric"],
            "next_hops": next_hops,
        },
        chosen_hops,
    )


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

    parsed_routes, hop_first_index = _parse_route_entries(routes)

    parsed_targets, positions, first_seen = _parse_address_list(targets)
    duplicates = _duplicate_entries(
        parsed_targets, positions, first_seen, "target"
    )
    if duplicates:
        raise DuplicateTargetsError(duplicates)

    results: list[dict] = []
    # hop_int -> [规范化 next_hop, 命中目标数]；同一 ECMP 结果的去重下一跳
    # 只出现一次，但对每个命中目标各计一次
    hop_stats: dict[int, list] = {}
    unreachable_count = 0

    for value, text in parsed_targets:
        item, chosen_hops = _build_item(text, _select_routes(parsed_routes, value))
        results.append(item)
        if not chosen_hops:
            unreachable_count += 1
            continue
        for hop_int, next_hop in chosen_hops:
            stat = hop_stats.get(hop_int)
            if stat is None:
                stat = [next_hop, 0]
                hop_stats[hop_int] = stat
            stat[1] += 1

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


def _classify_impact(before: dict, after: dict) -> str:
    """对比同一目标故障前后的解析结果，判定影响分类。

    - ``unchanged``：故障前后结果完全一致（目标原本不经过故障下一跳；
      原本不可达、故障后仍不可达也属此类）；
    - ``ecmp_reduced``：仍命中同一路由（网段与 metric 不变），仅 ECMP
      下一跳集合为原来的严格非空真子集；
    - ``unreachable``：故障前可达、故障后不可达；
    - ``rerouted``：其余一切变化（回退到较高 metric 的同前缀路由，或回退
      到更短前缀的路由）。路由移除只会收紧可达性，不会出现不可达→可达。
    """
    if after["status"] == UNREACHABLE:
        # before 可达 -> 故障后不可达；before 本就不可达则两边相等，
        # 落到末尾的 unchanged 判断。
        if before["status"] == REACHABLE:
            return IMPACT_UNREACHABLE
    else:
        if (
            before["status"] == REACHABLE
            and before["matched_cidr"] == after["matched_cidr"]
            and before["metric"] == after["metric"]
        ):
            before_hops = set(before["next_hops"])
            after_hops = set(after["next_hops"])
            if after_hops < before_hops:
                return IMPACT_ECMP_REDUCED
    if before == after:
        return IMPACT_UNCHANGED
    return IMPACT_REROUTED


def resolve_failover(
    routes: Sequence[tuple[str, str, int]],
    targets: Sequence[str],
    failed_next_hops: Sequence[str],
) -> dict:
    """模拟下一跳故障的路由重解析。

    先按 :func:`resolve_routes` 的同一套 LPM + 最低 metric + ECMP 规则算出
    正常结果（before）；再从路由表中**整体移除** next_hop 属于
    ``failed_next_hops`` 的**全部路由**（而非只从原 ECMP 列表删值），在
    剩余路由上**重新执行**完整选择得到 after——因此允许回退到较高 metric
    的同前缀路由或更短前缀的路由，也可能变得不可达。

    ``routes`` / ``targets`` 的约束与 :func:`resolve_routes` 完全一致；
    ``failed_next_hops`` 为 1～:data:`FAILED_NEXT_HOPS_MAX` 个**唯一**严格
    IPv4 地址（不要求确实出现在 routes 中；全部未出现时所有目标均为
    unchanged）。

    返回可直接 JSON 序列化的字典：``results`` 严格保持 targets 输入顺序，
    每项给出 ``before`` / ``after``（结构同 /routes/resolve 的单个结果项）
    与 ``impact``；``summary`` 汇总四类影响的目标数。目标重复抛出
    :class:`DuplicateTargetsError`，故障下一跳重复抛出
    :class:`DuplicateFailedNextHopsError`。
    """
    if not 0 <= len(routes) <= ROUTES_MAX:
        raise ValueError(f"路由数必须在 0～{ROUTES_MAX} 之间，实际 {len(routes)}")
    if not TARGETS_MIN <= len(targets) <= TARGETS_MAX:
        raise ValueError(
            f"目标数必须在 {TARGETS_MIN}～{TARGETS_MAX} 之间，实际 {len(targets)}"
        )
    if not FAILED_NEXT_HOPS_MIN <= len(failed_next_hops) <= FAILED_NEXT_HOPS_MAX:
        raise ValueError(
            f"故障下一跳数必须在 {FAILED_NEXT_HOPS_MIN}～"
            f"{FAILED_NEXT_HOPS_MAX} 之间，实际 {len(failed_next_hops)}"
        )

    parsed_routes, _ = _parse_route_entries(routes)

    parsed_targets, target_positions, target_first_seen = _parse_address_list(targets)
    target_duplicates = _duplicate_entries(
        parsed_targets, target_positions, target_first_seen, "target"
    )
    if target_duplicates:
        raise DuplicateTargetsError(target_duplicates)

    parsed_failed, failed_positions, failed_first_seen = _parse_address_list(
        failed_next_hops
    )
    failed_duplicates = _duplicate_entries(
        parsed_failed, failed_positions, failed_first_seen, "next_hop"
    )
    if failed_duplicates:
        raise DuplicateFailedNextHopsError(failed_duplicates)

    failed_set = {value for value, _ in parsed_failed}
    surviving_routes = [
        route for route in parsed_routes if route["hop_int"] not in failed_set
    ]

    impact_counts = {
        IMPACT_UNCHANGED: 0,
        IMPACT_ECMP_REDUCED: 0,
        IMPACT_REROUTED: 0,
        IMPACT_UNREACHABLE: 0,
    }
    results: list[dict] = []
    for value, text in parsed_targets:
        before, _ = _build_item(text, _select_routes(parsed_routes, value))
        after, _ = _build_item(text, _select_routes(surviving_routes, value))
        impact = _classify_impact(before, after)
        impact_counts[impact] += 1
        results.append(
            {"target": text, "before": before, "after": after, "impact": impact}
        )

    return {
        "route_count": len(parsed_routes),
        "target_count": len(parsed_targets),
        "failed_next_hop_count": len(parsed_failed),
        "results": results,
        "summary": {
            "unchanged_count": impact_counts[IMPACT_UNCHANGED],
            "ecmp_reduced_count": impact_counts[IMPACT_ECMP_REDUCED],
            "rerouted_count": impact_counts[IMPACT_REROUTED],
            "unreachable_count": impact_counts[IMPACT_UNREACHABLE],
        },
    }
