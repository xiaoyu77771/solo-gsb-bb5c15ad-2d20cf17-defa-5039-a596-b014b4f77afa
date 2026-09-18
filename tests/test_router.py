"""IPv4 路由解析核心逻辑的单元测试。"""
import random

import pytest
from ipaddress import ip_network

from app.analyzer import parse_cidr_v4
from app.router import (
    METRIC_MAX,
    ROUTES_MAX,
    TARGETS_MAX,
    DuplicateTargetsError,
    InvalidIPv4AddressError,
    InvalidMetricError,
    normalize_metric,
    parse_address_v4,
    resolve_routes,
)


# ---------- 地址解析 ----------

def test_parse_basic_address():
    value, text = parse_address_v4("8.8.8.8")
    assert value == (8 << 24) + (8 << 16) + (8 << 8) + 8
    assert text == "8.8.8.8"


def test_parse_address_trims_outer_whitespace():
    assert parse_address_v4("  0.0.0.0 ")[1] == "0.0.0.0"
    assert parse_address_v4("255.255.255.255")[0] == (1 << 32) - 1


@pytest.mark.parametrize(
    "bad",
    [
        "",
        "   ",
        "8.8.8.8/32",   # 禁止 CIDR 写法
        "10.0.0",       # 段数不足
        "999.0.0.1",    # 段越界
        "010.0.0.1",    # 前导零
        "10.0.0. ",     # 段内空白
        "2001:db8::1",  # IPv6
        "::1",
    ],
)
def test_parse_address_rejects_invalid_and_ipv6(bad):
    with pytest.raises(InvalidIPv4AddressError):
        parse_address_v4(bad)


def test_parse_address_rejects_non_string():
    with pytest.raises(InvalidIPv4AddressError):
        parse_address_v4(0x08080808)


# ---------- metric 校验 ----------

@pytest.mark.parametrize("metric", [0, 1, 100, 65535])
def test_normalize_metric_accepts_range(metric):
    assert normalize_metric(metric) == metric


@pytest.mark.parametrize("metric", [-1, 65536, 100_000])
def test_normalize_metric_rejects_out_of_range(metric):
    with pytest.raises(InvalidMetricError):
        normalize_metric(metric)


@pytest.mark.parametrize("metric", [True, False, 1.0, "10", None])
def test_normalize_metric_rejects_non_int(metric):
    with pytest.raises(InvalidMetricError):
        normalize_metric(metric)


# ---------- 解析主流程 ----------

def test_lpm_beats_shorter_prefix_even_with_worse_metric():
    # /8 默认路由 metric 很小，但 /16 前缀更长必须胜出
    result = resolve_routes(
        [
            ("0.0.0.0/0", "1.1.1.1", 0),
            ("10.0.0.0/8", "10.0.0.1", 200),
            ("10.1.0.0/16", "10.1.0.1", 500),
        ],
        ["10.1.9.9"],
    )
    item = result["results"][0]
    assert item["status"] == "reachable"
    assert item["matched_cidr"] == "10.1.0.0/16"
    assert item["metric"] == 500
    assert item["next_hops"] == ["10.1.0.1"]


def test_lowest_metric_kept_within_lpm_set():
    # /8 与 /8 同前缀：metric 20 胜出，metric 50 被丢弃
    result = resolve_routes(
        [
            ("0.0.0.0/0", "1.1.1.1", 100),
            ("10.0.0.0/8", "10.0.0.1", 50),
            ("10.0.0.0/8", "10.0.0.2", 20),
            ("10.0.0.0/8", "10.0.0.3", 20),
        ],
        ["10.2.3.4"],
    )
    item = result["results"][0]
    assert item["matched_cidr"] == "10.0.0.0/8"
    assert item["metric"] == 20
    assert item["next_hops"] == ["10.0.0.2", "10.0.0.3"]


def test_ecmp_next_hops_deduplicated_in_input_order():
    # 同一路由重复、相同下一跳跨不同最低 metric 路由：去重且按 routes 顺序
    result = resolve_routes(
        [
            ("10.1.0.0/16", "10.1.0.9", 90),  # 先出现但 metric 高，落选
            ("10.1.0.0/16", "10.1.0.1", 60),
            ("10.1.0.0/16", "10.1.0.2", 60),
            ("10.1.0.0/16", "10.1.0.1", 60),  # 完全重复
            ("10.1.0.0/16", "10.1.0.2", 60),  # 完全重复
        ],
        ["10.1.0.5"],
    )
    assert result["results"][0]["next_hops"] == ["10.1.0.1", "10.1.0.2"]


def test_higher_metric_hop_does_not_pollute_dedup_order():
    # 高 metric 路由虽然先出现且下一跳不同，但不参与 ECMP，也不影响去重顺序
    result = resolve_routes(
        [
            ("10.0.0.0/8", "10.9.9.9", 90),
            ("10.0.0.0/8", "10.0.0.2", 10),
            ("10.0.0.0/8", "10.0.0.1", 10),
        ],
        ["10.0.0.9"],
    )
    assert result["results"][0]["next_hops"] == ["10.0.0.2", "10.0.0.1"]


def test_unreachable_target_shape():
    result = resolve_routes([("10.0.0.0/8", "10.0.0.1", 0)], ["8.8.8.8"])
    assert result == {
        "route_count": 1,
        "target_count": 1,
        "results": [
            {
                "target": "8.8.8.8",
                "status": "unreachable",
                "matched_cidr": None,
                "metric": None,
                "next_hops": [],
            }
        ],
        "summary": {
            "reachable_count": 0,
            "unreachable_count": 1,
            "next_hop_counts": [],
        },
    }


def test_empty_routes_all_unreachable():
    result = resolve_routes([], ["1.2.3.4", "5.6.7.8"])
    assert [r["status"] for r in result["results"]] == ["unreachable", "unreachable"]
    assert result["summary"] == {
        "reachable_count": 0,
        "unreachable_count": 2,
        "next_hop_counts": [],
    }


def test_results_preserve_target_order_and_normalization():
    result = resolve_routes(
        [("0.0.0.0/0", "1.1.1.1", 0)],
        ["8.8.8.8", "1.1.1.1", "8.8.4.4"],
    )
    assert [r["target"] for r in result["results"]] == [
        "8.8.8.8",
        "1.1.1.1",
        "8.8.4.4",
    ]


def test_next_hop_counts_ecmp_counted_separately():
    # 三个等价下一跳均命中两个目标（ECMP 结果里每个下一跳分别计数）；
    # 第三个目标不可达，不参与计数
    result = resolve_routes(
        [
            ("10.0.0.0/8", "10.0.0.1", 10),
            ("10.0.0.0/8", "10.0.0.2", 10),
            ("10.0.0.0/8", "10.0.0.3", 10),
        ],
        ["10.0.0.9", "10.1.0.9", "8.8.8.8"],
    )
    assert result["summary"] == {
        "reachable_count": 2,
        "unreachable_count": 1,
        "next_hop_counts": [
            {"next_hop": "10.0.0.1", "target_count": 2},
            {"next_hop": "10.0.0.2", "target_count": 2},
            {"next_hop": "10.0.0.3", "target_count": 2},
        ],
    }


def test_next_hop_counts_partial_ecmp_overlap():
    # a 命中 h1/h2，b 命中 h2/h3：h2 计数 2，h1/h3 各 1
    result = resolve_routes(
        [
            ("10.0.0.0/8", "10.0.0.1", 10),
            ("10.0.0.0/8", "10.0.0.2", 10),
            ("10.0.0.0/8", "10.0.0.3", 10),
            ("10.1.0.0/16", "10.0.0.1", 5),  # LPM 更高前缀：a 改走 h1 单跳
        ],
        ["10.1.0.9", "10.2.0.9"],
    )
    assert result["results"][0]["next_hops"] == ["10.0.0.1"]
    assert result["results"][1]["next_hops"] == ["10.0.0.1", "10.0.0.2", "10.0.0.3"]
    assert result["summary"]["next_hop_counts"] == [
        {"next_hop": "10.0.0.1", "target_count": 2},
        {"next_hop": "10.0.0.2", "target_count": 1},
        {"next_hop": "10.0.0.3", "target_count": 1},
    ]


def test_next_hop_counts_ordered_by_first_route_occurrence():
    # h-a 首次出现在 index 0（对目标 a 落选），h-b 在 index 2、h-c 在 index 1；
    # 汇总顺序按 routes 中首次出现位置：h-a(0), h-c(1), h-b(2)
    result = resolve_routes(
        [
            ("10.0.0.0/8", "10.0.0.9", 90),
            ("10.0.0.0/8", "10.0.0.3", 10),
            ("10.0.0.0/8", "10.0.0.2", 10),
            ("9.0.0.0/8", "10.0.0.9", 10),
        ],
        ["10.0.0.9", "9.0.0.9"],
    )
    assert result["summary"]["next_hop_counts"] == [
        {"next_hop": "10.0.0.9", "target_count": 1},
        {"next_hop": "10.0.0.3", "target_count": 1},
        {"next_hop": "10.0.0.2", "target_count": 1},
    ]


def test_bare_ip_route_cidr_is_slash32():
    result = resolve_routes(
        [("8.8.8.8", "1.1.1.1", 5)], ["8.8.8.8", "8.8.8.9"]
    )
    assert result["results"][0]["matched_cidr"] == "8.8.8.8/32"
    assert result["results"][0]["metric"] == 5
    assert result["results"][1]["status"] == "unreachable"


def test_default_route_covers_entire_space_without_enumeration():
    result = resolve_routes(
        [("0.0.0.0/0", "1.1.1.1", 65535)],
        ["0.0.0.0", "127.0.0.1", "255.255.255.255"],
    )
    assert all(r["status"] == "reachable" for r in result["results"])
    assert all(r["matched_cidr"] == "0.0.0.0/0" for r in result["results"])
    assert result["summary"] == {
        "reachable_count": 3,
        "unreachable_count": 0,
        "next_hop_counts": [
            {"next_hop": "1.1.1.1", "target_count": 3}
        ],
    }


def test_duplicate_targets_raises_with_indexes():
    with pytest.raises(DuplicateTargetsError) as exc:
        resolve_routes(
            [("0.0.0.0/0", "1.1.1.1", 0)],
            ["8.8.8.8", "9.9.9.9", "8.8.8.8", "9.9.9.9"],
        )
    assert exc.value.duplicates == [
        {"target": "8.8.8.8", "indexes": [0, 2]},
        {"target": "9.9.9.9", "indexes": [1, 3]},
    ]


def test_invalid_route_inputs_raise():
    with pytest.raises(InvalidMetricError):
        resolve_routes([("10.0.0.0/8", "1.1.1.1", 65536)], ["10.0.0.1"])
    with pytest.raises(InvalidMetricError):
        resolve_routes([("10.0.0.0/8", "1.1.1.1", True)], ["10.0.0.1"])
    with pytest.raises(InvalidIPv4AddressError):
        resolve_routes([("10.0.0.0/8", "1.1.1.1/32", 0)], ["10.0.0.1"])
    with pytest.raises(InvalidIPv4AddressError):
        resolve_routes([("10.0.0.0/8", "::1", 0)], ["10.0.0.1"])
    with pytest.raises(Exception):
        # CIDR 非法（主机位非零）
        resolve_routes([("10.0.0.1/8", "1.1.1.1", 0)], ["10.0.0.1"])
    with pytest.raises(InvalidIPv4AddressError):
        resolve_routes([("10.0.0.0/8", "1.1.1.1", 0)], ["2001:db8::1"])


def test_size_limits():
    with pytest.raises(ValueError):
        resolve_routes([], [])
    many_routes = [
        ("10.0.0.0/8", f"10.0.{i // 256}.{(i % 256) + 1}", i % (METRIC_MAX + 1))
        for i in range(ROUTES_MAX + 1)
    ]
    with pytest.raises(ValueError):
        resolve_routes(many_routes, ["10.0.0.1"])
    with pytest.raises(ValueError):
        resolve_routes(
            [("0.0.0.0/0", "1.1.1.1", 0)],
            [f"8.8.{i // 256}.{i % 256}" for i in range(TARGETS_MAX + 1)],
        )


def test_max_limits_accepted():
    routes = [("0.0.0.0/0", "1.1.1.1", 0) for _ in range(ROUTES_MAX)]
    targets = [f"8.8.{i // 256}.{i % 256}" for i in range(TARGETS_MAX)]
    result = resolve_routes(routes, targets)
    assert result["route_count"] == ROUTES_MAX
    assert result["target_count"] == TARGETS_MAX
    assert len(result["results"]) == TARGETS_MAX
    # 重复下一跳只保留一项，命中全部 500 个目标
    assert result["summary"]["next_hop_counts"] == [
        {"next_hop": "1.1.1.1", "target_count": TARGETS_MAX}
    ]


# ---------- 属性测试：与逐路由暴力 oracle 对比 ----------

def _oracle_resolve(routes, target_value):
    """暴力 LPM + 最低 metric + 输入顺序去重，返回 (cidr, metric, hops) 或 None。"""
    matched = []
    best_prefix = -1
    for cidr, next_hop, metric in routes:
        start, end, text = parse_cidr_v4(cidr)
        if start <= target_value <= end:
            prefixlen = int(text.rsplit("/", 1)[1])
            if prefixlen > best_prefix:
                best_prefix = prefixlen
                matched = [(text, next_hop, metric)]
            elif prefixlen == best_prefix:
                matched.append((text, next_hop, metric))
    if not matched:
        return None
    min_metric = min(m for _, _, m in matched)
    cidr = matched[0][0]
    hops = []
    for _, next_hop, metric in matched:
        if metric == min_metric and next_hop not in hops:
            hops.append(next_hop)
    return cidr, min_metric, hops


def test_resolve_random_oracle():
    rng = random.Random(20260918)
    # 在 10.0.0.0/16 内随机生成不同前缀长度的路由
    base = ip_network("10.0.0.0/16")
    candidate_nets = ["0.0.0.0/0", "10.0.0.0/8"]
    for prefixlen in (17, 20, 24, 28, 30, 32):
        candidate_nets.extend(str(n) for n in base.subnets(new_prefix=prefixlen))

    for _ in range(200):
        n_routes = rng.randint(0, 20)
        route_cidrs = [rng.choice(candidate_nets) for _ in range(n_routes)]
        raw_routes = []
        hops_pool = [f"192.0.2.{i}" for i in range(1, 9)]
        for cidr in route_cidrs:
            # 允许重复下一跳、重复 metric
            raw_routes.append(
                (cidr, rng.choice(hops_pool), rng.randint(0, 5))
            )

        targets = [
            f"10.0.{rng.randint(0, 255)}.{rng.randint(0, 255)}"
            for _ in range(rng.randint(1, 6))
        ]
        # oracle 前先去重（接口语义要求唯一目标）
        uniq = list(dict.fromkeys(targets))
        result = resolve_routes(raw_routes, uniq)

        assert [r["target"] for r in result["results"]] == uniq
        expected_counts: dict[str, int] = {}
        for item, target_text in zip(result["results"], uniq):
            value = parse_address_v4(target_text)[0]
            expected = _oracle_resolve(raw_routes, value)
            if expected is None:
                assert item["status"] == "unreachable"
                assert item["matched_cidr"] is None
                assert item["metric"] is None
                assert item["next_hops"] == []
                continue
            cidr, metric, hops = expected
            assert item["status"] == "reachable"
            assert item["matched_cidr"] == cidr
            assert item["metric"] == metric
            assert item["next_hops"] == hops
            for hop in hops:
                expected_counts[hop] = expected_counts.get(hop, 0) + 1

        # 汇总计数与逐目标展开一致
        got_counts = {
            c["next_hop"]: c["target_count"]
            for c in result["summary"]["next_hop_counts"]
        }
        assert got_counts == expected_counts
        # 汇总排序键：routes 中首次出现的序号单调递增
        first_index = {}
        for i, (_, hop, _) in enumerate(raw_routes):
            first_index.setdefault(hop, i)
        ordered = [c["next_hop"] for c in result["summary"]["next_hop_counts"]]
        assert ordered == sorted(ordered, key=lambda h: first_index[h])

        reachable = sum(1 for r in result["results"] if r["status"] == "reachable")
        assert result["summary"]["reachable_count"] == reachable
        assert result["summary"]["unreachable_count"] == len(uniq) - reachable
