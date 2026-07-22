"""将监控问答中的自然语言时间归一化为本地时间戳范围。"""
from __future__ import annotations

import calendar
import re
import unicodedata
from datetime import datetime, timedelta
from typing import Any


_CN_DIGITS = {
    "零": 0,
    "〇": 0,
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
}
_CN_NUMBER = r"[〇零一二两三四五六七八九十]{1,4}"
_SHORT_NUMBER = rf"(?:\d{{1,2}}|{_CN_NUMBER})"
_LONG_NUMBER = rf"(?:\d{{1,4}}|{_CN_NUMBER})"
_PERIOD_NAMES = ("凌晨", "清晨", "早上", "上午", "中午", "正午", "下午", "傍晚", "晚上", "晚间", "夜间", "午夜", "半夜")
_PERIOD_PATTERN = "|".join(_PERIOD_NAMES)
_PERIOD_ALIAS = {
    "清晨": "早上",
    "正午": "中午",
    "晚间": "晚上",
    "夜间": "晚上",
    "半夜": "凌晨",
    "午夜": "凌晨",
}
_PERIOD_SPANS = {
    "凌晨": (0, 6),
    "早上": (6, 9),
    "上午": (8, 12),
    "中午": (11, 14),
    "下午": (12, 18),
    "傍晚": (17, 19),
    "晚上": (18, 24),
}
_FUZZY_MARKERS = ("左右", "前后", "大约", "约莫", "约", "差不多", "将近")


def normalize_time_range(question: str, now: datetime | None = None) -> tuple[float, float] | None:
    """将自然语言中的日期、时段和钟点转换为具体的起止时间戳。"""
    text = _normalize_text(question)
    if not text:
        return None
    current = _local_now(now)
    recent = _relative_duration_range(text, current)
    if recent is not None:
        return _timestamp_range(*recent)

    date_tokens = _date_tokens(text, current)
    time_tokens = _time_tokens(text)
    date_range = _explicit_date_range(text, date_tokens, time_tokens)
    if date_range is not None:
        return _timestamp_range(*date_range)

    date_span = _date_span(text, current, date_tokens)
    if date_span is None:
        date_span = _day_span(current)
    start_day, end_day = date_span

    if len(date_tokens) >= 2:
        return _timestamp_range(start_day, end_day)
    if end_day - start_day > timedelta(days=1):
        return _timestamp_range(start_day, end_day)

    time_range = _clock_range(text, start_day, time_tokens, current)
    if time_range is not None:
        return _timestamp_range(*time_range)

    period_range = _period_range(text, start_day)
    if period_range is not None:
        return _timestamp_range(*period_range)

    if date_tokens or _has_date_phrase(text):
        return _timestamp_range(start_day, end_day)
    return None


def has_time_expression(question: str) -> bool:
    """判断用户是否显式给出了应当作为查询约束的时间表达。"""
    text = _normalize_text(question)
    if not text:
        return False
    patterns = (
        r"(?:大前天|前天|昨天|昨日|今天|今日|明天|明日|后天|大后天|前年|去年|今年|明年|后年)",
        r"(?:上上|下下|上|下|本|这)?(?:周|星期)[一二三四五六日天末]?",
        rf"(?:最近|近|过去)?(?:{_SHORT_NUMBER}|半|一刻)(?:个)?(?:分钟?|小时|天|日|周|星期|月|年|刻钟)(?:前|后)?",
        rf"{_LONG_NUMBER}\s*(?:年|[./-])\s*{_SHORT_NUMBER}\s*(?:月|[./-])\s*{_SHORT_NUMBER}(?:日|号)?",
        rf"{_SHORT_NUMBER}月(?:{_SHORT_NUMBER}(?:日|号)?)?",
        rf"(?<![\d年月]){_SHORT_NUMBER}(?:日|号)",
        r"\d{1,2}[:：]\d{1,2}(?::\d{1,2})?",
        rf"(?:{_PERIOD_PATTERN})?{_SHORT_NUMBER}(?:点|时)",
        rf"(?:{_PERIOD_PATTERN})",
        r"(?:稍后|刚才|刚刚|白天|夜里|夜晚|上上个月|上个月|本月|这个月|下个月|下下个月)",
    )
    return any(re.search(pattern, text) for pattern in patterns)


def parse_model_time_range(value: Any, now: datetime | None = None) -> tuple[float, float] | None:
    """校验并转换模型返回的规范时间范围，只接受明确的端点。"""
    if isinstance(value, dict):
        start_value = value.get("start") or value.get("startTime") or value.get("begin")
        end_value = value.get("end") or value.get("endTime") or value.get("finish")
    elif isinstance(value, (list, tuple)) and len(value) == 2:
        start_value, end_value = value
    else:
        return None
    start = _parse_time_endpoint(start_value, now)
    end = _parse_time_endpoint(end_value, now)
    if start is None or end is None or end <= start:
        return None
    return start, end


def _normalize_text(question: str) -> str:
    text = unicodedata.normalize("NFKC", str(question or "")).lower()
    substitutions = (
        (r"\byesterday\b", "昨天"),
        (r"\btoday\b", "今天"),
        (r"\btomorrow\b", "明天"),
        (r"\blastweek\b", "上周"),
        (r"\bnextweek\b", "下周"),
        (r"\bthisweek\b", "本周"),
    )
    for pattern, replacement in substitutions:
        text = re.sub(pattern, replacement, text, flags=re.I)
    text = re.sub(r"(?i)(\d{1,2}(?::\d{2})?)\s*p\.?m\.?", r"下午\1", text)
    text = re.sub(r"(?i)(\d{1,2}(?::\d{2})?)\s*a\.?m\.?", r"上午\1", text)
    text = text.replace("礼拜", "星期").replace("周天", "周日")
    text = text.replace("夜里", "晚上").replace("夜晚", "晚上")
    text = text.replace("午后", "下午").replace("傍晚时分", "傍晚")
    text = re.sub(r"[\s()（）\[\]【】]", "", text)
    return text


def _local_now(now: datetime | None) -> datetime:
    current = now or datetime.now().astimezone()
    if current.tzinfo is None:
        current = current.replace(tzinfo=datetime.now().astimezone().tzinfo)
    return current


def _parse_integer(value: Any) -> int | None:
    token = str(value or "").strip()
    if not token:
        return None
    if re.fullmatch(r"\d+", token):
        return int(token)
    if all(character in _CN_DIGITS for character in token):
        return int("".join(str(_CN_DIGITS[character]) for character in token))
    if "十" not in token:
        return None
    if token.count("十") != 1:
        return None
    left, right = token.split("十", 1)
    if left and (len(left) != 1 or left not in _CN_DIGITS):
        return None
    if right and (len(right) != 1 or right not in _CN_DIGITS):
        return None
    tens = _CN_DIGITS[left] if left else 1
    ones = _CN_DIGITS[right] if right else 0
    return tens * 10 + ones


def _parse_amount(value: Any) -> float | None:
    token = str(value or "").strip()
    if token == "半":
        return 0.5
    if token == "一刻":
        return 1.0
    if re.fullmatch(r"\d+(?:\.\d+)?", token):
        return float(token)
    parsed = _parse_integer(token)
    return float(parsed) if parsed is not None else None


def _day_span(value: datetime) -> tuple[datetime, datetime]:
    start = value.replace(hour=0, minute=0, second=0, microsecond=0)
    return start, start + timedelta(days=1)


def _month_start(value: datetime, shift: int = 0) -> datetime:
    month_index = value.year * 12 + value.month - 1 + shift
    year, month_index = divmod(month_index, 12)
    return value.replace(year=year, month=month_index + 1, day=1, hour=0, minute=0, second=0, microsecond=0)


def _month_span(value: datetime, shift: int = 0) -> tuple[datetime, datetime]:
    start = _month_start(value, shift)
    return start, _month_start(start, 1)


def _make_day(current: datetime, year: int | None, month: int, day: int) -> datetime | None:
    try:
        return current.replace(year=year or current.year, month=month, day=day, hour=0, minute=0, second=0, microsecond=0)
    except ValueError:
        return None


def _date_tokens(text: str, current: datetime) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    patterns = (
        re.compile(rf"(?P<year>{_LONG_NUMBER})\s*(?:年|[./-])\s*(?P<month>{_SHORT_NUMBER})\s*(?:月|[./-])\s*(?P<day>{_SHORT_NUMBER})(?:日|号)?"),
        re.compile(rf"(?<![\d年])(?P<month>{_SHORT_NUMBER})月(?P<day>{_SHORT_NUMBER})(?:日|号)?"),
        re.compile(rf"(?<![\d年月])(?P<day>{_SHORT_NUMBER})(?:日|号)"),
    )
    for index, pattern in enumerate(patterns):
        for match in pattern.finditer(text):
            if any(not (match.end() <= item["start"] or match.start() >= item["end"]) for item in records):
                continue
            month = _parse_integer(match.groupdict().get("month"))
            day = _parse_integer(match.group("day"))
            year = _parse_integer(match.groupdict().get("year"))
            if month is None:
                month = current.month
            if day is None:
                continue
            resolved = _make_day(current, year, month, day)
            if resolved is None:
                continue
            records.append({"start": match.start(), "end": match.end(), "day": resolved, "kind": ("ymd", "md", "d")[index]})
    return sorted(records, key=lambda item: item["start"])


def _relative_duration_range(text: str, current: datetime) -> tuple[datetime, datetime] | None:
    amount_pattern = rf"(?:\d+(?:\.\d+)?|{_CN_NUMBER}|半|一刻)"
    recent = re.search(rf"(?:最近|近|过去)(?P<amount>{amount_pattern})(?:个)?(?P<unit>分钟?|小时|天|日|周|星期|月|年|刻钟)", text)
    if recent:
        amount = _parse_amount(recent.group("amount"))
        seconds = _duration_seconds(amount, recent.group("unit"))
        if seconds is not None:
            return current - timedelta(seconds=seconds), current
    point = re.search(rf"(?P<amount>{amount_pattern})(?:个)?(?P<unit>分钟?|小时|刻钟)(?P<direction>前|后)", text)
    if point:
        amount = _parse_amount(point.group("amount"))
        seconds = _duration_seconds(amount, point.group("unit"))
        if seconds is not None:
            target = current + timedelta(seconds=seconds if point.group("direction") == "后" else -seconds)
            width = 60 if point.group("unit").startswith("分") or point.group("unit") == "刻钟" else 3600
            return target, target + timedelta(seconds=width)
    if "刚才" in text or "刚刚" in text:
        return current - timedelta(minutes=10), current
    if "稍后" in text:
        return current, current + timedelta(hours=1)
    return None


def _duration_seconds(amount: float | None, unit: str) -> float | None:
    if amount is None:
        return None
    if unit in {"分", "分钟"}:
        return amount * 60
    if unit in {"小时"}:
        return amount * 3600
    if unit == "刻钟":
        return amount * 15 * 60
    if unit in {"天", "日"}:
        return amount * 86400
    if unit in {"周", "星期"}:
        return amount * 7 * 86400
    if unit == "月":
        return amount * 30 * 86400
    if unit == "年":
        return amount * 365 * 86400
    return None


def _date_span(text: str, current: datetime, date_tokens: list[dict[str, Any]]) -> tuple[datetime, datetime] | None:
    if date_tokens:
        return _day_span(date_tokens[0]["day"])
    relative = _relative_day_span(text, current)
    if relative is not None:
        return relative
    week = _week_span(text, current)
    if week is not None:
        return week
    month = _month_expression_span(text, current)
    if month is not None:
        return month
    year_match = re.search(r"(?:前年|去年|今年|明年|后年)", text)
    if year_match:
        offsets = {"前年": -2, "去年": -1, "今年": 0, "明年": 1, "后年": 2}
        year = current.year + offsets[year_match.group(0)]
        start = current.replace(year=year, month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
        return start, start.replace(year=year + 1)
    return None


def _relative_day_span(text: str, current: datetime) -> tuple[datetime, datetime] | None:
    phrases = (
        (("大前天",), -3),
        (("前天",), -2),
        (("昨天", "昨日"), -1),
        (("今天", "今日", "当天"), 0),
        (("明天", "明日"), 1),
        (("后天",), 2),
        (("大后天",), 3),
    )
    for values, offset in phrases:
        if any(value in text for value in values):
            start, _ = _day_span(current + timedelta(days=offset))
            return start, start + timedelta(days=1)
    amount_pattern = rf"(?:\d+|{_CN_NUMBER}|半)"
    match = re.search(rf"(?P<amount>{amount_pattern})(?:个)?(?P<unit>天|日|周|星期|月|年)(?P<direction>前|后)", text)
    if not match:
        return None
    amount = _parse_amount(match.group("amount"))
    if amount is None:
        return None
    direction = -1 if match.group("direction") == "前" else 1
    unit = match.group("unit")
    if unit in {"天", "日"}:
        return _day_span(current + timedelta(days=direction * amount))
    if unit in {"周", "星期"}:
        return _day_span(current + timedelta(days=direction * amount * 7))
    if unit == "月":
        target = _month_start(current, direction * int(amount))
        return _month_span(target)
    if unit == "年":
        target = current.replace(year=current.year + direction * int(amount))
        return _day_span(target)
    return None


def _week_span(text: str, current: datetime) -> tuple[datetime, datetime] | None:
    match = re.search(r"(?P<prefix>上上|下下|上|下|本|这)?(?:周|星期)(?P<weekday>[一二三四五六日天末])", text)
    prefix_offsets = {"上上": -2, "上": -1, "本": 0, "这": 0, "下": 1, "下下": 2, None: 0}
    monday, _ = _day_span(current - timedelta(days=current.weekday()))
    if match:
        offset = prefix_offsets[match.group("prefix")]
        week_start = monday + timedelta(days=offset * 7)
        weekday = match.group("weekday")
        if weekday == "末":
            return week_start + timedelta(days=5), week_start + timedelta(days=7)
        index = {"一": 0, "二": 1, "三": 2, "四": 3, "五": 4, "六": 5, "日": 6, "天": 6}[weekday]
        start = week_start + timedelta(days=index)
        return start, start + timedelta(days=1)
    match = re.search(r"(?P<prefix>上上|下下|上|下|本|这)?(?:周|星期)(?![一二三四五六日天末])", text)
    if match:
        week_start = monday + timedelta(days=prefix_offsets[match.group("prefix")] * 7)
        return week_start, week_start + timedelta(days=7)
    return None


def _month_expression_span(text: str, current: datetime) -> tuple[datetime, datetime] | None:
    relative = (("上上个月", -2), ("上个月", -1), ("本月", 0), ("这个月", 0), ("下个月", 1), ("下下个月", 2))
    for phrase, offset in relative:
        if phrase in text:
            return _month_span(current, offset)
    match = re.search(rf"(?:(?P<year>{_LONG_NUMBER})年)?(?P<month>{_SHORT_NUMBER})月(?P<part>初|中旬|中|末|底)?", text)
    if not match:
        return None
    year = _parse_integer(match.group("year"))
    month = _parse_integer(match.group("month"))
    if month is None or not 1 <= month <= 12:
        return None
    base = current.replace(year=year or current.year, month=month, day=1, hour=0, minute=0, second=0, microsecond=0)
    part = match.group("part")
    if part in {None, ""}:
        return base, _month_start(base, 1)
    days = calendar.monthrange(base.year, base.month)[1]
    if part == "初":
        return base, base + timedelta(days=min(10, days))
    if part in {"中", "中旬"}:
        start = base + timedelta(days=10)
        return start, base + timedelta(days=min(20, days))
    start = base + timedelta(days=20)
    return start, _month_start(base, 1)


def _time_tokens(text: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    colon = re.compile(rf"(?P<period>{_PERIOD_PATTERN})?(?P<hour>{_SHORT_NUMBER})[:：](?P<minute>\d{{1,2}})(?:[:：](?P<second>\d{{1,2}}))?")
    point = re.compile(rf"(?P<period>{_PERIOD_PATTERN})?(?P<hour>{_SHORT_NUMBER})(?:点|时)(?P<minute>半|一刻|三刻|{_SHORT_NUMBER})?(?:分)?(?P<second>\d{{1,2}})?(?:秒)?")
    for pattern, precision in ((colon, "minute"), (point, "hour")):
        for match in pattern.finditer(text):
            hour = _parse_integer(match.group("hour"))
            minute_value = match.group("minute")
            second_value = match.group("second")
            if minute_value == "半":
                minute = 30
            elif minute_value == "一刻":
                minute = 15
            elif minute_value == "三刻":
                minute = 45
            else:
                minute = _parse_integer(minute_value) if minute_value else 0
            second = _parse_integer(second_value) if second_value else 0
            if hour is None or minute is None or second is None or not 0 <= minute <= 59 or not 0 <= second <= 59:
                continue
            records.append({
                "start": match.start(),
                "end": match.end(),
                "hour": hour,
                "minute": minute,
                "second": second,
                "period": match.group("period"),
                "precision": "minute" if precision == "minute" or minute_value else "hour",
            })
    records.sort(key=lambda item: (item["start"], -(item["end"] - item["start"])))
    selected: list[dict[str, Any]] = []
    for item in records:
        if any(not (item["end"] <= previous["start"] or item["start"] >= previous["end"]) for previous in selected):
            continue
        selected.append(item)
    return selected


def _resolve_hour(hour: int, period: str | None) -> int | None:
    if not 0 <= hour <= 23:
        return None
    resolved_period = _PERIOD_ALIAS.get(period or "", period)
    if resolved_period in {"下午", "傍晚", "晚上"} and 1 <= hour < 12:
        return hour + 12
    if resolved_period == "中午" and 1 <= hour <= 5:
        return hour + 12
    if resolved_period in {"凌晨", "早上", "上午"} and hour == 12:
        return 0
    return hour


def _clock_datetime(day: datetime, token: dict[str, Any], inherited_period: str | None = None) -> datetime | None:
    hour = _resolve_hour(int(token["hour"]), token.get("period") or inherited_period)
    if hour is None:
        return None
    return day.replace(hour=hour, minute=int(token["minute"]), second=int(token["second"]), microsecond=0)


def _clock_range(text: str, day: datetime, tokens: list[dict[str, Any]], current: datetime) -> tuple[datetime, datetime] | None:
    for index in range(len(tokens) - 1):
        first, second = tokens[index], tokens[index + 1]
        bridge = text[first["end"]:second["start"]]
        start = _clock_datetime(day, first)
        end_day = day
        if "次日" in bridge or "第二天" in bridge:
            end_day = day + timedelta(days=1)
        else:
            relative_end_days = (("大前天", -3), ("前天", -2), ("昨天", -1), ("今天", 0), ("明天", 1), ("后天", 2), ("大后天", 3))
            for marker, offset in relative_end_days:
                if marker in bridge:
                    end_day, _ = _day_span(current + timedelta(days=offset))
                    break
        cleaned_bridge = bridge
        for marker in ("大前天", "前天", "昨天", "今天", "明天", "后天", "大后天", "次日", "第二天"):
            cleaned_bridge = cleaned_bridge.replace(marker, "")
        if not re.fullmatch(r"(?:左右|前后|大约|约莫|约|差不多|将近|从|起|之间)*(?:到|至|[-—~～－])(?:左右|前后|大约|约莫|约|差不多|将近|止|之间)*", cleaned_bridge):
            continue
        inherited_period = first.get("period")
        if (
            inherited_period in {"晚上", "晚间", "夜间"}
            and not second.get("period")
            and int(first["hour"]) >= 6
            and int(second["hour"]) <= 5
            and end_day == day
        ):
            end_day += timedelta(days=1)
            inherited_period = None
        end = _clock_datetime(end_day, second, inherited_period)
        if start is None or end is None:
            continue
        if end < start and end_day == day:
            end += timedelta(days=1)
        if end > start:
            return start, end
    if not tokens:
        return None
    token = tokens[0]
    start = _clock_datetime(day, token)
    if start is None:
        return None
    window = text[max(0, token["start"] - 6):min(len(text), token["end"] + 6)]
    if any(marker in window for marker in _FUZZY_MARKERS):
        return start - timedelta(minutes=30), start + timedelta(minutes=30)
    if token["precision"] == "hour":
        return start, start + timedelta(hours=1)
    return start, start + timedelta(minutes=1)


def _period_range(text: str, day: datetime) -> tuple[datetime, datetime] | None:
    for period in _PERIOD_NAMES:
        if period not in text:
            continue
        normalized_period = _PERIOD_ALIAS.get(period, period)
        start_hour, end_hour = _PERIOD_SPANS[normalized_period]
        start = day.replace(hour=start_hour, minute=0, second=0, microsecond=0)
        end = day + timedelta(days=1) if end_hour == 24 else day.replace(hour=end_hour, minute=0, second=0, microsecond=0)
        return start, end
    if "白天" in text:
        return day.replace(hour=6), day.replace(hour=18)
    return None


def _explicit_date_range(text: str, dates: list[dict[str, Any]], times: list[dict[str, Any]]) -> tuple[datetime, datetime] | None:
    if len(dates) < 2:
        return None
    for left, right in zip(dates, dates[1:]):
        bridge = text[left["end"]:right["start"]]
        if not re.search(r"到|至|[-—~～－]", bridge):
            continue
        left_times = [item for item in times if left["end"] <= item["start"] < right["start"]]
        right_times = [item for item in times if item["start"] >= right["end"]]
        start = _clock_datetime(left["day"], left_times[-1]) if left_times else left["day"]
        end = _clock_datetime(right["day"], right_times[0]) if right_times else right["day"] + timedelta(days=1)
        if start is not None and end is not None and end > start:
            return start, end
    return None


def _has_date_phrase(text: str) -> bool:
    return bool(re.search(rf"(?:{_LONG_NUMBER}年)?{_SHORT_NUMBER}月|{_SHORT_NUMBER}(?:日|号)|大前天|前天|昨天|今日|今天|明天|后天|大后天|(?:上|下|本|这)?(?:周|星期)|(?:上|下|本|这个)?月|前年|去年|今年|明年|后年", text))


def _timestamp_range(start: datetime, end: datetime) -> tuple[float, float]:
    return start.timestamp(), end.timestamp()


def _parse_time_endpoint(value: Any, now: datetime | None) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value or "").strip()
    if not text:
        return None
    if re.fullmatch(r"\d+(?:\.\d+)?", text):
        return float(text)
    normalized = text.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        parsed = None
    if parsed is None:
        for pattern in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y/%m/%d %H:%M:%S", "%Y/%m/%d %H:%M"):
            try:
                parsed = datetime.strptime(text, pattern)
                break
            except ValueError:
                continue
    if parsed is None:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=_local_now(now).tzinfo)
    return parsed.timestamp()