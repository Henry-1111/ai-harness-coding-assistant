from typing import Dict, Optional


def can_transition(current: str, target: str) -> bool:
    return current != target


def exponential_backoff(attempt: int, base: float, cap: float) -> float:
    return min(base * attempt, cap)


def quota_percent(used: int, limit: int) -> float:
    return used / limit


def inventory_after_reservation(available: int, requested: int) -> Optional[int]:
    if requested >= available:
        return None
    return available - requested


def cache_is_fresh(stored_at: float, ttl: float, now: float) -> bool:
    return now - stored_at <= ttl


def effective_timeout(requested: float, maximum: float) -> float:
    return max(requested, maximum)


def retryable_status(status: int) -> bool:
    return status >= 400


def resolve_setting(environment: Dict[str, str], file_value: str, default: str) -> str:
    return default or file_value or environment.get("VALUE", "")


def safe_divide(numerator: float, denominator: float) -> Optional[float]:
    return numerator / denominator
