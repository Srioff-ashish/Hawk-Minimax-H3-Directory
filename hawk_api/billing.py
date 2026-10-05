"""What the paid services cost: balances and spend per model, for the Billing pages in Studio.

Each provider is asked in its own way, because each exposes something different:

* **Atlas Cloud** has a billing API (``/public/v1``) that any account key can read: the balance, and daily
  cost and usage buckets per model for up to 180 days.
* **OpenRouter** answers ``/credits`` (bought and used) and ``/key`` (this key's daily, weekly and monthly
  spend) with the ordinary key. The per-model breakdown (``/activity``, last 30 days) needs a *management*
  key, which is optional: without one the page shows totals and says what the extra key would add.
* **Google** (Gemini API / AI Studio) has no API for the prepaid balance or for spend -- Google's own docs say
  both live only in the AI Studio Billing tab. So this server's own record is the source: every Nano Banana
  image keeps its estimated price on its asset (``source.cost_usd``), and a deleted asset keeps its source in
  the history table, so the total survives deletes. Images made before prices were recorded are counted at
  the 1K price and marked as estimates.

Money from Atlas arrives as six-decimal strings; everything returned here is a float in USD.

Torch-free (httpx only), like google_images, so the API process and the tests import it without a GPU.
"""

from __future__ import annotations

import datetime as dt
from typing import Iterable

import httpx

from . import google_images

ATLAS_BILLING_URL = "https://api.atlascloud.ai/public/v1"
OPENROUTER_URL = "https://openrouter.ai/api/v1"

#: Where each provider's own billing pages are, for the "open in ..." links.
CONSOLE = {
    "atlas": {"Billing": "https://www.atlascloud.ai/console/billing",
              "Console": "https://www.atlascloud.ai/console"},
    "openrouter": {"Credits": "https://openrouter.ai/settings/credits",
                   "Activity": "https://openrouter.ai/activity",
                   "Management keys": "https://openrouter.ai/settings/management-keys"},
    "google": {"AI Studio billing": "https://aistudio.google.com/billing",
               "AI Studio usage": "https://aistudio.google.com/usage",
               "Cloud billing reports": "https://console.cloud.google.com/billing/reports"},
}

#: The longest range Atlas accepts in one query.
MAX_DAYS = 180
#: OpenRouter's /activity covers only the last 30 completed UTC days.
OPENROUTER_ACTIVITY_DAYS = 30

GOOGLE_ENGINES = tuple(google_images.PRICES)


class BillingError(RuntimeError):
    """A provider's billing API could not be read. The message is safe to show: it never carries a key."""


def _money(value) -> float:
    """Atlas's {"value": "1.230000", "currency": "usd"}, or a bare number, as a float."""
    if isinstance(value, dict):
        value = value.get("value")
    try:
        return float(value or 0)
    except (TypeError, ValueError):
        return 0.0


def _period(days: int, today: dt.date | None = None) -> tuple[dt.date, dt.date, int]:
    """(first day, day after the last, days): the last ``days`` UTC days, today included."""
    days = max(1, min(int(days or 30), MAX_DAYS))
    today = today or dt.datetime.now(dt.timezone.utc).date()
    return today - dt.timedelta(days=days - 1), today + dt.timedelta(days=1), days


def _daily(amounts: dict[str, float], start: dt.date, end: dt.date) -> list[dict]:
    """One row per day of the period, zero where nothing was spent, so a chart has no gaps."""
    rows, day = [], start
    while day < end:
        key = day.isoformat()
        rows.append({"date": key, "amount": round(amounts.get(key, 0.0), 6)})
        day += dt.timedelta(days=1)
    return rows


def _models(table: dict[str, dict]) -> list[dict]:
    rows = [dict(row, amount=round(row["amount"], 6)) for row in table.values()]
    return sorted(rows, key=lambda row: (-row["amount"], row["model"]))


def _period_view(start: dt.date, end: dt.date, days: int) -> dict:
    return {"start": start.isoformat(), "end": (end - dt.timedelta(days=1)).isoformat(), "days": days}


def _error_text(response: httpx.Response, service: str) -> str:
    try:
        body = response.json()
    except ValueError:
        body = {}
    error = body.get("error") if isinstance(body, dict) else None
    message = (error.get("message") if isinstance(error, dict) else error) or response.reason_phrase
    return f"{service} answered {response.status_code}: {str(message)[:200]}"


# ------------------------------------------------------------------------------------------------- Atlas


async def _atlas_get(http: httpx.AsyncClient, key: str, path: str, params: list | None = None) -> dict:
    try:
        response = await http.get(f"{ATLAS_BILLING_URL}{path}", params=params,
                                  headers={"Authorization": f"Bearer {key}"})
    except httpx.HTTPError as exc:
        raise BillingError(f"Atlas Cloud could not be reached: {type(exc).__name__}") from None
    if response.status_code >= 400:
        raise BillingError(_error_text(response, "Atlas Cloud"))
    return response.json()


async def _atlas_buckets(http: httpx.AsyncClient, key: str, path: str, params: list) -> list[dict]:
    """Every daily bucket of a paged /model-costs or /model-usage query."""
    buckets, page = [], None
    for _ in range(50):  # 180 days at 1000 rows a page is one page; the bound only stops a runaway cursor
        body = await _atlas_get(http, key, path, params + ([("page", page)] if page else []))
        buckets += body.get("data") or []
        page = body.get("next_page")
        if not body.get("has_more") or not page:
            break
    return buckets


def _atlas_units(usage: dict) -> dict:
    """The usage figures worth showing for one model: requests, and tokens, images or video seconds."""
    usage = usage or {}
    out = {"requests": int(usage.get("requests") or 0)}
    tokens = usage.get("tokens") or {}
    if tokens:
        out["input_tokens"] = int(tokens.get("input") or 0) + int(tokens.get("cache_read") or 0)
        out["output_tokens"] = int(tokens.get("output") or 0)
    if isinstance(usage.get("images"), dict):
        out["images"] = int(usage["images"].get("count") or 0)
    if isinstance(usage.get("video"), dict):
        out["video_seconds"] = float(usage["video"].get("seconds") or 0)
    return out


async def atlas_report(key: str, days: int = 30, *, http: httpx.AsyncClient | None = None,
                       today: dt.date | None = None) -> dict:
    """Balance, daily spend and spend per model on Atlas Cloud, account-wide."""
    if not key:
        return {"provider": "atlas", "configured": False, "links": CONSOLE["atlas"],
                "message": "No Atlas key on the server. Add ATLAS_API_KEY, or set one under Connect & settings."}
    start, end, days = _period(days, today)
    params = [("start_date", start.isoformat()), ("end_date", end.isoformat()), ("scope", "account"),
              ("group_by[]", "model"), ("limit", "1000")]
    owned = http is None
    http = http or httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=15.0))
    try:
        balance = await _atlas_get(http, key, "/balance")
        costs = await _atlas_buckets(http, key, "/model-costs", params)
        usage = await _atlas_buckets(http, key, "/model-usage", params)
    finally:
        if owned:
            await http.aclose()

    per_day: dict[str, float] = {}
    per_model: dict[str, dict] = {}
    per_type: dict[str, float] = {}

    def row(result: dict) -> dict:
        model = result.get("model") or {}
        name = model.get("name") or model.get("id") or "unknown"
        kind = result.get("model_type") or model.get("type") or ""
        return per_model.setdefault(name, {"model": name, "type": kind, "amount": 0.0, "requests": 0})

    for bucket in costs:
        for result in bucket.get("results") or []:
            amount = _money(result.get("amount"))
            per_day[bucket.get("date", "")] = per_day.get(bucket.get("date", ""), 0.0) + amount
            entry = row(result)
            entry["amount"] += amount
            per_type[entry["type"] or "other"] = per_type.get(entry["type"] or "other", 0.0) + amount
    for bucket in usage:
        for result in bucket.get("results") or []:
            entry = row(result)
            for field, value in _atlas_units(result.get("usage")).items():
                entry[field] = entry.get(field, 0) + value

    return {
        "provider": "atlas", "configured": True, "currency": "usd", "period": _period_view(start, end, days),
        "balance": {"available": _money(balance.get("available")), "cash": _money(balance.get("cash")),
                    "bonus": _money(balance.get("bonus")) + _money(balance.get("subscription_bonus")),
                    "frozen": _money(balance.get("frozen"))},
        "total": round(sum(per_day.values()), 6),
        "daily": _daily(per_day, start, end),
        "by_type": {kind: round(amount, 6) for kind, amount in sorted(per_type.items(), key=lambda kv: -kv[1])},
        "models": _models(per_model),
        "links": CONSOLE["atlas"],
    }


# -------------------------------------------------------------------------------------------- OpenRouter


async def _openrouter_get(http: httpx.AsyncClient, key: str, path: str) -> httpx.Response:
    try:
        return await http.get(f"{OPENROUTER_URL}{path}", headers={"Authorization": f"Bearer {key}"})
    except httpx.HTTPError as exc:
        raise BillingError(f"OpenRouter could not be reached: {type(exc).__name__}") from None


async def openrouter_report(key: str, management_key: str = "", days: int = 30, *,
                            http: httpx.AsyncClient | None = None, today: dt.date | None = None) -> dict:
    """Credits bought and used, this key's spend, and -- with a management key -- spend per model."""
    if not (key or management_key):
        return {"provider": "openrouter", "configured": False, "links": CONSOLE["openrouter"],
                "message": "No OpenRouter key on the server. Add OPENROUTER_API_KEY, or set one under "
                           "Connect & settings."}
    days = max(1, min(int(days or 30), OPENROUTER_ACTIVITY_DAYS))
    start, end, days = _period(days, today)
    notes: list[str] = []
    owned = http is None
    http = http or httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=15.0))
    try:
        # /credits is documented as management-only but answers an ordinary key too; try the management key
        # first when there is one, so a change on OpenRouter's side does not empty the page.
        credits, last = None, "OpenRouter answered nothing"
        for candidate in [k for k in (management_key, key) if k]:
            response = await _openrouter_get(http, candidate, "/credits")
            if response.status_code < 400:
                credits = (response.json() or {}).get("data") or {}
                break
            last = _error_text(response, "OpenRouter")
        if credits is None:
            raise BillingError(last)
        key_info = {}
        if key:
            response = await _openrouter_get(http, key, "/key")
            if response.status_code < 400:
                key_info = (response.json() or {}).get("data") or {}
        activity = None
        if management_key:
            response = await _openrouter_get(http, management_key, "/activity")
            if response.status_code < 400:
                activity = (response.json() or {}).get("data") or []
            else:
                notes.append(f"Per-model activity unavailable: {_error_text(response, 'OpenRouter')}")
        else:
            notes.append("Add an OpenRouter management key under Connect & settings to see spend per model "
                         "and per day (OpenRouter keeps the last 30 days).")
    finally:
        if owned:
            await http.aclose()

    bought, used = _money(credits.get("total_credits")), _money(credits.get("total_usage"))
    report = {
        "provider": "openrouter", "configured": True, "currency": "usd", "period": _period_view(start, end, days),
        "balance": {"available": round(bought - used, 6), "bought": bought, "used": used},
        "key": {
            "label": key_info.get("label", ""), "limit": key_info.get("limit"),
            "limit_remaining": key_info.get("limit_remaining"), "usage": _money(key_info.get("usage")),
            "today": _money(key_info.get("usage_daily")), "week": _money(key_info.get("usage_weekly")),
            "month": _money(key_info.get("usage_monthly")), "expires_at": key_info.get("expires_at"),
        } if key_info else None,
        "notes": notes, "links": CONSOLE["openrouter"],
    }
    if activity is None:
        report.update(total=None, daily=None, models=None)
        return report

    per_day: dict[str, float] = {}
    per_model: dict[str, dict] = {}
    first, last_day = start.isoformat(), (end - dt.timedelta(days=1)).isoformat()
    for item in activity:
        date = str(item.get("date") or "")[:10]
        if not (first <= date <= last_day):
            continue
        amount = _money(item.get("usage")) + _money(item.get("byok_usage_inference"))
        per_day[date] = per_day.get(date, 0.0) + amount
        name = item.get("model") or "unknown"
        entry = per_model.setdefault(name, {"model": name, "type": "text", "amount": 0.0, "requests": 0,
                                            "input_tokens": 0, "output_tokens": 0, "providers": []})
        entry["amount"] += amount
        entry["requests"] += int(item.get("requests") or 0)
        entry["input_tokens"] += int(item.get("prompt_tokens") or 0)
        entry["output_tokens"] += int(item.get("completion_tokens") or 0)
        if item.get("provider_name") and item["provider_name"] not in entry["providers"]:
            entry["providers"].append(item["provider_name"])
    report.update(total=round(sum(per_day.values()), 6), daily=_daily(per_day, start, end),
                  models=_models(per_model))
    return report


# ------------------------------------------------------------------------------------------------ Google


def google_report(records: Iterable[dict], days: int = 30, *, configured: bool = True,
                  today: dt.date | None = None) -> dict:
    """Nano Banana spend as this server recorded it. ``records`` are asset dicts, live and deleted."""
    start, end, days = _period(days, today)
    first = dt.datetime.combine(start, dt.time(), dt.timezone.utc).timestamp()
    per_day: dict[str, float] = {}
    per_model: dict[str, dict] = {}
    estimated = 0
    for record in records:
        source = record.get("source") or {}
        engine = source.get("engine")
        created = float(record.get("created_at") or 0)
        if engine not in GOOGLE_ENGINES or created < first:
            continue
        cost = source.get("cost_usd")
        if not isinstance(cost, (int, float)):
            cost, estimated = google_images.price(engine, "1K"), estimated + 1
        date = dt.datetime.fromtimestamp(created, dt.timezone.utc).date().isoformat()
        per_day[date] = per_day.get(date, 0.0) + cost
        name = source.get("generator") or engine
        entry = per_model.setdefault(name, {"model": name, "type": "image", "engine": engine, "amount": 0.0,
                                            "images": 0, "requests": 0})
        entry["amount"] += cost
        entry["images"] += 1
        entry["requests"] += 1
    notes = ["Google has no API for the Gemini prepaid balance or invoices, so these are this server's own "
             "estimates from list prices. The exact figures are in AI Studio billing."]
    if estimated:
        notes.append(f"{estimated} image{'s' if estimated != 1 else ''} made before prices were recorded "
                     "counted at the 1K price.")
    return {
        "provider": "google", "configured": configured, "currency": "usd", "period": _period_view(start, end, days),
        "balance": None, "estimated_images": estimated,
        "total": round(sum(per_day.values()), 6), "daily": _daily(per_day, start, end),
        "models": _models(per_model), "prices": google_images.PRICES,
        "notes": notes, "links": CONSOLE["google"],
        **({} if configured else {"message": "No Google key on the server, so Nano Banana is not in use. "
                                             "Earlier images still count below."}),
    }
