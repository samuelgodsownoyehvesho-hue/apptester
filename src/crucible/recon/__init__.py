"""Reconnaissance: explore a target and build the app map."""

from crucible.recon.fetcher import (
    EgressDenied,
    Fetcher,
    FetchResult,
    HttpFetcher,
    StaticFetcher,
)
from crucible.recon.html import FieldSpec, FormSpec, PageFacts, extract_page_facts
from crucible.recon.scout import (
    DEFAULT_MAX_DEPTH,
    DEFAULT_MAX_PAGES,
    AppMapData,
    RobotsPolicy,
    RouteInfo,
    Scout,
    ScoutError,
)

__all__ = [
    "DEFAULT_MAX_DEPTH",
    "DEFAULT_MAX_PAGES",
    "AppMapData",
    "EgressDenied",
    "FetchResult",
    "Fetcher",
    "FieldSpec",
    "FormSpec",
    "HttpFetcher",
    "PageFacts",
    "RobotsPolicy",
    "RouteInfo",
    "Scout",
    "ScoutError",
    "StaticFetcher",
    "extract_page_facts",
]
