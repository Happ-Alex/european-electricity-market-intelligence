from __future__ import annotations

import requests
from requests import Response
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type


class RetryableHttpError(requests.HTTPError):
    """HTTP error that is safe to retry (rate limit or server-side failure)."""


class HttpClient:
    def __init__(self, timeout: int = 60) -> None:
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "european-electricity-market-intelligence/0.1"})

    @retry(
        stop=stop_after_attempt(6),
        wait=wait_exponential(multiplier=1, min=1, max=30),
        retry=retry_if_exception_type((requests.ConnectionError, requests.Timeout, RetryableHttpError)),
        reraise=True,
    )
    def get(self, url: str, *, params: dict | None = None) -> Response:
        response = self.session.get(url, params=params, timeout=self.timeout)
        if response.status_code == 429 or 500 <= response.status_code < 600:
            raise RetryableHttpError(
                f"Retryable HTTP {response.status_code} for {response.url}",
                response=response,
            )
        response.raise_for_status()
        return response
