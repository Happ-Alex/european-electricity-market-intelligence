from __future__ import annotations

import requests
from requests import Response
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type


class HttpClient:
    def __init__(self, timeout: int = 60) -> None:
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "european-electricity-market-intelligence/0.1"})

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=1, max=20),
        retry=retry_if_exception_type(requests.RequestException),
        reraise=True,
    )
    def get(self, url: str, *, params: dict | None = None) -> Response:
        response = self.session.get(url, params=params, timeout=self.timeout)
        response.raise_for_status()
        return response
