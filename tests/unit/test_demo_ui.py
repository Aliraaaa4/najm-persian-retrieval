"""Tests for the dependency-free browser demo UI."""

from __future__ import annotations

from fastapi.testclient import (
    TestClient,
)

from najm_retrieval.api.app import (
    create_app,
)


def _client(
) -> TestClient:
    application = create_app(
        runtime_loader=None,
        suggestion_loader=None,
    )

    return TestClient(
        application
    )


def test_root_serves_persian_demo_ui(
) -> None:
    with _client() as client:
        response = client.get(
            "/"
        )

    assert response.status_code == 200

    assert (
        response.headers[
            "content-type"
        ].startswith(
            "text/html"
        )
    )

    assert (
        'lang="fa"'
        in response.text
    )

    assert (
        'dir="rtl"'
        in response.text
    )

    assert (
        "بازیابی قابل اعتماد"
        in response.text
    )

    assert (
        "/static/styles.css"
        in response.text
    )

    assert (
        "/static/app.js"
        in response.text
    )

    assert (
        "/static/najm-logo.png"
        in response.text
    )


def test_demo_static_assets_are_served(
) -> None:
    with _client() as client:
        styles = client.get(
            "/static/styles.css"
        )

        script = client.get(
            "/static/app.js"
        )

        logo = client.get(
            "/static/najm-logo.png"
        )

    assert styles.status_code == 200
    assert script.status_code == 200
    assert logo.status_code == 200

    assert (
        logo.headers[
            "content-type"
        ].startswith(
            "image/png"
        )
    )

    assert (
        styles.headers[
            "content-type"
        ].startswith(
            "text/css"
        )
    )

    assert (
        "application/javascript"
        in script.headers[
            "content-type"
        ]
        or (
            script.headers[
                "content-type"
            ].startswith(
                "text/javascript"
            )
        )
    )

    assert (
        ".suggestion-button"
        in styles.text
    )

    assert (
        'fetch("/v1/retrieve"'
        in script.text
    )

    assert (
        "textContent"
        in script.text
    )


def test_demo_route_is_not_part_of_public_api_schema(
) -> None:
    with _client() as client:
        response = client.get(
            "/openapi.json"
        )

    assert response.status_code == 200

    paths = response.json()[
        "paths"
    ]

    assert "/" not in paths

    assert (
        "/v1/retrieve"
        in paths
    )
