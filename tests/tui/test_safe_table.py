"""SafeDataTable: a click on a table with no columns must not crash the app
(Textual's own click handler indexes ordered_columns and raises IndexError),
while clicks on a populated table keep working — header sorting included."""
from __future__ import annotations

import pytest
from textual.app import App, ComposeResult

from toolery.tui.safe_table import SafeDataTable


class _TableApp(App):
    # A real height, like the tabs' tables (height: 1fr): an empty table with
    # auto height has no clickable area and would never reach the bug.
    CSS = "SafeDataTable, DataTable { height: 10; }"

    def __init__(self, with_columns: bool) -> None:
        super().__init__()
        self.with_columns = with_columns
        self.headers_clicked: list[str] = []

    def compose(self) -> ComposeResult:
        yield SafeDataTable(id="t", cursor_type="row")

    def on_mount(self) -> None:
        if self.with_columns:
            tbl = self.query_one(SafeDataTable)
            tbl.add_column("Model", key="model")
            tbl.add_row("m")

    def on_data_table_header_selected(self, event) -> None:
        self.headers_clicked.append(str(event.column_key.value))


@pytest.mark.asyncio
async def test_click_on_a_table_without_columns_does_not_crash():
    app = _TableApp(with_columns=False)
    async with app.run_test() as pilot:
        await pilot.click("#t")                  # where the Profiles crash happened
        await pilot.click("#t", offset=(2, 0))   # header row
        await pilot.pause()
        assert app._exception is None


@pytest.mark.asyncio
async def test_header_click_still_selects_on_a_populated_table():
    app = _TableApp(with_columns=True)
    async with app.run_test() as pilot:
        await pilot.click("#t", offset=(2, 0))
        await pilot.pause()
        assert app.headers_clicked == ["model"]
