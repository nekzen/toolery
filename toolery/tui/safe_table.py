from __future__ import annotations

from textual import events
from textual.widgets import DataTable


class SafeDataTable(DataTable):
    """DataTable that ignores clicks while it has no columns.

    Textual's click handler indexes ``ordered_columns`` with the clicked
    column, so a click on a table with zero columns raises IndexError and
    takes the whole app down. Several tabs build their columns lazily or
    clear them (``clear(columns=True)``) while there is nothing to show —
    Profiles before "Apply profile", Rankings with an empty database,
    Scenarios when the filters match nothing — so a stray click there crashed
    the TUI.

    Textual dispatches ``_on_click`` to every class in the MRO, so returning
    early is not enough: ``prevent_default()`` is what stops DataTable's own
    handler from running afterwards.
    """

    async def _on_click(self, event: events.Click) -> None:
        if not self.ordered_columns:
            event.prevent_default()
