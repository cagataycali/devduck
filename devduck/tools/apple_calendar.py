"""📅 Apple Calendar & Reminders via EventKit framework."""

from typing import Dict, Any, Optional
from strands import tool


@tool
def apple_calendar(
    action: str = "today",
    days: int = 7,
    title: str = None,
    notes: str = None,
    calendar_name: str = None,
    query: str = None,
) -> Dict[str, Any]:
    """📅 Apple Calendar & Reminders — read events, create reminders.

    Args:
        action: Action to perform:
            - "today": Today's events
            - "upcoming": Events in the next N days
            - "calendars": List all calendars
            - "reminders": List reminders
            - "create_reminder": Create a new reminder
            - "search": Search events by title
        days: Number of days to look ahead (default: 7)
        title: Title for new reminder
        notes: Notes for new reminder
        calendar_name: Calendar name filter
        query: Search query

    Returns:
        Dict with calendar/reminder data
    """
    try:
        import EventKit
        from Foundation import NSDate, NSCalendar, NSDateComponents
    except ImportError:
        return {"status": "error", "content": [{"text": "Install: pip install pyobjc-framework-EventKit"}]}

    store = EventKit.EKEventStore.alloc().init()

    if action in ("today", "upcoming"):
        return _get_events(store, days if action == "upcoming" else 1, calendar_name)
    elif action == "calendars":
        return _list_calendars(store)
    elif action == "reminders":
        return _get_reminders(store)
    elif action == "create_reminder":
        if not title:
            return {"status": "error", "content": [{"text": "title required for create_reminder"}]}
        return _create_reminder(store, title, notes)
    elif action == "search":
        return _search_events(store, query or title or "", days)
    else:
        return {"status": "error", "content": [{"text": f"Unknown action: {action}. Use: today, upcoming, calendars, reminders, create_reminder, search"}]}


def _get_events(store, days, calendar_name=None):
    """Get events for the next N days."""
    try:
        from Foundation import NSDate, NSCalendar
        import EventKit

        now = NSDate.date()
        cal = NSCalendar.currentCalendar()

        # End date = now + days
        end = NSDate.dateWithTimeIntervalSinceNow_(days * 86400)

        # Get all calendars or filter
        calendars = store.calendarsForEntityType_(EventKit.EKEntityTypeEvent)
        if calendar_name:
            calendars = [c for c in calendars if calendar_name.lower() in c.title().lower()]

        predicate = store.predicateForEventsWithStartDate_endDate_calendars_(now, end, calendars)
        events = store.eventsMatchingPredicate_(predicate)

        if not events:
            return {"status": "success", "content": [{"text": f"📅 No events in the next {days} day(s)."}]}

        # Sort by start date
        events = sorted(events, key=lambda e: e.startDate().timeIntervalSince1970())

        lines = [f"📅 Events ({days} day{'s' if days > 1 else ''}):\n"]
        current_day = None

        for event in events:
            start = event.startDate()
            # Format date
            from datetime import datetime
            dt = datetime.fromtimestamp(start.timeIntervalSince1970())
            day_str = dt.strftime("%A, %b %d")

            if day_str != current_day:
                current_day = day_str
                lines.append(f"\n  📆 {day_str}:")

            time_str = dt.strftime("%H:%M")
            end_dt = datetime.fromtimestamp(event.endDate().timeIntervalSince1970())
            end_str = end_dt.strftime("%H:%M")
            cal_title = event.calendar().title() if event.calendar() else ""
            location = event.location() or ""

            lines.append(f"    {time_str}-{end_str}  {event.title()}")
            if location:
                lines.append(f"               📍 {location}")
            if cal_title:
                lines.append(f"               [{cal_title}]")

        return {"status": "success", "content": [{"text": "\n".join(lines)}]}

    except Exception as e:
        return {"status": "error", "content": [{"text": f"Calendar error: {e}"}]}


def _list_calendars(store):
    """List all calendars."""
    try:
        import EventKit

        lines = ["📅 Calendars:\n"]

        for entity_type, label in [(EventKit.EKEntityTypeEvent, "Event"), (EventKit.EKEntityTypeReminder, "Reminder")]:
            calendars = store.calendarsForEntityType_(entity_type)
            lines.append(f"  {label} Calendars ({len(calendars)}):")
            for cal in calendars:
                source = cal.source().title() if cal.source() else "Local"
                immutable = "🔒" if not cal.allowsContentModifications() else "✏️"
                lines.append(f"    {immutable} {cal.title()} [{source}]")

        return {"status": "success", "content": [{"text": "\n".join(lines)}]}

    except Exception as e:
        return {"status": "error", "content": [{"text": f"List calendars error: {e}"}]}


def _get_reminders(store):
    """Get incomplete reminders."""
    try:
        import EventKit
        from Foundation import NSRunLoop, NSDate, NSDefaultRunLoopMode

        calendars = store.calendarsForEntityType_(EventKit.EKEntityTypeReminder)
        predicate = store.predicateForIncompleteRemindersWithDueDateStarting_ending_calendars_(
            None, None, calendars
        )

        results = []
        done = [False]

        def callback(reminders):
            if reminders:
                results.extend(reminders)
            done[0] = True

        store.fetchRemindersMatchingPredicate_completion_(predicate, callback)

        # Wait for async callback
        import time
        deadline = time.time() + 10
        while not done[0] and time.time() < deadline:
            NSRunLoop.currentRunLoop().runMode_beforeDate_(
                NSDefaultRunLoopMode, NSDate.dateWithTimeIntervalSinceNow_(0.1)
            )

        if not results:
            return {"status": "success", "content": [{"text": "📋 No incomplete reminders."}]}

        lines = [f"📋 Reminders ({len(results)} incomplete):\n"]
        for r in results:
            priority = ["", "!!!", "!!", "!", ""][min(r.priority(), 4)] if r.priority() else ""
            cal_name = r.calendar().title() if r.calendar() else ""
            due = ""
            if r.dueDateComponents():
                dc = r.dueDateComponents()
                if dc.year() and dc.year() < 9999:
                    due = f" (due: {dc.year()}-{dc.month():02d}-{dc.day():02d})"

            lines.append(f"  ☐ {priority}{r.title()}{due} [{cal_name}]")

        return {"status": "success", "content": [{"text": "\n".join(lines)}]}

    except Exception as e:
        return {"status": "error", "content": [{"text": f"Reminders error: {e}"}]}


def _create_reminder(store, title, notes=None):
    """Create a new reminder."""
    try:
        import EventKit

        reminder = EventKit.EKReminder.reminderWithEventStore_(store)
        reminder.setTitle_(title)
        if notes:
            reminder.setNotes_(notes)

        # Use default reminders calendar
        default_cal = store.defaultCalendarForNewReminders()
        if default_cal:
            reminder.setCalendar_(default_cal)

        success, error = store.saveReminder_commit_error_(reminder, True, None)

        if success:
            cal_name = default_cal.title() if default_cal else "Default"
            return {"status": "success", "content": [{"text": f"✅ Reminder created: '{title}' [{cal_name}]"}]}
        else:
            return {"status": "error", "content": [{"text": f"Failed to create reminder: {error}"}]}

    except Exception as e:
        return {"status": "error", "content": [{"text": f"Create reminder error: {e}"}]}


def _search_events(store, query, days=30):
    """Search events by title."""
    try:
        import EventKit
        from Foundation import NSDate

        now = NSDate.date()
        end = NSDate.dateWithTimeIntervalSinceNow_(days * 86400)
        start = NSDate.dateWithTimeIntervalSinceNow_(-days * 86400)

        calendars = store.calendarsForEntityType_(EventKit.EKEntityTypeEvent)
        predicate = store.predicateForEventsWithStartDate_endDate_calendars_(start, end, calendars)
        events = store.eventsMatchingPredicate_(predicate)

        if not events:
            return {"status": "success", "content": [{"text": f"🔍 No events found."}]}

        # Filter by query
        query_lower = query.lower()
        matched = [e for e in events if query_lower in (e.title() or "").lower() or query_lower in (e.location() or "").lower()]

        if not matched:
            return {"status": "success", "content": [{"text": f"🔍 No events matching '{query}' in ±{days} days."}]}

        matched.sort(key=lambda e: e.startDate().timeIntervalSince1970())

        lines = [f"🔍 Found {len(matched)} events matching '{query}':\n"]
        from datetime import datetime
        for event in matched[:20]:
            dt = datetime.fromtimestamp(event.startDate().timeIntervalSince1970())
            lines.append(f"  📆 {dt.strftime('%b %d %H:%M')} — {event.title()}")

        return {"status": "success", "content": [{"text": "\n".join(lines)}]}

    except Exception as e:
        return {"status": "error", "content": [{"text": f"Search error: {e}"}]}
