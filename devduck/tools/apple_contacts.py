"""👤 Apple Contacts via Contacts framework."""

from typing import Dict, Any
from strands import tool


@tool
def apple_contacts(
    action: str = "search",
    query: str = "",
    limit: int = 20,
) -> Dict[str, Any]:
    """👤 Apple Contacts — search, list, and read contacts.

    Args:
        action: Action to perform:
            - "search": Search contacts by name/email/phone
            - "list": List all contacts (with limit)
            - "groups": List contact groups
            - "stats": Contact database statistics
        query: Search query
        limit: Max results (default: 20)

    Returns:
        Dict with contact data
    """
    try:
        import Contacts
    except ImportError:
        return {"status": "error", "content": [{"text": "Install: pip install pyobjc-framework-Contacts"}]}

    store = Contacts.CNContactStore.alloc().init()

    if action == "search":
        return _search_contacts(store, query, limit)
    elif action == "list":
        return _list_contacts(store, limit)
    elif action == "groups":
        return _list_groups(store)
    elif action == "stats":
        return _contact_stats(store)
    else:
        return {"status": "error", "content": [{"text": f"Unknown action: {action}. Use: search, list, groups, stats"}]}


def _format_contact(contact):
    """Format a single contact for display."""
    import Contacts

    lines = []
    name_parts = []
    if contact.givenName():
        name_parts.append(str(contact.givenName()))
    if contact.familyName():
        name_parts.append(str(contact.familyName()))
    name = " ".join(name_parts) or "(No Name)"

    org = str(contact.organizationName()) if contact.organizationName() else ""
    lines.append(f"  👤 {name}" + (f" — {org}" if org else ""))

    # Phones
    phones = contact.phoneNumbers()
    if phones:
        for p in phones:
            label = str(p.label() or "").replace("_$!<", "").replace(">!$_", "")
            val = str(p.value().stringValue())
            lines.append(f"     📞 {val} [{label}]")

    # Emails
    emails = contact.emailAddresses()
    if emails:
        for e in emails:
            lines.append(f"     ✉️  {e.value()}")

    return "\n".join(lines)


def _get_keys():
    """Get the standard keys to fetch."""
    import Contacts
    return [
        Contacts.CNContactGivenNameKey,
        Contacts.CNContactFamilyNameKey,
        Contacts.CNContactOrganizationNameKey,
        Contacts.CNContactPhoneNumbersKey,
        Contacts.CNContactEmailAddressesKey,
    ]


def _search_contacts(store, query, limit):
    """Search contacts by name."""
    try:
        import Contacts

        if not query:
            return {"status": "error", "content": [{"text": "query required for search"}]}

        keys = _get_keys()
        predicate = Contacts.CNContact.predicateForContactsMatchingName_(query)

        error = None
        contacts = store.unifiedContactsMatchingPredicate_keysToFetch_error_(
            predicate, keys, None
        )

        if isinstance(contacts, tuple):
            contacts, error = contacts

        if error:
            return {"status": "error", "content": [{"text": f"Search error: {error}"}]}

        if not contacts:
            return {"status": "success", "content": [{"text": f"🔍 No contacts matching '{query}'."}]}

        lines = [f"🔍 Found {len(contacts)} contact(s) matching '{query}':\n"]
        for c in contacts[:limit]:
            lines.append(_format_contact(c))

        return {"status": "success", "content": [{"text": "\n".join(lines)}]}

    except Exception as e:
        return {"status": "error", "content": [{"text": f"Search error: {e}"}]}


def _list_contacts(store, limit):
    """List contacts."""
    try:
        import Contacts

        keys = _get_keys()
        request = Contacts.CNContactFetchRequest.alloc().initWithKeysToFetch_(keys)
        request.setSortOrder_(Contacts.CNContactSortOrderGivenName)

        contacts = []
        error_holder = [None]

        def handler(contact, stop):
            if len(contacts) >= limit:
                stop[0] = True
                return
            contacts.append(contact)

        store.enumerateContactsWithFetchRequest_error_usingBlock_(
            request, None, handler
        )

        if not contacts:
            return {"status": "success", "content": [{"text": "📇 No contacts found."}]}

        lines = [f"📇 Contacts (showing {len(contacts)}):\n"]
        for c in contacts:
            lines.append(_format_contact(c))

        return {"status": "success", "content": [{"text": "\n".join(lines)}]}

    except Exception as e:
        return {"status": "error", "content": [{"text": f"List error: {e}"}]}


def _list_groups(store):
    """List contact groups."""
    try:
        import Contacts

        groups, error = store.groupsMatchingPredicate_error_(None, None)

        if not groups:
            return {"status": "success", "content": [{"text": "👥 No contact groups found."}]}

        lines = [f"👥 Contact Groups ({len(groups)}):\n"]
        for g in groups:
            lines.append(f"  • {g.name()}")

        return {"status": "success", "content": [{"text": "\n".join(lines)}]}

    except Exception as e:
        return {"status": "error", "content": [{"text": f"Groups error: {e}"}]}


def _contact_stats(store):
    """Get contact database statistics."""
    try:
        import Contacts

        keys = [Contacts.CNContactGivenNameKey, Contacts.CNContactFamilyNameKey]
        request = Contacts.CNContactFetchRequest.alloc().initWithKeysToFetch_(keys)

        count = [0]
        def handler(contact, stop):
            count[0] += 1

        store.enumerateContactsWithFetchRequest_error_usingBlock_(
            request, None, handler
        )

        groups, _ = store.groupsMatchingPredicate_error_(None, None)
        group_count = len(groups) if groups else 0

        lines = [
            "📊 Contacts Statistics:",
            f"  Total contacts: {count[0]}",
            f"  Groups: {group_count}",
        ]

        return {"status": "success", "content": [{"text": "\n".join(lines)}]}

    except Exception as e:
        return {"status": "error", "content": [{"text": f"Stats error: {e}"}]}
