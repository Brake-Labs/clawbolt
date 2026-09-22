"""Gmail integration: search, read, and send messages on the user's behalf.

The integration uses Gmail's REST API with two OAuth scopes:

* ``gmail.readonly`` for ``messages.list``, ``messages.get``, and
  ``messages.attachments.get``
* ``gmail.send`` for composing new mail and threaded replies

The factory registers five agent tools (``gmail_search``, ``gmail_get_message``,
``gmail_open_attachment``, ``gmail_list_recent``, ``gmail_send``) all
defaulting to ``ask`` permission so
the user is prompted before any mailbox access or outbound message.
"""
