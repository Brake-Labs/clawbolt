"""Realistic synthetic email bodies for the Gmail body-trim tests.

Every name, address, and number here is made up. Each fixture pairs the text
that is the message's own new content (``KEEP``) with text that should be
trimmed away (``DROP``), so tests can assert on both.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Gmail reply with a wrapped "On ... wrote:" header and a nested quote
# ---------------------------------------------------------------------------

GMAIL_REPLY_KEEP = (
    "Hi Bob,\n"
    "\n"
    "Thursday at 8am works. Please bring the revised quote for the second\n"
    "bathroom, and use the matte black fixtures we picked last week.\n"
    "\n"
    "Thanks,\n"
    "Jane"
)
GMAIL_REPLY_DROP = "> Can we start on Thursday instead of Friday?"
GMAIL_REPLY = (
    GMAIL_REPLY_KEEP
    + "\n\n"
    + "On Mon, Sep 28, 2026 at 9:14 AM Bob Builder <bob@acmeplumbing.example.com>\n"
    "wrote:\n"
    "\n"
    "> Hi Jane,\n"
    ">\n"
    "> Can we start on Thursday instead of Friday? The tile delivery moved up a\n"
    "> day, and I would rather have the crew on site when it lands. I have also\n"
    "> attached the revised quote for the second bathroom, which now includes\n"
    "> the shower valve replacement and the upgraded exhaust fan you asked\n"
    "> about. The total went from $4,200 to $4,850.\n"
    ">\n"
    "> Let me know what works.\n"
    ">\n"
    "> Bob Builder\n"
    "> Acme Plumbing | +1 555 555 0123\n"
    "> https://acmeplumbing.example.com\n"
    ">\n"
    "> On Fri, Sep 25, 2026 at 4:02 PM Jane Doe <jane.doe@example.com> wrote:\n"
    ">\n"
    ">> Hi Bob,\n"
    ">>\n"
    ">> Thanks for coming out yesterday. We would like to go ahead with the\n"
    ">> main bathroom and get a quote for the second one as well. We picked\n"
    ">> the matte black fixtures from the catalog you left. Our only\n"
    ">> constraint is that the work has to be finished before October 20,\n"
    ">> because family is visiting.\n"
    ">>\n"
    ">> Jane\n"
    ">>\n"
    ">> On Thu, Sep 24, 2026 at 11:30 AM Bob Builder <bob@acmeplumbing.example.com>\n"
    ">> wrote:\n"
    ">>\n"
    ">>> Hi Jane, following up on the site visit. Here is what we saw: the\n"
    ">>> supply lines under the vanity are original galvanized and should be\n"
    ">>> replaced, the toilet flange is cracked, and the shower valve is a\n"
    ">>> single-handle cartridge that is no longer made. None of this is\n"
    ">>> urgent, but it is all easier to do while the walls are open.\n"
    ">>>\n"
    ">>> Bob\n"
)

# ---------------------------------------------------------------------------
# Outlook reply: "____" rule, then a From:/Sent:/To:/Subject: header block
# ---------------------------------------------------------------------------

OUTLOOK_REPLY_KEEP = (
    "Hi Bob,\n"
    "\n"
    "Approved. Please schedule the panel upgrade for the week of October 5 and\n"
    "send the invoice to accounts@example.com when the permit is pulled.\n"
    "\n"
    "Pat Manager\n"
    "Facilities | Example Property Management\n"
    "555-555-0142"
)
OUTLOOK_REPLY_DROP = "Subject: RE: Panel upgrade at 123 Main St"
OUTLOOK_REPLY = (
    OUTLOOK_REPLY_KEEP + "\n\n" + "________________________________\n"
    "From: Bob Builder <bob@acmeplumbing.example.com>\n"
    "Sent: Monday, September 28, 2026 8:02 AM\n"
    "To: Pat Manager <pat.manager@example.com>\n"
    "Cc: accounts@example.com\n"
    "Subject: RE: Panel upgrade at 123 Main St\n"
    "\n"
    "Hi Pat,\n"
    "\n"
    "The utility confirmed they can do the disconnect on either October 6 or\n"
    "October 8. The upgrade itself is one day of work, and the tenant in unit\n"
    "2 will be without power for roughly six hours. The price from my earlier\n"
    "estimate stands: $3,900 including permit and inspection.\n"
    "\n"
    "Bob Builder\n"
    "Acme Plumbing and Electric\n"
    "\n"
    "________________________________\n"
    "From: Pat Manager <pat.manager@example.com>\n"
    "Sent: Friday, September 25, 2026 3:47 PM\n"
    "To: Bob Builder <bob@acmeplumbing.example.com>\n"
    "Subject: Panel upgrade at 123 Main St\n"
    "\n"
    "Bob, the owner signed off on the 200A upgrade. What dates can you do,\n"
    "and does the estimate from August still hold? We need to give the\n"
    "tenant at least 48 hours notice of the outage.\n"
    "\n"
    "Pat\n"
    "\n"
    "CONFIDENTIALITY NOTICE: This e-mail message, including any attachments,\n"
    "is for the sole use of the intended recipient(s) and may contain\n"
    "confidential and privileged information. Any unauthorized review, use,\n"
    "disclosure or distribution is prohibited.\n"
)

# ---------------------------------------------------------------------------
# Inspector report with a "-- " signature delimiter and a legal notice
# ---------------------------------------------------------------------------

SIGNATURE_KEEP = (
    "Hello,\n"
    "\n"
    "Rough-in inspection for permit BLD-2026-04417 at 123 Main St: PASSED with\n"
    "two corrections to complete before final.\n"
    "\n"
    "1. Add a nail plate where the drain line crosses the stud in the north\n"
    "   bathroom wall.\n"
    "2. Label the new 20A kitchen circuits at the panel.\n"
    "\n"
    "You can schedule the final inspection online once both are done."
)
SIGNATURE_DROP = "Senior Building Inspector"
SIGNATURE = (
    SIGNATURE_KEEP + "\n\n" + "-- \n"
    "Sam Inspector\n"
    "Senior Building Inspector\n"
    "City of Example | Building and Safety Division\n"
    "100 Civic Center Plaza, Example City, CA 90000\n"
    "Office: 555-555-0199 | Inspection line: 555-555-0100\n"
    "https://example.gov/building\n"
    "\n"
    "This message and any attachments are intended solely for the use of the\n"
    "addressee and may contain information that is privileged and\n"
    "confidential. If you have received this message in error, please notify\n"
    "the sender and delete it. Emails to and from the City may be subject to\n"
    "public records disclosure.\n"
)

# ---------------------------------------------------------------------------
# Bank payment alert: short content, long legal and unsubscribe footer
# ---------------------------------------------------------------------------

PAYMENT_ALERT_KEEP = (
    "Payment received\n"
    "\n"
    "A payment of $1,250.00 was deposited to your business checking account\n"
    "ending in 4821 on September 28, 2026.\n"
    "\n"
    "From: Example Property Management LLC\n"
    "Memo: Invoice 1042, 123 Main St water heater\n"
    "\n"
    "View this transaction in online banking: https://bank.example.com/activity"
)
PAYMENT_ALERT_DROP = "Member FDIC"
PAYMENT_ALERT = (
    PAYMENT_ALERT_KEEP
    + "\n\n"
    + "This is an automated message. Please do not reply to this email; the\n"
    "mailbox is not monitored. If you have questions about this transaction,\n"
    "sign in to online banking or call the number on the back of your card.\n"
    "\n"
    "You are receiving this alert because you enrolled in deposit\n"
    "notifications for this account. To change which alerts you receive,\n"
    "manage your alert preferences in online banking under Settings >\n"
    "Alerts. Alerts are sent for your convenience and do not replace your\n"
    "account statement.\n"
    "\n"
    "Protect yourself from fraud: Example Bank will never ask you for your\n"
    "password, PIN, or one-time passcode by email or text. Learn more at\n"
    "https://bank.example.com/security. Unsubscribe from marketing email:\n"
    "https://bank.example.com/unsubscribe?u=abc123\n"
    "\n"
    "Privacy Policy: https://bank.example.com/privacy | Terms of Use:\n"
    "https://bank.example.com/terms\n"
    "\n"
    "Example Bank, N.A. Member FDIC. Equal Housing Lender. NMLS ID 000000.\n"
    "© 2026 Example Bank Corporation. All rights reserved. 1 Example Plaza,\n"
    "Example City, CA 90000.\n"
)

FIXTURES: dict[str, tuple[str, str, str]] = {
    "gmail_reply": (GMAIL_REPLY, GMAIL_REPLY_KEEP, GMAIL_REPLY_DROP),
    "outlook_reply": (OUTLOOK_REPLY, OUTLOOK_REPLY_KEEP, OUTLOOK_REPLY_DROP),
    "signature": (SIGNATURE, SIGNATURE_KEEP, SIGNATURE_DROP),
    "payment_alert": (PAYMENT_ALERT, PAYMENT_ALERT_KEEP, PAYMENT_ALERT_DROP),
}
