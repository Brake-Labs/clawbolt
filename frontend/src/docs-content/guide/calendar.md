# Calendar

With Google Calendar connected, you can check your schedule, book jobs, and manage appointments through text.

## Examples

```
You: What do I have tomorrow?

Clawbolt: Here's your schedule for tomorrow (Tuesday):
          - 8:00 AM: Test Customer site visit
          - 11:30 AM: Lunch with supplier
          - 2:00 PM: Estimate walkthrough at 123 Main Street
```

```
You: Schedule a bathroom remodel for Test Customer
     for next Monday 8am to 4pm

Clawbolt: I'd like to create this event:
          "Job: Test Customer - Bathroom remodel"
          Monday March 30, 8:00 AM - 4:00 PM
          Approve?

You: Yes

Clawbolt: Done! Event created on your calendar.
```

Your assistant always asks for approval before creating, changing, or deleting events. Read-only actions (checking schedule, checking availability) happen automatically.

## Timezone

Your assistant uses your timezone when interpreting times. Make sure it's set correctly in the dashboard (User page).

## Reconnecting or switching Google accounts

The calendars you pick belong to the Google account you connected. The Integrations page shows which account that is, under Google Calendar.

Each time you connect or reconnect, Clawbolt checks your picks against the account you just connected:

- Calendars that account can still see stay picked, with your per-calendar settings.
- Calendars it cannot see are removed from your picks. This happens when you reconnect with a different Google account.
- If anything was removed, or you had nothing picked, your main calendar for that account is turned on so the assistant has somewhere to work.

If the assistant says a saved calendar is not visible to your connected account, either reconnect Google Calendar with the account that owns that calendar, or re-pick your calendars on the Integrations page.

## Getting connected

Ask your assistant to connect Google Calendar, or use the Tools page in the web dashboard. See [Integrations](/docs/guide/integrations) for the connection flow and [Google Calendar](/docs/features/calendar) for operator setup.
