# Photos & Files

Send a photo to your assistant and it can describe what it sees. With Google Drive or CompanyCam connected, it can also save the photo where you need it.

## Sending a photo

Send a photo like you'd send it to anyone else:

```
You: [sends photo of kitchen before demo]

Clawbolt: I can see a dated kitchen with dark wood cabinets,
          laminate countertops, and vinyl flooring. Looks like
          a full gut job. I've saved this to your Job Photos
          folder under today's date.
```

Great for before/after documentation, damage records, material identification, and progress tracking.

## PDFs and email attachments

Send a PDF and the assistant reads its text, so you can ask about an inspection report or a quote without retyping it. Scanned PDFs have no text to read; the assistant can still save them.

With Gmail connected, you don't need to forward files. Ask about the email and the assistant opens the attachment itself, after you approve:

```
You: The client emailed me an inspection report. What did it flag?

Clawbolt: The report flags three items: a GFCI outlet missing
          in the garage, a cracked flue liner, and
          a loose handrail on the back steps.
```

## Saving photos

With Google Drive connected, saved files live under a top-level `Clawbolt` folder. Client work can be organized under `Client - Address/photos`; files without clear job context default to `Inbox`. CompanyCam can store photos in its project structure instead.

Without a storage integration, the assistant can still analyze and reply to photos but cannot save them. See [File Cataloging](/docs/features/file-cataloging) for user behavior and [Google Drive Setup](https://github.com/mozilla-ai/clawbolt/blob/main/docs/self-host/google-drive-setup.md) for operator setup.
