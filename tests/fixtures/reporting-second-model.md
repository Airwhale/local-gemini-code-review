## File: app.py

### L15: [HIGH] File handle leaks on exceptions

The file is never closed when processing raises an exception.

### L82: [HIGH] Empty input crashes parser

Check for an empty input before reading the first item.
