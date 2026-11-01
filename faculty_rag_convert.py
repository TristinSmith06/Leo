
from pathlib import Path
import re


def convert_faculty_file(input_file, output_file):
    text = Path(input_file).read_text(encoding="utf-8-sig")
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Split entries on one or more blank lines
    entries = re.split(r"\n\s*\n", text.strip())

    formatted_entries = []

    for entry in entries:
        fields = [
            line.strip()
            for line in entry.splitlines()
            if line.strip()
        ]

        if len(fields) < 2:
            print(f"Skipping invalid entry: {fields}")
            continue

        name = fields[0]
        title = fields[1]

        # Identify email addresses and phone numbers
        email = next(
            (field for field in fields[2:] if "@" in field),
            None
        )

        phone_numbers = [
            field for field in fields[2:]
            if "@" not in field
        ]

        # Build the full entry
        sentence = f"{name} holds the title of {title}."

        contacts = []

        if email:
            contacts.append(f"email at {email}")

        for phone in phone_numbers:
            contacts.append(f"phone at {phone}")

        if contacts:
            if len(contacts) == 1:
                contact_text = contacts[0]
            elif len(contacts) == 2:
                contact_text = f"{contacts[0]} or {contacts[1]}"
            else:
                contact_text = (
                    ", ".join(contacts[:-1])
                    + f", or {contacts[-1]}"
                )

            sentence += (
                f" Contact information: They can be reached by "
                f"{contact_text}."
            )

        # First line: hashtag and name
        # Second line: complete RAG-friendly entry
        formatted_entry = f"# {name}\n{sentence}"
        formatted_entries.append(formatted_entry)

    output_text = "\n\n".join(formatted_entries)

    Path(output_file).write_text(
        output_text + "\n" if output_text else "",
        encoding="utf-8"
    )

    print(f"Entries found: {len(entries)}")
    print(f"Entries converted: {len(formatted_entries)}")
    print(f"Output saved to: {Path(output_file).resolve()}")


if __name__ == "__main__":
    convert_faculty_file("faculty_directory.txt", "faculty_rag.txt")
