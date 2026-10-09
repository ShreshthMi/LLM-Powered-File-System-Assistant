# LLM-Powered File System Assistant

Ask questions about resume files in plain English. Claude chooses file tools, the Python program runs them locally, and Claude answers from the results. Each tool call and its outcome are visible in the terminal.

Built for the Airtribe assignment: **Part A** provides four file tools; **Part B** connects them to Anthropic through a manual tool-use loop. The default model is `claude-opus-5-5`. An Anthropic API key with API access is required; requests use paid API credits.

[Demo video](demo/demo.mp4) · [Demo transcript and narration](demo/narration.md) · [Validation report](docs/validation.md)

## Project structure

```text
.
├── fs_tools.py                  # read, list, write and search, inside one sandbox
├── extractors.py                # PDF, DOCX and text extraction
├── llm_file_assistant.py         # Anthropic tool-use loop and CLI
├── resumes/                     # 8 fictional resumes: 3 PDF, 3 DOCX, 2 TXT
├── scripts/generate_resumes.py   # reproducible sample generator
├── tests/                       # unit tests and actual-CLI HTTP integration tests
├── examples/                    # recorded outputs and an example summary
├── docs/validation.md           # automated and live validation results
├── demo/                        # narrated MP4, captions and narration
├── requirements.txt
├── pytest.ini
└── .env.example
```

`summaries/` is created by the assistant when you request a summary. It is ignored by Git. The committed example lives in `examples/`, so a fresh clone can generate the default summary without conflicting with overwrite protection.

## Setup

Python 3.10 or newer is required; development and tests used Python 3.14.7.

```bash
git clone https://github.com/ShreshthMi/LLM-Powered-File-System-Assistant.git
cd LLM-Powered-File-System-Assistant
python3 -m venv .venv
source .venv/bin/activate                 # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

Open `.env` and put your API key after `ANTHROPIC_API_KEY=`. Get a key through the [Anthropic Console](https://console.anthropic.com/). Keep `.env` local; Git ignores it, and the file tools refuse hidden files.

| Setting | Default | Purpose |
|---|---|---|
| `ANTHROPIC_API_KEY` | required | Anthropic API credential |
| `ANTHROPIC_MODEL` | `claude-opus-5-5` | Model ID |
| `ANTHROPIC_WORKSPACE_ID` | optional | Required for keys that can access multiple workspaces |
| `ANTHROPIC_EFFORT` | `medium` | `low`, `medium`, `high`, `xhigh` or `max`; lower effort helps shorten demos |
| `FS_ASSISTANT_ROOT` | project folder | The folder the tools can access |

If the API says your key is not scoped to a workspace, add `ANTHROPIC_WORKSPACE_ID=wrkspc_...` to `.env`. Find the ID in the Console workspace settings.

Real environment variables take precedence over `.env`, except blank values. The SDK also supports `ANTHROPIC_BASE_URL`, which the offline integration tests use to reach a local test server.

## Usage

```bash
python llm_file_assistant.py
python llm_file_assistant.py "Read all resumes in the resumes folder"
python llm_file_assistant.py "Find resumes mentioning Python experience"
python llm_file_assistant.py "Create a summary file for resume_john_doe.pdf"
python llm_file_assistant.py --quiet "List the resumes"
```

Chat retains history, so follow-ups work. Type `/new` to clear history; `exit`, `quit` or Ctrl-D ends the chat. Ctrl-C cancels the current request; a single-question run exits with status 130 when interrupted. `--quiet` hides tool logs.

### Recorded examples

The committed [example transcripts](examples/) contain **live Claude Opus 5.5 recordings**, using `ANTHROPIC_EFFORT=low` and freshly generated fictional resumes. File operations actually executed. The video replays excerpts of these captured outputs with waiting time shortened; it uses synthesized narration. See the [validation report](docs/validation.md) for all 15 sessions.

Abridged live Python search log:

```text
🔧 list_files({"directory": "resumes"})
   → 8 files
🔧 search_in_file({"filepath": "resumes/resume_aisha_bello.docx", "keyword": "Python"})
   → 0 matches
🔧 search_in_file({"filepath": "resumes/resume_david_kim.txt", "keyword": "Python"})
... search_in_file runs on each of the other six resumes ...
```

Claude identified Doe, Kim, Sharma and Rodriguez as having professional Python experience, and explicitly excluded Sofia Novak's Monty Python hobby. Read the [full Python transcript](examples/02_python.txt).

Live summary flow:

```text
🔧 read_file({"filepath": "resumes/resume_john_doe.pdf"})
   → read 1,215 characters, 1 page
🔧 write_file({"filepath": "summaries/resume_john_doe_summary.md", "content": "# John Doe — Resume Summary\n\n**Source file:** resumes/resume_john_doe.pdf\n**Curr… (1,489 chars)"})
   → wrote 1,507 bytes to summaries/resume_john_doe_summary.md
```

Read the [full summary transcript](examples/03_summary.txt) and [generated summary](examples/resume_john_doe_summary.md).

## Sample resumes

Every person and company in the fixtures is fictional. Contacts use `example.com` and 555-01xx numbers. Regenerate the samples with:

```bash
python scripts/generate_resumes.py
```

| Resume | Role | Planted case |
|---|---|---|
| `resume_john_doe.pdf` | Backend engineer | Django/Python; default summary target |
| `resume_priya_sharma.pdf` | Data scientist | Lower-case python; two pages |
| `resume_marcus_chen.pdf` | Java developer | Professional Java experience |
| `resume_john_smith.docx` | Frontend engineer | JavaScript; second John for ambiguity |
| `resume_elena_rodriguez.docx` | DevOps / SRE | PYTHON; email in header, skills in table |
| `resume_aisha_bello.docx` | Product manager | Mentions a Java team in passing |
| `resume_david_kim.txt` | ML engineer | Python/NumPy |
| `resume_sofia_novak.txt` | Mobile engineer | Monty Python hobby; Kotlin experience |

## Tool reference

All paths are relative to the sandbox root. Absolute paths inside that root are accepted. Outside paths and hidden path components are refused. Expected failures return data instead of raising to the agent.

### `read_file(filepath: str) -> dict`

Extracts PDF text with pypdf; DOCX body text, headers, footers, tables, text boxes and content controls with python-docx; and TXT, MD, CSV or JSON text. File signatures detect mislabelled formats. Text decoding handles BOMs and a Windows-1252 fallback.

Returns `success`, `filepath`, `content`, `metadata`, `warning` and `error`. Metadata includes `name`, `extension`, `size_bytes`, ISO-8601 `modified`, `char_count` and `word_count`; PDFs also include `page_count`. Content and metadata are null on failure.

Scanned PDFs return empty text with a warning; empty text files return a warning, while empty PDF/DOCX files are errors. Corrupt files and password-protected documents get explanatory errors. Permission-only PDFs that open with an empty password can be extracted.

### `list_files(directory: str, extension: str = None) -> list`

Returns sorted file entries with `name`, `path`, `extension`, `size_bytes` and `modified`. It skips folders, hidden files and Office lock files. The filter accepts `.pdf`, `pdf`, `*.PDF` or several extensions such as `.pdf,.docx`; empty/wildcard filters and `all`, `any`, `none` or `null` mean no filter. It is not recursive.

Errors use a one-element list: `[{"error": "..."}]`. Misspelled paths can receive a “Did you mean …?” hint.

### `write_file(filepath: str, content: str) -> dict`

Writes UTF-8 TXT, MD, CSV or JSON atomically and creates missing parent directories. Returns `success`, `filepath`, `bytes_written`, `overwritten`, `warning` and `error`.

It replaces an existing file only when the assistant created it. A hidden `.fs_tools_written.json` record persists that ownership between sessions. Files you authored, resumes, read-only targets and disallowed formats are protected; choose a new destination instead.

### `search_in_file(filepath: str, keyword: str, whole_word: bool = False) -> dict`

Reuses `read_file`, then searches case-insensitively for a literal keyword. Regex characters such as `C++`, `C#` and `.NET` stay literal. Phrase whitespace can match line breaks. Whole-word mode separates Java from JavaScript.

Returns `success`, `filepath`, `keyword`, `match_count`, `matches`, `truncated`, `warning` and `error`. Each match contains `position`, `line`, `match` and surrounding `context`. At most 50 matches are returned, while `match_count` reports the full total. Empty keywords are rejected.

## The manual tool-use loop

Each tool has a name, detailed description and JSON `input_schema`, with `strict: true` and `additionalProperties: false`. Claude chooses tools automatically; the program does not force tool selection. This follows Anthropic's [strict tool-use contract](https://platform.claude.com/docs/en/agents-and-tools/tool-use/strict-tool-use).

```mermaid
sequenceDiagram
    participant U as User
    participant A as Python CLI
    participant C as Anthropic Messages API
    participant T as Local file tools
    U->>A: Question
    loop Until answer, maximum 15 requests per turn
        A->>C: System prompt, schemas and conversation
        C-->>A: Content blocks and stop reason
        alt Tool calls
            A->>T: Run each complete tool_use block
            T-->>A: Structured results
            A->>A: Log calls; append assistant blocks unchanged
            A->>A: Append all tool_result blocks in one user message
        else Final text
            A-->>U: Answer
        end
    end
```

The entire assistant content list, including signed thinking blocks, is preserved. Earlier conversation turns are never rewritten. Complete calls run only after a `tool_use` stop reason; calls cut off by `max_tokens` receive error results without executing. Results over 20,000 characters are shortened and labelled.

Authentication failures, connection errors and API errors are explained without a traceback. The SDK retries transient failures up to twice, with a 600-second request timeout. Refusals and API failures discard the new conversation turn; file operations already performed remain on disk. Empty replies and round-limit outcomes retain completed tool results so the user can continue. Partial answers are shown with a warning and a nonzero single-shot exit status.

## Testing and validation

```bash
pytest
pytest --cov=fs_tools --cov=extractors --cov=llm_file_assistant --cov-report=term-missing
```

The verified suite has **257 passing tests** and **94% overall line coverage** (98% for `llm_file_assistant.py`). Tests need no API key or paid model calls. HTTP integration tests start a local server, so the environment must permit localhost sockets.

Tests cover document extraction, malformed files, sandbox escapes and symlink races, overwrite protection, searches, the three assignment flows, parallel tool results, preserved thinking blocks, API error responses, interrupted writes, conversation history, `/new`, CLI errors and terminal control characters. Test CLI copies exclude `.env`, and settings are supplied explicitly.

On Windows, probes requiring unavailable symlink privileges or POSIX permission semantics are skipped. On POSIX hosts with directory-descriptor support, paths are opened component by component without following symlinks; the Windows fallback performs path checks without that same race resistance.

All 15 live sessions passed against Claude Opus 5.5 at low effort. The checks confirmed that Python results exclude Sofia's hobby, professional Java results exclude JavaScript and Aisha's incidental team mention, ambiguous John prompts request clarification, irrelevant questions use no tools, and suspicious file instructions are ignored. The [validation report](docs/validation.md) records both automated and live-model evidence.

## Limits

- Scanned PDFs require OCR, which is outside scope.
- Reads are capped at 20 MB, writes at 5 MB, PDFs at 300 pages and roughly 20 seconds of extraction. PDF extraction limits may return partial content with a warning.
- Single tool results sent to Claude are capped at 20,000 characters. Ask targeted searches for details beyond that limit.
- Legacy `.doc`, RTF, ODT and image files are unsupported. Encrypted Word files must be saved as unencrypted DOCX first.
- The model can still make interpretation mistakes. Tool logs make its file operations inspectable.
- Existing files not created by the assistant cannot be overwritten. Keep committed sample outputs in `examples/`.
- Long conversations can exceed API context limits; use `/new` to start fresh.
