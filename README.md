# CrewAI Multi-Agent Job Search

A command-line job-search assistant that uses a CrewAI team to discover current openings, compare their requirements with a candidate profile, and produce a ranked, source-linked application shortlist.

## How it works

The script runs three agents in sequence:

1. **Job Market Researcher** searches the web for current openings and captures source links and listing details.
2. **Candidate Fit Analyst** compares each listing with the supplied profile, scores the fit, and calls out evidence and gaps without inventing qualifications.
3. **Job Search Strategist** produces a concise Markdown report with a ranked shortlist and practical next steps.

The research agent uses Serper for web search. CrewAI coordinates the agents and the language model configured for the run.

## Requirements

- Python 3.10 or newer
- A Serper API key for web search
- An API key and provider configuration for the CrewAI language model (OpenAI is used by default)

Install the Python dependencies:

```bash
python -m pip install crewai crewai-tools
```

Set the required API keys in your environment. The model defaults to `gpt-4o-mini`; set `CREWAI_MODEL` to select another CrewAI-supported model.

PowerShell:

```powershell
$env:SERPER_API_KEY = "your-serper-api-key"
$env:OPENAI_API_KEY = "your-openai-api-key"
# Optional:
$env:CREWAI_MODEL = "gpt-4o-mini"
```

macOS/Linux:

```bash
export SERPER_API_KEY="your-serper-api-key"
export OPENAI_API_KEY="your-openai-api-key"
# Optional:
export CREWAI_MODEL="gpt-4o-mini"
```

## Usage

Provide search terms and either a short profile or a UTF-8 profile file:

```bash
python job_search_agent.py \
  --query "Python data engineer" \
  --profile "4 years building Python ETL pipelines; experienced with SQL, Airflow, and AWS" \
  --location "New York, NY" \
  --work-mode hybrid \
  --max-results 8 \
  --output job-search-report.md
```

Or keep the profile in a local text file:

```bash
python job_search_agent.py \
  --query "machine learning engineer" \
  --profile-file candidate-profile.txt \
  --location "Remote" \
  --work-mode remote
```

`--query` and exactly one of `--profile` or `--profile-file` are required. `--location` defaults to `anywhere`, `--work-mode` accepts `any`, `remote`, `hybrid`, or `onsite`, and `--max-results` accepts values from 1 to 25. The report is printed to the terminal and can optionally be saved with `--output`.

## Responsible use and limitations

Search results and job postings can be outdated, incomplete, or inaccurate. Check each source link and confirm the opening on the employer's site before applying. Fit scores are an AI-generated starting point, not a hiring prediction. The script does not submit applications.

Only provide profile information you are comfortable sending to the configured language-model and search providers. Keep API keys in environment variables; do not commit keys or sensitive candidate details to the repository.
