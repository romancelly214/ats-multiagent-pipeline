"""CrewAI-powered job discovery and candidate-fit analysis."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find relevant job listings and analyze how well they match a candidate profile."
    )
    parser.add_argument(
        "--query",
        required=True,
        help='Job titles, skills, or search terms (for example: "Python data engineer").',
    )
    profile_group = parser.add_mutually_exclusive_group(required=True)
    profile_group.add_argument(
        "--profile",
        help="A short candidate summary including relevant skills and experience.",
    )
    profile_group.add_argument(
        "--profile-file",
        type=Path,
        help="Path to a UTF-8 text file containing the candidate profile.",
    )
    parser.add_argument(
        "--location",
        default="anywhere",
        help='Preferred location (default: "anywhere").',
    )
    parser.add_argument(
        "--work-mode",
        choices=("any", "remote", "hybrid", "onsite"),
        default="any",
        help="Preferred work arrangement (default: any).",
    )
    parser.add_argument(
        "--max-results",
        type=int,
        default=10,
        help="Maximum number of distinct opportunities to include (1-25, default: 10).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path for saving the final report as Markdown.",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("CREWAI_MODEL", "gpt-4o-mini"),
        help="CrewAI language model (default: CREWAI_MODEL or gpt-4o-mini).",
    )
    args = parser.parse_args()

    if not 1 <= args.max_results <= 25:
        parser.error("--max-results must be between 1 and 25.")

    if args.profile_file is not None:
        try:
            args.profile = args.profile_file.read_text(encoding="utf-8").strip()
        except OSError as exc:
            parser.error(f"Could not read profile file: {exc}")

    if not args.profile:
        parser.error("The candidate profile must not be empty.")

    return args


def build_crew(model: str) -> Any:
    try:
        from crewai import Agent, Crew, Process, Task
        from crewai_tools import SerperDevTool
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "CrewAI dependencies are missing. Install them with "
            "`pip install crewai crewai-tools`."
        ) from exc

    search_tool = SerperDevTool()

    researcher = Agent(
        role="Job Market Researcher",
        goal="Find current, verifiable job openings that match the candidate's search criteria.",
        backstory=(
            "You are a careful job-market researcher. You use web search to find "
            "real openings and preserve the source URL for every listing."
        ),
        tools=[search_tool],
        llm=model,
        verbose=False,
    )
    analyst = Agent(
        role="Candidate Fit Analyst",
        goal="Compare job requirements with the candidate profile and rank opportunities fairly.",
        backstory=(
            "You assess evidence in the candidate profile against stated job "
            "requirements. You distinguish demonstrated strengths from gaps "
            "and never invent qualifications."
        ),
        llm=model,
        verbose=False,
    )
    strategist = Agent(
        role="Job Search Strategist",
        goal="Turn the researched and ranked openings into a practical application shortlist.",
        backstory=(
            "You give concise, actionable job-search guidance. You cite listing "
            "URLs and tailor next steps to the candidate's actual experience."
        ),
        llm=model,
        verbose=False,
    )

    research_task = Task(
        description=(
            "Search the web for up to {max_results} distinct, currently advertised "
            "job openings matching these terms: {query}. Preferred location: "
            "{location}. Preferred work mode: {work_mode}. Prioritize employer "
            "career pages and reputable job boards. For each opening, record the "
            "job title, company, location/work mode, key requirements, source URL, "
            "and any available posting date. Exclude duplicates and do not present "
            "an opening as verified if its source cannot be found."
        ),
        expected_output=(
            "A numbered list of up to {max_results} distinct openings with title, "
            "company, location/work mode, key requirements, posting date when "
            "available, and a direct source URL for each."
        ),
        agent=researcher,
    )
    analysis_task = Task(
        description=(
            "Evaluate each researched opening using this candidate profile: "
            "{profile}. Compare only evidence in the profile with the listed "
            "requirements. Assign a fit score from 1 to 5, explain the strongest "
            "matches, identify important gaps, and suggest one honest resume or "
            "application emphasis. Do not infer skills or experience the profile "
            "does not state."
        ),
        expected_output=(
            "The openings ranked by fit, each with its source URL, a 1-5 fit score, "
            "matching evidence, gaps or unknowns, and a tailored application emphasis."
        ),
        agent=analyst,
        context=[research_task],
    )
    report_task = Task(
        description=(
            "Prepare a concise Markdown job-search report from the researched "
            "openings and fit analysis. Include an executive summary, a ranked "
            "shortlist with source links and fit scores, key gaps to address, and "
            "practical next steps. Clearly label missing or uncertain information; "
            "do not claim that any application has been submitted."
        ),
        expected_output=(
            "A readable Markdown report with a ranked, source-linked shortlist, "
            "transparent fit reasoning, and actionable next steps."
        ),
        agent=strategist,
        context=[research_task, analysis_task],
    )

    return Crew(
        agents=[researcher, analyst, strategist],
        tasks=[research_task, analysis_task, report_task],
        process=Process.sequential,
        verbose=False,
    )


def main() -> int:
    args = parse_args()
    if not os.environ.get("SERPER_API_KEY"):
        print(
            "SERPER_API_KEY is required for web search. Add your Serper API key "
            "to the environment and try again.",
            file=sys.stderr,
        )
        return 2

    try:
        crew = build_crew(args.model)
        result = crew.kickoff(
            inputs={
                "query": args.query,
                "profile": args.profile,
                "location": args.location,
                "work_mode": args.work_mode,
                "max_results": args.max_results,
            }
        )
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    report = str(result).strip()
    print(report)

    if args.output is not None:
        args.output.write_text(report + "\n", encoding="utf-8")
        print(f"\nReport saved to {args.output}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
