"""
Example 4: Multi-Agent Orchestration

Demonstrates:
  - SequentialPipeline: chaining agents (Researcher → Writer)
  - CrewAgent: role-based collaboration with a manager

These are the two main multi-agent patterns used in production.
"""

from lightagentx import OpenAILLM, SingleAgent, SequentialPipeline, CrewAgent


def demo_sequential():
    """Sequential Pipeline: Researcher → Writer → Editor"""
    print("\n" + "=" * 60)
    print("📋 Sequential Pipeline Demo")
    print("=" * 60)

    llm = OpenAILLM(model="gpt-4o-mini")

    # Create specialist agents
    researcher = SingleAgent(
        name="Researcher",
        description="Expert at gathering and organizing information on any topic",
        llm=llm,
        system_prompt=(
            "You are a research specialist. When given a topic, provide key facts, "
            "statistics, and insights. Be thorough but concise. Format as bullet points."
        ),
        verbose=True,
    )

    writer = SingleAgent(
        name="Writer",
        description="Expert at turning research into engaging content",
        llm=llm,
        system_prompt=(
            "You are a content writer. Take the research provided and turn it into "
            "a well-written, engaging article. Keep it concise (2-3 paragraphs). "
            "Use a professional but accessible tone."
        ),
        verbose=True,
    )

    # Create the pipeline: Researcher → Writer
    pipeline = SequentialPipeline(
        name="Content Pipeline",
        agents=[researcher, writer],
        verbose=True,
    )

    result = pipeline.run(
        "The impact of large language models on software development"
    )

    print(f"\n{'=' * 60}")
    print("📝 Final Pipeline Output:")
    print("=" * 60)
    print(result)


def demo_crew():
    """Crew: Manager delegates to specialists"""
    print("\n" + "=" * 60)
    print("👥 Crew Agent Demo")
    print("=" * 60)

    llm = OpenAILLM(model="gpt-4o-mini")

    # Create specialist agents
    technical_expert = SingleAgent(
        name="TechExpert",
        description="Expert at explaining technical concepts and architectures",
        llm=llm,
        system_prompt=(
            "You are a technical expert. When given a topic, explain the technical "
            "aspects clearly. Focus on how things work under the hood."
        ),
        verbose=True,
    )

    business_analyst = SingleAgent(
        name="BusinessAnalyst",
        description="Expert at analyzing business impact and market trends",
        llm=llm,
        system_prompt=(
            "You are a business analyst. When given a topic, analyze the business "
            "implications, market opportunities, and potential challenges."
        ),
        verbose=True,
    )

    # Create the crew with a manager
    crew = CrewAgent(
        name="Analysis Team",
        agents=[technical_expert, business_analyst],
        manager_llm=llm,
        verbose=True,
    )

    result = crew.run(
        "Analyze the rise of AI coding assistants and their impact on the software industry"
    )

    print(f"\n{'=' * 60}")
    print("📝 Final Crew Output:")
    print("=" * 60)
    print(result)


def main():
    import sys

    if len(sys.argv) > 1:
        if sys.argv[1] == "sequential":
            demo_sequential()
        elif sys.argv[1] == "crew":
            demo_crew()
        else:
            print(f"Unknown demo: {sys.argv[1]}")
            print("Usage: python 04_multi_agent.py [sequential|crew]")
    else:
        # Run both demos
        demo_sequential()
        demo_crew()


if __name__ == "__main__":
    main()
