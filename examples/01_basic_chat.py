"""
Example 1: Basic Chat with Memory

Demonstrates:
  - Creating an OpenAI LLM instance
  - Using BufferMemory for conversation history
  - Simple multi-turn conversation

This is the simplest possible usage — no tools, no agents, just LLM + memory.
"""

from lightagentx import OpenAILLM, BufferMemory


def main():
    # Create the LLM (uses OPENAI_API_KEY from environment)
    llm = OpenAILLM(model="gpt-4o-mini", temperature=0.7)

    # Create memory to store conversation history
    memory = BufferMemory(max_messages=20)
    memory.add_message("system", "You are a friendly and helpful AI assistant.")

    print("LightAgentX Basic Chat")
    print("Type 'quit' to exit, 'clear' to reset memory\n")

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() == "quit":
            print("Goodbye!")
            break
        if user_input.lower() == "clear":
            memory.clear()
            memory.add_message("system", "You are a friendly and helpful AI assistant.")
            print("Memory cleared!\n")
            continue

        # Add user message to memory
        memory.add_message("user", user_input)

        # Get all messages and send to LLM
        messages = memory.get_messages()
        response = llm.chat(messages)

        # Add assistant response to memory
        memory.add_message("assistant", response.content)

        print(f"\nAssistant: {response.content}\n")


if __name__ == "__main__":
    main()
