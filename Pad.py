from langchain.agents import AgentExecutor
from langchain.schema import AIMessage

class CustomAgentExecutor(AgentExecutor):
    def invoke(self, input, **kwargs):
        # Convert agent_scratchpad to list of BaseMessage if it's a string
        if "agent_scratchpad" in input and isinstance(input["agent_scratchpad"], str):
            logger.warning(f"Converting string agent_scratchpad to AIMessage: {input['agent_scratchpad']}")
            input["agent_scratchpad"] = [AIMessage(content=input["agent_scratchpad"])]
        logger.info(f"Agent scratchpad input: {input.get('agent_scratchpad', 'None')}")
        return super().invoke(input, **kwargs)

def _create_agent(self) -> CustomAgentExecutor:
    """Create a ReAct agent with a custom prompt."""
    system_prompt = (
        "You are a transaction risk analysis expert. Your goal is to analyze transactions for potential money laundering, fraud, or other risks, and determine if they should be APPROVED, FLAGGED, or REJECTED. "
        "Available tools: {tool_names}\n"
        "Tool descriptions:\n{tools}\n"
        "Each tool expects a JSON dictionary as input with specific keys, as described in the tool's description. "
        "When calling a tool, format the response as a JSON object with 'tool' and 'input' keys, e.g., "
        "{'tool': 'determine_purpose_and_sof', 'input': {'sender_id': 'S123', 'receiver_id': 'R456', 'amount': 1000.0, 'mtcn': 'M789'}}.\n"
        "Use insights from previous analyses to guide your strategy, as provided in the input.\n"
        "For intermediate steps, append tool calls and results to the conversation history as messages.\n"
        "After completing all tool calls, provide a final answer with the key 'output' containing the transaction assessment (e.g., 'APPROVED', 'FLAGGED', or 'REJECTED')."
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])

    # Log prompt variables for debugging
    logger.info(f"Prompt template variables: {prompt.input_variables}")

    agent = create_react_agent(
        llm=self.llm,
        tools=self.tools,
        prompt=prompt
    )
    return CustomAgentExecutor(
        agent=agent,
        tools=self.tools,
        verbose=True,
        memory=self.memory,
        handle_parsing_errors=self._handle_parsing_error,
        max_iterations=15
    )
