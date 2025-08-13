from langchain.agents import AgentExecutor
from langchain.schema import AIMessage, BaseMessage
from typing import Any, Dict, List, Tuple

class CustomAgentExecutor(AgentExecutor):
    def _prepare_intermediate_steps(self, intermediate_steps: List[Tuple[Any, Any]]) -> List[BaseMessage]:
        """Convert intermediate steps to a list of BaseMessage objects."""
        messages = []
        for action, observation in intermediate_steps:
            # Log raw step for debugging
            logger.info(f"Intermediate step - action: {action}, observation: {observation}")
            # Convert action and observation to AIMessage
            action_content = str(action) if not isinstance(action, str) else action
            observation_content = str(observation) if not isinstance(observation, str) else observation
            messages.append(AIMessage(content=f"Action: {action_content}"))
            messages.append(AIMessage(content=f"Observation: {observation_content}"))
        logger.info(f"Prepared agent_scratchpad: {messages}")
        return messages

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
