def _create_simple_agent(self) -> AgentExecutor:
        """Create a simple agent that bypasses scratchpad issues."""
        from langchain.schema import AgentAction, AgentFinish
        from langchain.agents import Tool
        
        # Convert StructuredTools to simple Tools
        simple_tools = []
        for tool in self.tools:
            simple_tools.append(
                Tool(
                    name=tool.name,
                    description=tool.description,
                    func=lambda x, t=tool: t.run(x)
                )
            )
        
        # Create a custom agent class
        class SimpleTransactionAgent:
            def __init__(self, llm, tools):
                self.llm = llm
                self.tools = {tool.name: tool for tool in tools}
                
            def plan(self, input_text: str, intermediate_steps: list) -> AgentAction or AgentFinish:
                # Simple planning logic
                if not intermediate_steps:
                    return AgentAction(
                        tool="fetch_customer_data",
                        tool_input={"mtcn": input_text.split()[-1]},  # Extract MTCN from input
                        log="Starting with customer data fetch"
                    )
                else:
                    return AgentFinish(
                        return_values={"output": "Analysis complete"},
                        log="Finished analysis"
                    )
        
        agent = SimpleTransactionAgent(self.llm, simple_tools)
        
        return AgentExecutor(
            agent=agent,
            tools=simple_tools,
            verbose=True,
            max_iterations=5
        )    def _create_agent_alternative(self) -> AgentExecutor:
        """Alternative agent creation method with manual scratchpad handling."""
        from langchain.agents import create_react_agent
        from langchain.agents.format_scratchpad import format_log_to_str
        from langchain.agents.output_parsers import ReActSingleInputOutputParser
        
        # Custom prompt that handles string scratchpad
        system_prompt = """You are a transaction risk analysis expert. Your goal is to analyze transactions for potential money laundering, fraud, or other risks.

You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought: {agent_scratchpad}"""

        prompt = ChatPromptTemplate.from_template(system_prompt)
        
        # Create the agent
        agent = create_react_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=prompt,
        )
        
        return AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=15,
        )import json
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
from pydantic import BaseModel, Field

from langchain.agents import AgentExecutor
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage
from langchain_core.tools import StructuredTool, ToolException
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_openai_functions_agent
from langchain.agents.format_scratchpad.openai_tools import format_to_openai_tool_messages
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
import asyncio

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Mock services (unchanged)
class DataFetchService:
    def fetch_customer_data(self, mtcn: str) -> Dict:
        return {
            "sender_id": "SENDER123",
            "receiver_id": "RECEIVER456",
            "amount": 5000.00,
            "currency": "USD",
            "transaction_date": "2024-01-15",
            "sender_name": "John Doe",
            "receiver_name": "Jane Smith"
        }

class HighBandValidationService:
    def analyze(self, sender_id: str, amount: float) -> Dict:
        return {
            "risk_score": 0.7,
            "red_flags": ["High amount", "Frequent transactions"],
            "band_classification": "High"
        }

class IdentityVerificationService:
    def fetch_sender_identity(self, sender_id: str) -> Dict:
        return {
            "verified": True,
            "identity_type": "Passport",
            "country": "US",
            "risk_level": "Low"
        }
    
    def fetch_receiver_identity(self, receiver_id: str) -> Dict:
        return {
            "verified": True,
            "identity_type": "National ID",
            "country": "CA",
            "risk_level": "Medium"
        }

class SalaryValidationService:
    def validate_salary(self, sender_id: str) -> Dict:
        return {
            "salary_band": "High",
            "employment_verified": True,
            "occupation": "Software Engineer"
        }

class BusinessValidationService:
    def validate_business(self, sender_id: str) -> Dict:
        return {
            "business_registered": True,
            "business_type": "Tech Consulting",
            "annual_revenue": 500000
        }

class PurposeSOFValidationService:
    def fetch_purpose_sof(self, sender_id: str, receiver_id: str, amount: float, mtcn: str) -> Dict:
        return {
            "purpose_of_transaction": "Family Support",
            "source_of_funds": "Salary",
            "relationship": "Sibling",
            "consistency_score": 0.8
        }

class SanctionsCheckService:
    def check_sanctions(self, sender_id: str, receiver_id: str) -> Dict:
        return {
            "sender_sanctioned": False,
            "receiver_sanctioned": False,
            "watchlist_hits": []
        }

# Pydantic schemas for structured tool inputs
class CustomerDataArgs(BaseModel):
    mtcn: str = Field(..., description="Money Transfer Control Number")

class HighBandRiskArgs(BaseModel):
    sender_id: str = Field(..., description="ID of the sender")
    amount: float = Field(..., description="Transaction amount")

class IdentityArgs(BaseModel):
    user_id: str = Field(..., description="ID of the user (sender or receiver)")

class PurposeSOFArgs(BaseModel):
    sender_id: str = Field(..., description="ID of the sender")
    receiver_id: str = Field(..., description="ID of the receiver")
    amount: float = Field(..., description="Transaction amount")
    mtcn: str = Field(..., description="Money Transfer Control Number")

class SanctionsArgs(BaseModel):
    sender_id: str = Field(..., description="ID of the sender")
    receiver_id: str = Field(..., description="ID of the receiver")

class AssessmentArgs(BaseModel):
    all_collected_data: Dict = Field(..., description="Collected data from all previous tool calls")

# Agentic Transaction Analysis System
class AgenticTransactionAnalyzer:
    def __init__(self, openai_api_key: str):
        self.llm = ChatOpenAI(
            base_url="http://164.52.202.192:8000/v1",
            model="meta-llama/Meta-Llama-3.1-70B-Instruct",
            max_tokens=2048,
            temperature=0
        )
        
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        self.analysis_history = []
        
        # Initialize tools
        self.tools = self._initialize_tools()
        
        # Initialize agent - try OpenAI Functions first, fall back to ReAct if needed
        try:
            self.agent = self._create_agent()
        except Exception as e:
            logger.warning(f"OpenAI Functions agent failed: {e}. Trying alternative ReAct agent.")
            self.agent = self._create_agent_alternative()

    def _create_agent(self) -> AgentExecutor:
        """Create an OpenAI Functions agent with proper message handling."""
        
        # Define the system message
        system_prompt = """You are a transaction risk analysis expert. Your goal is to analyze transactions for potential money laundering, fraud, or other risks, and determine if they should be APPROVED, FLAGGED, or REJECTED.

You have access to various tools to gather information about transactions, verify identities, check sanctions, and assess risks.

When using tools:
- Always provide the required parameters as specified in each tool's description
- Use the information gathered strategically to build a complete risk profile
- Consider all factors: amount, frequency, parties involved, purpose, source of funds
- Make your final recommendation clear and well-reasoned

Your analysis should be thorough but efficient. Start with basic transaction data and then gather additional information based on initial findings."""

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder("chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad")
        ])

        # Create the agent with OpenAI Functions approach
        agent = create_openai_functions_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=prompt,
        )
        
        return AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,
            memory=self.memory,
            handle_parsing_errors=True,
            max_iterations=15,
            return_intermediate_steps=True
        )

    def _initialize_tools(self) -> List[StructuredTool]:
        """Initialize all available tools for the agent."""
        return [
            StructuredTool.from_function(
                func=self._fetch_customer_data_tool,
                name="fetch_customer_data",
                description="Fetch customer transaction data from database. Expects a JSON dictionary with key: 'mtcn'. Example: {'mtcn': 'M789'}.",
                args_schema=CustomerDataArgs,
                handle_tool_error=True
            ),
            StructuredTool.from_function(
                func=self._analyze_high_band_tool,
                name="analyze_high_band_risk",
                description="Analyze transaction for high-band risk indicators. Expects a JSON dictionary with keys: 'sender_id', 'amount'. Example: {'sender_id': 'S123', 'amount': 1000.0}.",
                args_schema=HighBandRiskArgs,
                handle_tool_error=True
            ),
            StructuredTool.from_function(
                func=self._verify_sender_identity_tool,
                name="verify_sender_identity",
                description="Verify sender identity and risk level. Expects a JSON dictionary with key: 'user_id'. Example: {'user_id': 'S123'}.",
                args_schema=IdentityArgs,
                handle_tool_error=True
            ),
            StructuredTool.from_function(
                func=self._verify_receiver_identity_tool,
                name="verify_receiver_identity",
                description="Verify receiver identity and risk level. Expects a JSON dictionary with key: 'user_id'. Example: {'user_id': 'R456'}.",
                args_schema=IdentityArgs,
                handle_tool_error=True
            ),
            StructuredTool.from_function(
                func=self._validate_salary_tool,
                name="validate_salary_information",
                description="Validate sender's salary and employment information. Expects a JSON dictionary with key: 'user_id'. Example: {'user_id': 'S123'}.",
                args_schema=IdentityArgs,
                handle_tool_error=True
            ),
            StructuredTool.from_function(
                func=self._validate_business_tool,
                name="validate_business_profile",
                description="Validate sender's business profile. Expects a JSON dictionary with key: 'user_id'. Example: {'user_id': 'S123'}.",
                args_schema=IdentityArgs,
                handle_tool_error=True
            ),
            StructuredTool.from_function(
                func=self._determine_purpose_sof_tool,
                name="determine_purpose_and_sof",
                description=(
                    "Determine purpose of transaction and source of funds. Expects a JSON dictionary with keys: "
                    "'sender_id', 'receiver_id', 'amount', 'mtcn'. Example: "
                    "{'sender_id': 'S123', 'receiver_id': 'R456', 'amount': 1000.0, 'mtcn': 'M789'}."
                ),
                args_schema=PurposeSOFArgs,
                handle_tool_error=True
            ),
            StructuredTool.from_function(
                func=self._check_sanctions_tool,
                name="check_sanctions_and_watchlists",
                description="Check sender and receiver against sanctions and watchlists. Expects a JSON dictionary with keys: 'sender_id', 'receiver_id'. Example: {'sender_id': 'S123', 'receiver_id': 'R456'}.",
                args_schema=SanctionsArgs,
                handle_tool_error=True
            ),
            StructuredTool.from_function(
                func=self._generate_assessment_tool,
                name="generate_risk_assessment",
                description="Generate final risk assessment and recommendation. Expects a JSON dictionary with key: 'all_collected_data'. Example: {'all_collected_data': {...}}.",
                args_schema=AssessmentArgs,
                handle_tool_error=True
            )
        ]

    # Tool implementations
    def _fetch_customer_data_tool(self, args: CustomerDataArgs) -> str:
        try:
            service = DataFetchService()
            data = service.fetch_customer_data(args.mtcn)
            logger.info(f"Fetched customer data for MTCN: {args.mtcn}")
            return json.dumps(data)
        except Exception as e:
            logger.error(f"Error fetching customer data: {e}")
            raise ToolException(f"Error fetching customer data: {str(e)}")

    def _analyze_high_band_tool(self, args: HighBandRiskArgs) -> str:
        try:
            service = HighBandValidationService()
            data = service.analyze(args.sender_id, args.amount)
            logger.info(f"Analyzed high-band risk for sender: {args.sender_id}")
            return json.dumps(data)
        except Exception as e:
            logger.error(f"Error analyzing high-band risk: {e}")
            raise ToolException(f"Error analyzing high-band risk: {str(e)}")

    def _verify_sender_identity_tool(self, args: IdentityArgs) -> str:
        try:
            service = IdentityVerificationService()
            data = service.fetch_sender_identity(args.user_id)
            logger.info(f"Verified sender identity: {args.user_id}")
            return json.dumps(data)
        except Exception as e:
            logger.error(f"Error verifying sender identity: {e}")
            raise ToolException(f"Error verifying sender identity: {str(e)}")

    def _verify_receiver_identity_tool(self, args: IdentityArgs) -> str:
        try:
            service = IdentityVerificationService()
            data = service.fetch_receiver_identity(args.user_id)
            logger.info(f"Verified receiver identity: {args.user_id}")
            return json.dumps(data)
        except Exception as e:
            logger.error(f"Error verifying receiver identity: {e}")
            raise ToolException(f"Error verifying receiver identity: {str(e)}")

    def _validate_salary_tool(self, args: IdentityArgs) -> str:
        try:
            service = SalaryValidationService()
            data = service.validate_salary(args.user_id)
            logger.info(f"Validated salary for sender: {args.user_id}")
            return json.dumps(data)
        except Exception as e:
            logger.error(f"Error validating salary: {e}")
            raise ToolException(f"Error validating salary: {str(e)}")

    def _validate_business_tool(self, args: IdentityArgs) -> str:
        try:
            service = BusinessValidationService()
            data = service.validate_business(args.user_id)
            logger.info(f"Validated business for sender: {args.user_id}")
            return json.dumps(data)
        except Exception as e:
            logger.error(f"Error validating business: {e}")
            raise ToolException(f"Error validating business: {str(e)}")

    def _determine_purpose_sof_tool(self, args: PurposeSOFArgs) -> str:
        try:
            service = PurposeSOFValidationService()
            data = service.fetch_purpose_sof(args.sender_id, args.receiver_id, args.amount, args.mtcn)
            logger.info(f"Determined purpose/SOF for transaction: {args.mtcn}")
            return json.dumps(data)
        except Exception as e:
            logger.error(f"Error determining purpose/SOF: {e}")
            raise ToolException(f"Error determining purpose/SOF: {str(e)}")

    def _check_sanctions_tool(self, args: SanctionsArgs) -> str:
        try:
            service = SanctionsCheckService()
            data = service.check_sanctions(args.sender_id, args.receiver_id)
            logger.info(f"Checked sanctions for sender: {args.sender_id}, receiver: {args.receiver_id}")
            return json.dumps(data)
        except Exception as e:
            logger.error(f"Error checking sanctions: {e}")
            raise ToolException(f"Error checking sanctions: {str(e)}")

    def _generate_assessment_tool(self, args: AssessmentArgs) -> str:
        try:
            data = args.all_collected_data
            assessment_prompt = f"""
            Based on the following transaction analysis data, generate a comprehensive risk assessment:
            
            {json.dumps(data, indent=2)}
            
            Provide:
            1. Overall risk score (0-10)
            2. Key risk factors identified
            3. Recommendation (APPROVE/FLAG/REJECT)
            4. Detailed reasoning
            5. Required actions if any
            """
            
            response = self.llm([HumanMessage(content=assessment_prompt)])
            logger.info("Generated final risk assessment")
            return response.content
        except Exception as e:
            logger.error(f"Error generating assessment: {e}")
            raise ToolException(f"Error generating assessment: {str(e)}")

    def analyze_transaction(self, mtcn: str) -> Dict[str, Any]:
        """
        Main method to analyze a transaction using agentic approach
        """
        start_time = datetime.now()
        
        # Retrieve relevant patterns from pattern_library (for AdvancedAgenticTransactionAnalyzer)
        patterns = getattr(self, 'pattern_library', {})
        relevant_patterns = []
        for pattern_key, pattern_data in patterns.items():
            analysis = pattern_data.get("analysis", {})
            reflection = pattern_data.get("reflection", {})
            if analysis.get("mtcn") == mtcn or "patterns_to_remember" in reflection:
                relevant_patterns.append({
                    "mtcn": analysis.get("mtcn"),
                    "critical_factors": reflection.get("critical_factors", []),
                    "most_valuable_tools": reflection.get("most_valuable_tools", []),
                    "patterns_to_remember": reflection.get("patterns_to_remember", []),
                    "timestamp": pattern_data.get("timestamp")
                })

        # Include patterns in the prompt
        patterns_str = json.dumps(relevant_patterns, indent=2) if relevant_patterns else "No relevant patterns found."
        
        # Initial prompt for the agent
        initial_prompt = f"""
        Analyze transaction {mtcn} for potential money laundering, fraud, or other risks.
        
        Your goal is to determine if this transaction should be APPROVED, FLAGGED for review, or REJECTED.
        
        Strategy:
        1. Start by fetching basic customer data using fetch_customer_data with MTCN: {mtcn}
        2. Based on initial findings, decide what additional information you need
        3. Use tools strategically to build a complete risk profile
        4. Consider all factors: amount, frequency, parties involved, purpose, source of funds
        5. Generate a final assessment with clear recommendation using generate_risk_assessment
        
        Insights from previous analyses:
        {patterns_str}
        
        Begin your analysis now.
        """
        
        try:
            # Run the agent with proper input handling
            result = self.agent.invoke({
                "input": initial_prompt,
                "chat_history": self.memory.chat_memory.messages if hasattr(self.memory, 'chat_memory') else []
            })
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Store analysis in history
            analysis_record = {
                "mtcn": mtcn,
                "timestamp": datetime.now().isoformat(),
                "processing_time": processing_time,
                "result": result.get("output", result),
                "intermediate_steps": result.get("intermediate_steps", [])
            }
            
            self.analysis_history.append(analysis_record)
            
            return {
                "success": True,
                "mtcn": mtcn,
                "result": result.get("output", result),
                "processing_time": processing_time,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in transaction analysis: {e}")
            return {
                "success": False,
                "mtcn": mtcn,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    def get_analysis_history(self) -> List[Dict]:
        """Return history of all analyses"""
        return self.analysis_history
    
    def clear_memory(self):
        """Clear the agent's conversation memory"""
        self.memory.clear()

# Enhanced version with reflection and learning capabilities
class AdvancedAgenticTransactionAnalyzer(AgenticTransactionAnalyzer):
    def __init__(self, openai_api_key: str):
        super().__init__(openai_api_key)
        self.knowledge_base = {}
        self.pattern_library = {}
    
    def reflect_on_analysis(self, analysis_result: Dict) -> Dict:
        """Reflect on completed analysis to extract insights"""
        reflection_prompt = f"""
        Reflect on this completed transaction analysis:
        
        Transaction: {analysis_result.get('mtcn', 'Unknown')}
        Result: {analysis_result.get('result', 'No result')}
        
        Questions for reflection:
        1. What were the most critical risk factors identified?
        2. Which tools provided the most valuable information?
        3. Were there any redundant steps that could be optimized?
        4. What patterns should be remembered for similar future cases?
        5. How could the analysis process be improved?
        
        Provide insights in JSON format:
        {{
            "critical_factors": ["factor1", "factor2"],
            "most_valuable_tools": ["tool1", "tool2"],
            "optimization_suggestions": ["suggestion1", "suggestion2"],
            "patterns_to_remember": ["pattern1", "pattern2"],
            "process_improvements": ["improvement1", "improvement2"]
        }}
        """
        
        try:
            response = self.llm([HumanMessage(content=reflection_prompt)])
            return json.loads(response.content)
        except Exception as e:
            logger.error(f"Error in reflection: {e}")
            return {"error": str(e)}
    
    def learn_from_analysis(self, analysis_result: Dict):
        """Learn from completed analysis and store insights"""
        reflection = self.reflect_on_analysis(analysis_result)
        
        # Store patterns and insights
        pattern_key = f"analysis_{len(self.pattern_library) + 1}"
        self.pattern_library[pattern_key] = {
            "analysis": analysis_result,
            "reflection": reflection,
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"Learned from analysis {analysis_result.get('mtcn')}")

    def analyze_transaction_with_learning(self, mtcn: str) -> Dict[str, Any]:
        """Analyze transaction and automatically learn from the result"""
        result = self.analyze_transaction(mtcn)
        
        if result["success"]:
            self.learn_from_analysis(result)
        
        return result

# Async version for better performance
class AsyncAgenticTransactionAnalyzer(AgenticTransactionAnalyzer):
    def __init__(self, openai_api_key: str):
        super().__init__(openai_api_key)
    
    async def analyze_transaction_async(self, mtcn: str) -> Dict[str, Any]:
        """Async version of transaction analysis"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.analyze_transaction, mtcn)
    
    async def batch_analyze_transactions(self, mtcn_list: List[str]) -> List[Dict[str, Any]]:
        """Analyze multiple transactions concurrently"""
        tasks = [self.analyze_transaction_async(mtcn) for mtcn in mtcn_list]
        results = await asyncio.gather(*tasks)
        return results

# Usage example
def main():
    # Initialize the advanced analyzer
    analyzer = AdvancedAgenticTransactionAnalyzer(openai_api_key="your-api-key-here")
    
    # Analyze a transaction
    result = analyzer.analyze_transaction_with_learning("MTCN123456")
    print("Analysis Result:")
    print(json.dumps(result, indent=2))
    
    # Analyze another transaction to use patterns
    result = analyzer.analyze_transaction_with_learning("MTCN789012")
    print("\nSecond Analysis Result:")
    print(json.dumps(result, indent=2))
    
    # Get analysis history
    history = analyzer.get_analysis_history()
    print(f"\nTotal analyses performed: {len(history)}")
    
    # Print pattern library
    print("\nPattern Library:")
    print(json.dumps(analyzer.pattern_library, indent=2))
    
    # Clear memory for next analysis
    analyzer.clear_memory()

if __name__ == "__main__":
    main()
