import json
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
from pydantic import BaseModel, Field

from langchain.agents import initialize_agent, AgentType, AgentExecutor
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage
from langchain_core.tools import StructuredTool, ToolException
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_react_agent
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
        ..

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
        
        # Initialize agent with a custom prompt
        self.agent = self._create_agent()

    def _create_agent(self) -> AgentExecutor:
        """Create a ReAct agent with a custom prompt."""
        system_prompt = (
            "You are a transaction risk analysis expert. Your goal is to analyze transactions for potential money laundering, fraud, or other risks, and determine if they should be APPROVED, FLAGGED, or REJECTED. "
            "Each tool expects a JSON dictionary as input with specific keys, as described in the tool's description. "
            "When calling a tool, format the input as a JSON dictionary and include all required keys. "
            "Examples:\n"
            "- For 'determine_purpose_and_sof': {'sender_id': 'S123', 'receiver_id': 'R456', 'amount': 1000.0, 'mtcn': 'M789'}\n"
            "- For 'analyze_high_band_risk': {'sender_id': 'S123', 'amount': 1000.0}\n"
            "- For 'fetch_customer_data': {'mtcn': 'M789'}\n"
            "Always respond with the tool name and the exact JSON dictionary for the tool input, e.g., "
            "{'tool': 'determine_purpose_and_sof', 'input': {'sender_id': 'S123', 'receiver_id': 'R456', 'amount': 1000.0, 'mtcn': 'M789'}}."
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder("chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad")
        ])

        agent = create_react_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=prompt
        )
        return AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,
            memory=self.memory,
            handle_parsing_errors=self._handle_parsing_error,
            max_iterations=15
        )

    def _handle_parsing_error(self, error: Exception) -> dict:
        """Handle parsing errors by attempting to fix malformed inputs."""
        error_str = str(error)
        try:
            # Attempt to parse malformed JSON or text input
            result = {}
            if "," in error_str:
                pairs = error_str.replace("{", "").replace("}", "").split(",")
                for pair in pairs:
                    if ":" in pair:
                        key, value = pair.split(":", 1)
                        key = key.strip().strip("'\"")
                        value = value.strip().strip("'\"")
                        if key == "amount":
                            try:
                                value = float(value)
                            except ValueError:
                                continue
                        result[key] = value
            return result
        except Exception as e:
            logger.error(f"Failed to parse error: {e}")
            return {"error": f"Invalid input format: {error_str}"}

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
        
        # Initial prompt for the agent
        initial_prompt = f"""
        You are a transaction risk analysis expert. Analyze transaction {mtcn} for potential money laundering, fraud, or other risks.
        
        Your goal is to determine if this transaction should be APPROVED, FLAGGED for review, or REJECTED.
        
        Available tools:
        - fetch_customer_data: Get basic transaction details
        - analyze_high_band_risk: Check for high-value transaction risks
        - verify_sender_identity: Verify sender's identity
        - verify_receiver_identity: Verify receiver's identity
        - validate_salary_information: Check sender's employment/salary
        - validate_business_profile: Check sender's business details
        - determine_purpose_and_sof: Analyze transaction purpose and source of funds
        - check_sanctions_and_watchlists: Check against sanctions/watchlists
        - generate_risk_assessment: Create final assessment and recommendation
        
        Strategy:
        1. Start by fetching basic customer data using fetch_customer_data with MTCN: {mtcn}
        2. Based on initial findings, decide what additional information you need
        3. Use tools strategically to build a complete risk profile
        4. Consider all factors: amount, frequency, parties involved, purpose, source of funds
        5. Generate a final assessment with clear recommendation using generate_risk_assessment
        
        Begin your analysis now:
        """
        
        try:
            # Run the agent
            result = self.agent.run(initial_prompt)
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Store analysis in history
            analysis_record = {
                "mtcn": mtcn,
                "timestamp": datetime.now().isoformat(),
                "processing_time": processing_time,
                "result": result,
                "conversation_history": self.memory.chat_memory.messages
            }
            
            self.analysis_history.append(analysis_record)
            
            return {
                "success": True,
                "mtcn": mtcn,
                "result": result,
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

# Enhanced version with reflection and learning capabilities (unchanged)
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

# Async version for better performance (unchanged)
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
    # Initialize the agentic analyzer
    analyzer = AgenticTransactionAnalyzer(openai_api_key="your-api-key-here")
    
    # Analyze a transaction
    result = analyzer.analyze_transaction("MTCN123456")
    
    print("Analysis Result:")
    print(json.dumps(result, indent=2))
    
    # Get analysis history
    history = analyzer.get_analysis_history()
    print(f"\nTotal analyses performed: {len(history)}")
    
    # Clear memory for next analysis
    analyzer.clear_memory()

if __name__ == "__main__":
    main()
