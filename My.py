import json
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime

from langchain.agents import Tool, initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage
import asyncio

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Mock services (replace with actual implementations)
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
        
        # Initialize agent
        self.agent = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            memory=self.memory,
            handle_parsing_errors=True,
            max_iterations=15
        )
    
    def _initialize_tools(self) -> List[Tool]:
        """Initialize all available tools for the agent"""
        return [
            Tool(
                name="fetch_customer_data",
                func=self._fetch_customer_data_tool,
                description="Fetch customer transaction data from database. Input: mtcn"
            ),
            Tool(
                name="analyze_high_band_risk",
                func=self._analyze_high_band_tool,
                description="Analyze transaction for high-band risk indicators. Input: sender_id, amount"
            ),
            Tool(
                name="verify_sender_identity",
                func=self._verify_sender_identity_tool,
                description="Verify sender identity and risk level. Input: sender_id"
            ),
            Tool(
                name="verify_receiver_identity",
                func=self._verify_receiver_identity_tool,
                description="Verify receiver identity and risk level. Input: receiver_id"
            ),
            Tool(
                name="validate_salary_information",
                func=self._validate_salary_tool,
                description="Validate sender's salary and employment information. Input: sender_id"
            ),
            Tool(
                name="validate_business_profile",
                func=self._validate_business_tool,
                description="Validate sender's business profile. Input: sender_id"
            ),
            Tool(
                name="determine_purpose_and_sof",
                func=self._determine_purpose_sof_tool,
                description="Determine purpose of transaction and source of funds. Input: sender_id, receiver_id, amount, mtcn"
            ),
            Tool(
                name="check_sanctions_and_watchlists",
                func=self._check_sanctions_tool,
                description="Check sender and receiver against sanctions and watchlists. Input: sender_id, receiver_id"
            ),
            Tool(
                name="generate_risk_assessment",
                func=self._generate_assessment_tool,
                description="Generate final risk assessment and recommendation. Input: all_collected_data"
            )
        ]
    
    # Tool implementations
    def _fetch_customer_data_tool(self, mtcn: str) -> str:
        try:
            service = DataFetchService()
            data = service.fetch_customer_data(mtcn)
            logger.info(f"Fetched customer data for MTCN: {mtcn}")
            return json.dumps(data)
        except Exception as e:
            logger.error(f"Error fetching customer data: {e}")
            return json.dumps({"error": str(e)})
    
    def _analyze_high_band_tool(self, input_str: str) -> str:
        try:
            sender_id, amount = input_str.split(",")
            service = HighBandValidationService()
            data = service.analyze(sender_id.strip(), float(amount.strip()))
            logger.info(f"Analyzed high-band risk for sender: {sender_id}")
            return json.dumps(data)
        except Exception as e:
            logger.error(f"Error analyzing high-band risk: {e}")
            return json.dumps({"error": str(e)})
    
    def _verify_sender_identity_tool(self, sender_id: str) -> str:
        try:
            service = IdentityVerificationService()
            data = service.fetch_sender_identity(sender_id)
            logger.info(f"Verified sender identity: {sender_id}")
            return json.dumps(data)
        except Exception as e:
            logger.error(f"Error verifying sender identity: {e}")
            return json.dumps({"error": str(e)})
    
    def _verify_receiver_identity_tool(self, receiver_id: str) -> str:
        try:
            service = IdentityVerificationService()
            data = service.fetch_receiver_identity(receiver_id)
            logger.info(f"Verified receiver identity: {receiver_id}")
            return json.dumps(data)
        except Exception as e:
            logger.error(f"Error verifying receiver identity: {e}")
            return json.dumps({"error": str(e)})
    
    def _validate_salary_tool(self, sender_id: str) -> str:
        try:
            service = SalaryValidationService()
            data = service.validate_salary(sender_id)
            logger.info(f"Validated salary for sender: {sender_id}")
            return json.dumps(data)
        except Exception as e:
            logger.error(f"Error validating salary: {e}")
            return json.dumps({"error": str(e)})
    
    def _validate_business_tool(self, sender_id: str) -> str:
        try:
            service = BusinessValidationService()
            data = service.validate_business(sender_id)
            logger.info(f"Validated business for sender: {sender_id}")
            return json.dumps(data)
        except Exception as e:
            logger.error(f"Error validating business: {e}")
            return json.dumps({"error": str(e)})
    
    def _determine_purpose_sof_tool(self, input_str: str) -> str:
        try:
            parts = input_str.split(",")
            if len(parts) != 4:
                raise ValueError("Expected 4 parameters: sender_id, receiver_id, amount, mtcn")
            
            sender_id, receiver_id, amount, mtcn = [p.strip() for p in parts]
            service = PurposeSOFValidationService()
            data = service.fetch_purpose_sof(sender_id, receiver_id, float(amount), mtcn)
            logger.info(f"Determined purpose/SOF for transaction: {mtcn}")
            return json.dumps(data)
        except Exception as e:
            logger.error(f"Error determining purpose/SOF: {e}")
            return json.dumps({"error": str(e)})
    
    def _check_sanctions_tool(self, input_str: str) -> str:
        try:
            sender_id, receiver_id = input_str.split(",")
            service = SanctionsCheckService()
            data = service.check_sanctions(sender_id.strip(), receiver_id.strip())
            logger.info(f"Checked sanctions for sender: {sender_id}, receiver: {receiver_id}")
            return json.dumps(data)
        except Exception as e:
            logger.error(f"Error checking sanctions: {e}")
            return json.dumps({"error": str(e)})
    
    def _generate_assessment_tool(self, all_data: str) -> str:
        try:
            # Parse all collected data and generate final assessment
            data = json.loads(all_data) if isinstance(all_data, str) else all_data
            
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
            return json.dumps({"error": str(e)})
    
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
        1. Start by fetching basic customer data
        2. Based on initial findings, decide what additional information you need
        3. Use tools strategically to build a complete risk profile
        4. Consider all factors: amount, frequency, parties involved, purpose, source of funds
        5. Generate a final assessment with clear recommendation
        
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

if __name__ == "__main__":
    main()
