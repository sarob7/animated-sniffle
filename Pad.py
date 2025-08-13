import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import re
import asyncio

from langchain.agents import Tool, initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class AnalysisPattern:
    pattern_id: str
    triggers: List[str]  # Conditions that trigger this pattern
    recommended_tools: List[str]  # Tools to prioritize
    risk_indicators: List[str]  # Key risk factors to watch
    confidence_score: float  # How reliable this pattern is
    usage_count: int = 0
    success_rate: float = 0.0

@dataclass
class TransactionProfile:
    amount_range: str
    sender_country: str
    receiver_country: str
    purpose_category: str
    risk_level: str

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

# Enhanced Agentic Transaction Analysis System with Pattern Learning
class EnhancedAgenticTransactionAnalyzer:
    def __init__(self, openai_api_key: str):
        self.llm = ChatOpenAI(
            base_url="http://164.52.202.192:8000/v1",
            model="meta-llama/Meta-Llama-3.1-70B-Instruct",
            max_tokens=2048,
            temperature=0
        )
        
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        self.analysis_history = []
        self.learned_patterns: Dict[str, AnalysisPattern] = {}
        self.knowledge_base = {
            "high_risk_indicators": [],
            "common_approval_patterns": [],
            "frequent_rejection_reasons": [],
            "tool_effectiveness": {}
        }
        
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
        
        # Load default patterns
        self._initialize_default_patterns()
    
    def _initialize_default_patterns(self):
        """Initialize with some common patterns"""
        # High amount pattern
        self.learned_patterns["high_amount_pattern"] = AnalysisPattern(
            pattern_id="high_amount_pattern",
            triggers=["amount > 10000", "currency == USD"],
            recommended_tools=["analyze_high_band_risk", "validate_salary_information", "validate_business_profile"],
            risk_indicators=["large_amount", "insufficient_income", "business_mismatch"],
            confidence_score=0.8
        )
        
        # Family remittance pattern
        self.learned_patterns["family_remittance_pattern"] = AnalysisPattern(
            pattern_id="family_remittance_pattern",
            triggers=["purpose == Family Support", "relationship == family"],
            recommended_tools=["determine_purpose_and_sof", "verify_receiver_identity"],
            risk_indicators=["inconsistent_relationship", "excessive_frequency"],
            confidence_score=0.9
        )
    
    def _safe_parse_input(self, input_str: str, expected_params: int) -> List[str]:
        """Safely parse tool input with robust error handling"""
        try:
            # Handle JSON input
            if input_str.strip().startswith('{'):
                data = json.loads(input_str)
                if isinstance(data, dict):
                    return list(data.values())[:expected_params]
            
            # Handle comma-separated input
            if ',' in input_str:
                parts = [part.strip().strip('"\'') for part in input_str.split(',')]
                return parts[:expected_params]
            
            # Handle single parameter
            if expected_params == 1:
                return [input_str.strip().strip('"\'')]
            
            # Handle space-separated input as fallback
            parts = input_str.strip().split()
            return parts[:expected_params]
            
        except Exception as e:
            logger.warning(f"Input parsing failed for '{input_str}': {e}")
            # Return the original string split by common delimiters
            cleaned_input = re.sub(r'[{}"\']', '', input_str)
            parts = re.split(r'[,\s]+', cleaned_input)
            return [part.strip() for part in parts if part.strip()][:expected_params]
    
    def _get_matching_patterns(self, transaction_data: Dict) -> List[AnalysisPattern]:
        """Find patterns that match the current transaction"""
        matching_patterns = []
        
        for pattern in self.learned_patterns.values():
            match_score = 0
            total_triggers = len(pattern.triggers)
            
            for trigger in pattern.triggers:
                try:
                    # Simple pattern matching - can be enhanced with more sophisticated logic
                    if self._evaluate_trigger(trigger, transaction_data):
                        match_score += 1
                except Exception as e:
                    logger.warning(f"Error evaluating trigger '{trigger}': {e}")
            
            # Pattern matches if more than 50% of triggers are satisfied
            if total_triggers > 0 and (match_score / total_triggers) > 0.5:
                matching_patterns.append(pattern)
        
        return sorted(matching_patterns, key=lambda p: p.confidence_score, reverse=True)
    
    def _evaluate_trigger(self, trigger: str, data: Dict) -> bool:
        """Evaluate if a trigger condition is met"""
        try:
            # Simple evaluation logic - can be enhanced
            if "amount >" in trigger:
                threshold = float(trigger.split(">")[1].strip())
                return data.get("amount", 0) > threshold
            elif "==" in trigger:
                key, value = trigger.split("==")
                return str(data.get(key.strip(), "")).lower() == value.strip().strip('"\'').lower()
            elif "purpose" in trigger.lower():
                return trigger.lower().replace("purpose == ", "").strip('"\'') in str(data.get("purpose_of_transaction", "")).lower()
            elif "relationship" in trigger.lower():
                return trigger.lower().replace("relationship == ", "").strip('"\'') in str(data.get("relationship", "")).lower()
        except Exception:
            pass
        return False
    
    def _initialize_tools(self) -> List[Tool]:
        """Initialize all available tools for the agent with robust input parsing"""
        return [
            Tool(
                name="fetch_customer_data",
                func=self._fetch_customer_data_tool,
                description="Fetch customer transaction data from database. Input: mtcn (string)"
            ),
            Tool(
                name="analyze_high_band_risk",
                func=self._analyze_high_band_tool,
                description="Analyze transaction for high-band risk indicators. Input: 'sender_id,amount' or JSON with sender_id and amount"
            ),
            Tool(
                name="verify_sender_identity",
                func=self._verify_sender_identity_tool,
                description="Verify sender identity and risk level. Input: sender_id (string)"
            ),
            Tool(
                name="verify_receiver_identity",
                func=self._verify_receiver_identity_tool,
                description="Verify receiver identity and risk level. Input: receiver_id (string)"
            ),
            Tool(
                name="validate_salary_information",
                func=self._validate_salary_tool,
                description="Validate sender's salary and employment information. Input: sender_id (string)"
            ),
            Tool(
                name="validate_business_profile",
                func=self._validate_business_tool,
                description="Validate sender's business profile. Input: sender_id (string)"
            ),
            Tool(
                name="determine_purpose_and_sof",
                func=self._determine_purpose_sof_tool,
                description="Determine purpose of transaction and source of funds. Input: 'sender_id,receiver_id,amount,mtcn' or JSON format"
            ),
            Tool(
                name="check_sanctions_and_watchlists",
                func=self._check_sanctions_tool,
                description="Check sender and receiver against sanctions and watchlists. Input: 'sender_id,receiver_id' or JSON format"
            ),
            Tool(
                name="generate_risk_assessment",
                func=self._generate_assessment_tool,
                description="Generate final risk assessment and recommendation. Input: JSON string of all collected data"
            )
        ]
    
    # Enhanced tool implementations with robust input parsing
    def _fetch_customer_data_tool(self, mtcn: str) -> str:
        try:
            # Clean the input
            mtcn = self._safe_parse_input(mtcn, 1)[0] if mtcn else ""
            service = DataFetchService()
            data = service.fetch_customer_data(mtcn)
            logger.info(f"Fetched customer data for MTCN: {mtcn}")
            return json.dumps(data)
        except Exception as e:
            logger.error(f"Error fetching customer data: {e}")
            return json.dumps({"error": str(e)})
    
    def _analyze_high_band_tool(self, input_str: str) -> str:
        try:
            params = self._safe_parse_input(input_str, 2)
            if len(params) < 2:
                return json.dumps({"error": "Missing parameters. Need sender_id and amount."})
            
            sender_id, amount_str = params[0], params[1]
            amount = float(amount_str)
            
            service = HighBandValidationService()
            data = service.analyze(sender_id, amount)
            logger.info(f"Analyzed high-band risk for sender: {sender_id}")
            return json.dumps(data)
        except Exception as e:
            logger.error(f"Error analyzing high-band risk: {e}")
            return json.dumps({"error": str(e), "input_received": input_str})
    
    def _verify_sender_identity_tool(self, sender_id: str) -> str:
        try:
            sender_id = self._safe_parse_input(sender_id, 1)[0] if sender_id else ""
            service = IdentityVerificationService()
            data = service.fetch_sender_identity(sender_id)
            logger.info(f"Verified sender identity: {sender_id}")
            return json.dumps(data)
        except Exception as e:
            logger.error(f"Error verifying sender identity: {e}")
            return json.dumps({"error": str(e)})
    
    def _verify_receiver_identity_tool(self, receiver_id: str) -> str:
        try:
            receiver_id = self._safe_parse_input(receiver_id, 1)[0] if receiver_id else ""
            service = IdentityVerificationService()
            data = service.fetch_receiver_identity(receiver_id)
            logger.info(f"Verified receiver identity: {receiver_id}")
            return json.dumps(data)
        except Exception as e:
            logger.error(f"Error verifying receiver identity: {e}")
            return json.dumps({"error": str(e)})
    
    def _validate_salary_tool(self, sender_id: str) -> str:
        try:
            sender_id = self._safe_parse_input(sender_id, 1)[0] if sender_id else ""
            service = SalaryValidationService()
            data = service.validate_salary(sender_id)
            logger.info(f"Validated salary for sender: {sender_id}")
            return json.dumps(data)
        except Exception as e:
            logger.error(f"Error validating salary: {e}")
            return json.dumps({"error": str(e)})
    
    def _validate_business_tool(self, sender_id: str) -> str:
        try:
            sender_id = self._safe_parse_input(sender_id, 1)[0] if sender_id else ""
            service = BusinessValidationService()
            data = service.validate_business(sender_id)
            logger.info(f"Validated business for sender: {sender_id}")
            return json.dumps(data)
        except Exception as e:
            logger.error(f"Error validating business: {e}")
            return json.dumps({"error": str(e)})
    
    def _determine_purpose_sof_tool(self, input_str: str) -> str:
        try:
            params = self._safe_parse_input(input_str, 4)
            if len(params) < 4:
                return json.dumps({"error": "Missing parameters. Need sender_id, receiver_id, amount, mtcn."})
            
            sender_id, receiver_id, amount_str, mtcn = params
            amount = float(amount_str)
            
            service = PurposeSOFValidationService()
            data = service.fetch_purpose_sof(sender_id, receiver_id, amount, mtcn)
            logger.info(f"Determined purpose/SOF for transaction: {mtcn}")
            return json.dumps(data)
        except Exception as e:
            logger.error(f"Error determining purpose/SOF: {e}")
            return json.dumps({"error": str(e), "input_received": input_str})
    
    def _check_sanctions_tool(self, input_str: str) -> str:
        try:
            params = self._safe_parse_input(input_str, 2)
            if len(params) < 2:
                return json.dumps({"error": "Missing parameters. Need sender_id and receiver_id."})
            
            sender_id, receiver_id = params[0], params[1]
            service = SanctionsCheckService()
            data = service.check_sanctions(sender_id, receiver_id)
            logger.info(f"Checked sanctions for sender: {sender_id}, receiver: {receiver_id}")
            return json.dumps(data)
        except Exception as e:
            logger.error(f"Error checking sanctions: {e}")
            return json.dumps({"error": str(e), "input_received": input_str})
    
    def _generate_assessment_tool(self, all_data: str) -> str:
        try:
            # Parse all collected data and generate final assessment
            if isinstance(all_data, str):
                try:
                    data = json.loads(all_data)
                except json.JSONDecodeError:
                    # If not valid JSON, treat as text summary
                    data = {"summary": all_data}
            else:
                data = all_data
            
            # Get matching patterns for context
            matching_patterns = self._get_matching_patterns(data)
            pattern_context = ""
            if matching_patterns:
                pattern_context = f"\nBased on learned patterns, pay special attention to: {', '.join([p.pattern_id for p in matching_patterns[:3]])}"
            
            assessment_prompt = f"""
            Based on the following transaction analysis data, generate a comprehensive risk assessment:
            
            {json.dumps(data, indent=2)}
            {pattern_context}
            
            Provide:
            1. Overall risk score (0-10)
            2. Key risk factors identified
            3. Recommendation (APPROVE/FLAG/REJECT)
            4. Detailed reasoning
            5. Required actions if any
            6. Confidence level in the assessment
            """
            
            response = self.llm([HumanMessage(content=assessment_prompt)])
            logger.info("Generated final risk assessment")
            return response.content
        except Exception as e:
            logger.error(f"Error generating assessment: {e}")
            return json.dumps({"error": str(e)})
    
    def _generate_pattern_aware_prompt(self, mtcn: str, initial_data: Dict) -> str:
        """Generate analysis prompt that incorporates learned patterns"""
        matching_patterns = self._get_matching_patterns(initial_data)
        
        base_prompt = f"""
        You are a transaction risk analysis expert analyzing transaction {mtcn}. 
        Determine if this transaction should be APPROVED, FLAGGED for review, or REJECTED.
        
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
        """
        
        if matching_patterns:
            pattern_guidance = f"""
            PATTERN-BASED RECOMMENDATIONS:
            Based on similar past transactions, I recommend prioritizing these tools:
            {', '.join(matching_patterns[0].recommended_tools)}
            
            Pay special attention to these risk indicators:
            {', '.join(matching_patterns[0].risk_indicators)}
            
            This analysis pattern has been successful {matching_patterns[0].usage_count} times with {matching_patterns[0].success_rate:.1%} success rate.
            """
            base_prompt += pattern_guidance
        
        strategy_section = """
        
        ANALYSIS STRATEGY:
        1. Start by fetching basic customer data
        2. Use pattern-based recommendations to prioritize tools efficiently  
        3. Focus on the most critical risk indicators first
        4. Build a complete risk profile systematically
        5. Consider amount, frequency, parties involved, purpose, source of funds
        6. Generate final assessment with clear recommendation and confidence level
        
        Begin your analysis:
        """
        
        return base_prompt + strategy_section
    
    def analyze_transaction(self, mtcn: str) -> Dict[str, Any]:
        """
        Main method to analyze a transaction using pattern-aware agentic approach
        """
        start_time = datetime.now()
        
        try:
            # First, get basic transaction data to inform pattern matching
            basic_data_service = DataFetchService()
            initial_data = basic_data_service.fetch_customer_data(mtcn)
            
            # Generate pattern-aware prompt
            initial_prompt = self._generate_pattern_aware_prompt(mtcn, initial_data)
            
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
                "initial_data": initial_data,
                "patterns_used": [p.pattern_id for p in self._get_matching_patterns(initial_data)],
                "conversation_history": [msg.content for msg in self.memory.chat_memory.messages]
            }
            
            self.analysis_history.append(analysis_record)
            
            # Learn from this analysis
            self._learn_from_analysis(analysis_record)
            
            return {
                "success": True,
                "mtcn": mtcn,
                "result": result,
                "processing_time": processing_time,
                "patterns_applied": analysis_record["patterns_used"],
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
    
    def _learn_from_analysis(self, analysis_record: Dict):
        """Learn from completed analysis and update patterns"""
        try:
            # Extract insights from the analysis
            result_text = analysis_record.get("result", "").lower()
            
            # Determine if analysis was successful (simplified logic)
            was_successful = "error" not in result_text and len(result_text) > 100
            
            # Update pattern usage statistics
            for pattern_id in analysis_record.get("patterns_used", []):
                if pattern_id in self.learned_patterns:
                    pattern = self.learned_patterns[pattern_id]
                    pattern.usage_count += 1
                    if was_successful:
                        # Update success rate using exponential moving average
                        pattern.success_rate = 0.9 * pattern.success_rate + 0.1 * 1.0
                    else:
                        pattern.success_rate = 0.9 * pattern.success_rate + 0.1 * 0.0
            
            # Extract new patterns if this was a novel scenario
            self._extract_new_patterns(analysis_record)
            
            logger.info(f"Learned from analysis of {analysis_record['mtcn']}")
            
        except Exception as e:
            logger.error(f"Error in learning process: {e}")
    
    def _extract_new_patterns(self, analysis_record: Dict):
        """Extract new patterns from successful analyses"""
        try:
            initial_data = analysis_record.get("initial_data", {})
            processing_time = analysis_record.get("processing_time", 0)
            
            # Create new pattern for fast-processing successful analyses
            if processing_time < 30 and "patterns_used" in analysis_record:
                pattern_id = f"efficient_pattern_{len(self.learned_patterns)}"
                
                # Create triggers based on transaction characteristics
                triggers = []
                if initial_data.get("amount", 0) > 1000:
                    triggers.append(f"amount > {initial_data['amount'] * 0.8}")
                if initial_data.get("currency"):
                    triggers.append(f"currency == {initial_data['currency']}")
                
                new_pattern = AnalysisPattern(
                    pattern_id=pattern_id,
                    triggers=triggers,
                    recommended_tools=self._extract_tools_from_conversation(analysis_record),
                    risk_indicators=[],
                    confidence_score=0.6  # Start with moderate confidence
                )
                
                self.learned_patterns[pattern_id] = new_pattern
                logger.info(f"Created new pattern: {pattern_id}")
                
        except Exception as e:
            logger.error(f"Error extracting new patterns: {e}")
    
    def _extract_tools_from_conversation(self, analysis_record: Dict) -> List[str]:
        """Extract tools that were actually used in the conversation"""
        tools_used = []
        conversation = analysis_record.get("conversation_history", [])
        
        tool_names = [
            "fetch_customer_data", "analyze_high_band_risk", "verify_sender_identity",
            "verify_receiver_identity", "validate_salary_information", "validate_business_profile",
            "determine_purpose_and_sof", "check_sanctions_and_watchlists", "generate_risk_assessment"
        ]
        
        for message in conversation:
            if isinstance(message, str):
                for tool_name in tool_names:
                    if tool_name in message:
                        tools_used.append(tool_name)
        
        return list(set(tools_used))  # Remove duplicates
    
    def get_analysis_history(self) -> List[Dict]:
        """Return history of all analyses"""
        return self.analysis_history
    
    def get_learned_patterns(self) -> Dict[str, AnalysisPattern]:
        """Return all learned patterns"""
        return self.learned_patterns
    
    def get_pattern_statistics(self) -> Dict:
        """Get statistics about pattern usage and effectiveness"""
        stats = {
            "total_patterns": len(self.learned_patterns),
            "most_used_patterns": [],
            "most_successful_patterns": [],
            "total_analyses": len(self.analysis_history)
        }
        
        # Sort patterns by usage
        by_usage = sorted(self.learned_patterns.values(), key=lambda p: p.usage_count, reverse=True)
        stats["most_used_patterns"] = [(p.pattern_id, p.usage_count) for p in by_usage[:5]]
        
        # Sort patterns by success rate
        by_success = sorted(self.learned_patterns.values(), key=lambda p: p.success_rate, reverse=True)
        stats["most_successful_patterns"] = [(p.pattern_id, p.success_rate) for p in by_success[:5]]
        
        return stats
    
    def clear_memory(self):
        """Clear the agent's conversation memory"""
        self.memory.clear()

# Async version with pattern learning
class AsyncEnhancedAgenticTransactionAnalyzer(EnhancedAgenticTransactionAnalyzer):
    def __init__(self, openai_api_key: str):
        super().__init__(openai_api_key)
    
    async def analyze_transaction_async(self, mtcn: str) -> Dict[str, Any]:
        """Async version of transaction analysis"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.analyze_transaction, mtcn)
    
    async def batch_analyze_transactions(self, mtcn_list: List[str]) -> List[Dict[str, Any]]:
        """Analyze multiple transactions concurrently with pattern learning"""
        tasks = [self.analyze_transaction_async(mtcn) for mtcn in mtcn_list]
        results = await asyncio.gather(*tasks)
        return results

# Usage example
def main():
    # Initialize the enhanced analyzer
    analyzer = EnhancedAgenticTransactionAnalyzer(openai_api_key="your-api-key-here")
    
    # Analyze multiple transactions to see pattern learning in action
    test_transactions = ["MTCN123456", "MTCN123457", "MTCN123458"]
    
    for mtcn in test_transactions:
        print(f"\nAnalyzing transaction: {mtcn}")
        result = analyzer.analyze_transaction(mtcn)
        print(json.dumps(result, indent=2))
    
    # View pattern statistics
    print("\nPattern Learning Statistics:")
    stats = analyzer.get_pattern_statistics()
    print(json.dumps(stats, indent=2))
    
    # View learned patterns
    print("\nLearned Patterns:")
    for pattern_id, pattern in analyzer.get_learned_patterns().items():
        print(f"{pattern_id}: Used {pattern.usage_count} times, {pattern.success_rate:.1%} success rate")

if __name__ == "__main__":
    main()
