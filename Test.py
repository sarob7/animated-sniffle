import json
import pytest
import unittest
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any, Tuple
import logging
import time
from dataclasses import dataclass
from datetime import datetime
import re

# Import your main classes (assuming they're in the same file or imported)
# from your_module import EnhancedAgenticTransactionAnalyzer, AnalysisPattern

@dataclass
class TestCase:
    name: str
    mtcn: str
    input_data: Dict
    expected_outcome: str  # APPROVE, FLAG, REJECT
    expected_tools: List[str]  # Tools that should be called
    expected_risk_factors: List[str]
    confidence_threshold: float = 0.7

@dataclass
class TestResult:
    test_name: str
    passed: bool
    actual_outcome: str
    expected_outcome: str
    tools_used: List[str]
    missing_tools: List[str]
    extra_tools: List[str]
    execution_time: float
    consistency_score: float
    error_message: str = ""

class MockServices:
    """Controlled mock services for deterministic testing"""
    
    @staticmethod
    def create_high_risk_scenario():
        """Create data that should trigger rejection"""
        return {
            "DataFetchService": {
                "sender_id": "SENDER_HIGH_RISK",
                "receiver_id": "RECEIVER_SANCTIONS",
                "amount": 50000.00,
                "currency": "USD",
                "transaction_date": "2024-01-15",
                "sender_name": "John Doe",
                "receiver_name": "Suspicious Entity"
            },
            "HighBandValidationService": {
                "risk_score": 0.95,
                "red_flags": ["Extremely high amount", "Frequent large transactions", "Unusual pattern"],
                "band_classification": "Critical"
            },
            "IdentityVerificationService_sender": {
                "verified": False,
                "identity_type": "Unknown",
                "country": "High-risk country",
                "risk_level": "Critical"
            },
            "IdentityVerificationService_receiver": {
                "verified": False,
                "identity_type": "Unknown", 
                "country": "Sanctioned country",
                "risk_level": "Critical"
            },
            "SalaryValidationService": {
                "salary_band": "Low",
                "employment_verified": False,
                "occupation": "Unemployed"
            },
            "BusinessValidationService": {
                "business_registered": False,
                "business_type": "Unknown",
                "annual_revenue": 0
            },
            "PurposeSOFValidationService": {
                "purpose_of_transaction": "Unclear",
                "source_of_funds": "Unknown",
                "relationship": "No relationship",
                "consistency_score": 0.1
            },
            "SanctionsCheckService": {
                "sender_sanctioned": True,
                "receiver_sanctioned": True,
                "watchlist_hits": ["OFAC", "EU Sanctions", "UN Sanctions"]
            }
        }
    
    @staticmethod
    def create_low_risk_scenario():
        """Create data that should trigger approval"""
        return {
            "DataFetchService": {
                "sender_id": "SENDER_VERIFIED",
                "receiver_id": "RECEIVER_FAMILY",
                "amount": 500.00,
                "currency": "USD",
                "transaction_date": "2024-01-15",
                "sender_name": "John Smith",
                "receiver_name": "Mary Smith"
            },
            "HighBandValidationService": {
                "risk_score": 0.2,
                "red_flags": [],
                "band_classification": "Low"
            },
            "IdentityVerificationService_sender": {
                "verified": True,
                "identity_type": "Passport",
                "country": "US",
                "risk_level": "Low"
            },
            "IdentityVerificationService_receiver": {
                "verified": True,
                "identity_type": "National ID",
                "country": "CA", 
                "risk_level": "Low"
            },
            "SalaryValidationService": {
                "salary_band": "Medium",
                "employment_verified": True,
                "occupation": "Teacher"
            },
            "BusinessValidationService": {
                "business_registered": True,
                "business_type": "Education",
                "annual_revenue": 45000
            },
            "PurposeSOFValidationService": {
                "purpose_of_transaction": "Family Support",
                "source_of_funds": "Salary",
                "relationship": "Sister",
                "consistency_score": 0.95
            },
            "SanctionsCheckService": {
                "sender_sanctioned": False,
                "receiver_sanctioned": False,
                "watchlist_hits": []
            }
        }

    @staticmethod
    def create_medium_risk_scenario():
        """Create data that should trigger flagging"""
        return {
            "DataFetchService": {
                "sender_id": "SENDER_MEDIUM",
                "receiver_id": "RECEIVER_BUSINESS",
                "amount": 8000.00,
                "currency": "USD",
                "transaction_date": "2024-01-15",
                "sender_name": "Business Owner",
                "receiver_name": "Trade Partner"
            },
            "HighBandValidationService": {
                "risk_score": 0.6,
                "red_flags": ["High amount for individual"],
                "band_classification": "Medium"
            },
            "IdentityVerificationService_sender": {
                "verified": True,
                "identity_type": "Business License",
                "country": "US",
                "risk_level": "Medium"
            },
            "IdentityVerificationService_receiver": {
                "verified": True,
                "identity_type": "Business Registration",
                "country": "MX",
                "risk_level": "Medium"
            },
            "SalaryValidationService": {
                "salary_band": "High",
                "employment_verified": True,
                "occupation": "Business Owner"
            },
            "BusinessValidationService": {
                "business_registered": True,
                "business_type": "Import/Export",
                "annual_revenue": 200000
            },
            "PurposeSOFValidationService": {
                "purpose_of_transaction": "Business Payment",
                "source_of_funds": "Business Revenue",
                "relationship": "Business Partner",
                "consistency_score": 0.7
            },
            "SanctionsCheckService": {
                "sender_sanctioned": False,
                "receiver_sanctioned": False,
                "watchlist_hits": []
            }
        }

class DeterministicTester:
    """Test framework for deterministic testing of agentic behavior"""
    
    def __init__(self, analyzer_class):
        self.analyzer_class = analyzer_class
        self.test_cases = self._create_test_cases()
        self.results = []
    
    def _create_test_cases(self) -> List[TestCase]:
        """Define comprehensive test cases"""
        return [
            TestCase(
                name="high_risk_rejection",
                mtcn="MTCN_HIGH_RISK",
                input_data=MockServices.create_high_risk_scenario(),
                expected_outcome="REJECT",
                expected_tools=["fetch_customer_data", "check_sanctions_and_watchlists", "verify_sender_identity"],
                expected_risk_factors=["sanctioned_parties", "high_risk_country", "unverified_identity"]
            ),
            TestCase(
                name="low_risk_approval", 
                mtcn="MTCN_LOW_RISK",
                input_data=MockServices.create_low_risk_scenario(),
                expected_outcome="APPROVE",
                expected_tools=["fetch_customer_data", "determine_purpose_and_sof", "verify_sender_identity"],
                expected_risk_factors=[]
            ),
            TestCase(
                name="medium_risk_flagging",
                mtcn="MTCN_MEDIUM_RISK", 
                input_data=MockServices.create_medium_risk_scenario(),
                expected_outcome="FLAG",
                expected_tools=["fetch_customer_data", "analyze_high_band_risk", "validate_business_profile"],
                expected_risk_factors=["high_amount", "cross_border_business"]
            )
        ]
    
    def mock_services_for_test(self, test_case: TestCase):
        """Create mocks that return deterministic data"""
        mocks = {}
        
        # Mock each service to return specific test data
        for service_key, data in test_case.input_data.items():
            if service_key == "DataFetchService":
                mocks['DataFetchService.fetch_customer_data'] = Mock(return_value=data)
            elif service_key == "HighBandValidationService":
                mocks['HighBandValidationService.analyze'] = Mock(return_value=data)
            elif service_key == "IdentityVerificationService_sender":
                mocks['IdentityVerificationService.fetch_sender_identity'] = Mock(return_value=data)
            elif service_key == "IdentityVerificationService_receiver": 
                mocks['IdentityVerificationService.fetch_receiver_identity'] = Mock(return_value=data)
            elif service_key == "SalaryValidationService":
                mocks['SalaryValidationService.validate_salary'] = Mock(return_value=data)
            elif service_key == "BusinessValidationService":
                mocks['BusinessValidationService.validate_business'] = Mock(return_value=data)
            elif service_key == "PurposeSOFValidationService":
                mocks['PurposeSOFValidationService.fetch_purpose_sof'] = Mock(return_value=data)
            elif service_key == "SanctionsCheckService":
                mocks['SanctionsCheckService.check_sanctions'] = Mock(return_value=data)
        
        return mocks
    
    def run_single_test(self, test_case: TestCase, num_runs: int = 3) -> TestResult:
        """Run a single test case multiple times to check consistency"""
        outcomes = []
        tools_used_list = []
        execution_times = []
        
        for run in range(num_runs):
            start_time = time.time()
            
            # Create analyzer with mocked services
            analyzer = self.analyzer_class("test-api-key")
            
            # Apply mocks
            mocks = self.mock_services_for_test(test_case)
            with patch.multiple('__main__', **mocks):
                try:
                    result = analyzer.analyze_transaction(test_case.mtcn)
                    
                    # Extract outcome and tools used
                    outcome = self._extract_outcome(result.get("result", ""))
                    tools_used = self._extract_tools_used(result)
                    
                    outcomes.append(outcome)
                    tools_used_list.append(tools_used)
                    execution_times.append(time.time() - start_time)
                    
                except Exception as e:
                    outcomes.append("ERROR")
                    tools_used_list.append([])
                    execution_times.append(time.time() - start_time)
        
        # Analyze consistency
        consistency_score = len(set(outcomes)) == 1  # All outcomes should be the same
        most_common_outcome = max(set(outcomes), key=outcomes.count)
        most_common_tools = self._get_most_common_tools(tools_used_list)
        
        # Check if expected tools were used
        missing_tools = [tool for tool in test_case.expected_tools if tool not in most_common_tools]
        extra_tools = [tool for tool in most_common_tools if tool not in test_case.expected_tools]
        
        # Determine if test passed
        outcome_correct = most_common_outcome == test_case.expected_outcome
        tools_reasonable = len(missing_tools) <= 1  # Allow some flexibility
        
        return TestResult(
            test_name=test_case.name,
            passed=outcome_correct and consistency_score and tools_reasonable,
            actual_outcome=most_common_outcome,
            expected_outcome=test_case.expected_outcome,
            tools_used=most_common_tools,
            missing_tools=missing_tools,
            extra_tools=extra_tools,
            execution_time=sum(execution_times) / len(execution_times),
            consistency_score=1.0 if consistency_score else 0.0,
            error_message="" if outcome_correct else f"Expected {test_case.expected_outcome}, got {most_common_outcome}"
        )
    
    def _extract_outcome(self, result_text: str) -> str:
        """Extract the decision from the result text"""
        result_lower = result_text.lower()
        if "reject" in result_lower:
            return "REJECT"
        elif "flag" in result_lower:
            return "FLAG"  
        elif "approve" in result_lower:
            return "APPROVE"
        else:
            return "UNCLEAR"
    
    def _extract_tools_used(self, result: Dict) -> List[str]:
        """Extract which tools were actually used"""
        tools_used = []
        conversation_history = result.get("conversation_history", [])
        
        tool_names = [
            "fetch_customer_data", "analyze_high_band_risk", "verify_sender_identity",
            "verify_receiver_identity", "validate_salary_information", "validate_business_profile", 
            "determine_purpose_and_sof", "check_sanctions_and_watchlists", "generate_risk_assessment"
        ]
        
        for message in conversation_history:
            if isinstance(message, str):
                for tool_name in tool_names:
                    if tool_name in message.lower():
                        tools_used.append(tool_name)
        
        return list(set(tools_used))
    
    def _get_most_common_tools(self, tools_lists: List[List[str]]) -> List[str]:
        """Get tools that appeared in most runs"""
        tool_counts = {}
        for tools in tools_lists:
            for tool in tools:
                tool_counts[tool] = tool_counts.get(tool, 0) + 1
        
        threshold = len(tools_lists) / 2  # Tool must appear in at least half the runs
        return [tool for tool, count in tool_counts.items() if count >= threshold]
    
    def run_all_tests(self) -> List[TestResult]:
        """Run all test cases"""
        self.results = []
        
        for test_case in self.test_cases:
            print(f"Running test: {test_case.name}")
            result = self.run_single_test(test_case)
            self.results.append(result)
            
        return self.results
    
    def generate_test_report(self) -> str:
        """Generate comprehensive test report"""
        if not self.results:
            return "No test results available. Run tests first."
        
        passed_tests = sum(1 for r in self.results if r.passed)
        total_tests = len(self.results)
        
        report = f"""
=== AGENTIC TRANSACTION ANALYZER TEST REPORT ===

Overall Results: {passed_tests}/{total_tests} tests passed ({passed_tests/total_tests*100:.1f}%)

Detailed Results:
"""
        
        for result in self.results:
            status = "✅ PASS" if result.passed else "❌ FAIL"
            report += f"""
{status} {result.test_name}
  Expected: {result.expected_outcome} | Actual: {result.actual_outcome}
  Consistency Score: {result.consistency_score:.1%}
  Execution Time: {result.execution_time:.2f}s
  Tools Used: {', '.join(result.tools_used)}
  Missing Tools: {', '.join(result.missing_tools) if result.missing_tools else 'None'}
  Extra Tools: {', '.join(result.extra_tools) if result.extra_tools else 'None'}
  Error: {result.error_message if result.error_message else 'None'}
"""
        
        return report

class ConsistencyTester:
    """Test consistency across multiple runs with same input"""
    
    def __init__(self, analyzer_class):
        self.analyzer_class = analyzer_class
    
    def test_consistency(self, mtcn: str, num_runs: int = 5) -> Dict[str, Any]:
        """Test if the same MTCN produces consistent results"""
        
        results = []
        outcomes = []
        tools_used_list = []
        execution_times = []
        
        # Use the same mock data for all runs
        mock_data = MockServices.create_medium_risk_scenario()
        
        for i in range(num_runs):
            analyzer = self.analyzer_class("test-api-key")
            
            # Apply consistent mocks
            with patch('DataFetchService.fetch_customer_data', return_value=mock_data["DataFetchService"]):
                start_time = time.time()
                result = analyzer.analyze_transaction(f"{mtcn}_{i}")
                execution_times.append(time.time() - start_time)
                
                # Extract key information
                outcome = self._extract_outcome(result.get("result", ""))
                tools_used = self._extract_tools_used(result)
                
                outcomes.append(outcome)
                tools_used_list.append(tools_used)
                results.append(result)
        
        # Analyze consistency
        unique_outcomes = set(outcomes)
        consistency_score = 1.0 if len(unique_outcomes) == 1 else len(unique_outcomes) / num_runs
        
        return {
            "mtcn": mtcn,
            "num_runs": num_runs,
            "outcomes": outcomes,
            "unique_outcomes": list(unique_outcomes),
            "consistency_score": consistency_score,
            "most_common_outcome": max(set(outcomes), key=outcomes.count),
            "tools_variance": len(set(str(sorted(tools)) for tools in tools_used_list)),
            "avg_execution_time": sum(execution_times) / len(execution_times),
            "execution_time_variance": max(execution_times) - min(execution_times)
        }
    
    def _extract_outcome(self, result_text: str) -> str:
        """Extract decision from result text"""
        result_lower = result_text.lower()
        if "reject" in result_lower:
            return "REJECT"
        elif "flag" in result_lower:
            return "FLAG"
        elif "approve" in result_lower:
            return "APPROVE" 
        else:
            return "UNCLEAR"
    
    def _extract_tools_used(self, result: Dict) -> List[str]:
        """Extract tools from result"""
        # Implementation similar to DeterministicTester
        return []  # Simplified for brevity

class PerformanceTester:
    """Test performance and efficiency metrics"""
    
    def __init__(self, analyzer_class):
        self.analyzer_class = analyzer_class
    
    def test_performance_metrics(self) -> Dict[str, Any]:
        """Test various performance metrics"""
        
        test_cases = [
            ("simple_case", MockServices.create_low_risk_scenario()),
            ("complex_case", MockServices.create_high_risk_scenario()),
            ("medium_case", MockServices.create_medium_risk_scenario())
        ]
        
        results = {}
        
        for case_name, mock_data in test_cases:
            analyzer = self.analyzer_class("test-api-key")
            
            start_time = time.time()
            result = analyzer.analyze_transaction(f"PERF_{case_name}")
            execution_time = time.time() - start_time
            
            # Count tool calls
            tools_used = self._count_tool_calls(result)
            
            results[case_name] = {
                "execution_time": execution_time,
                "tools_called": len(tools_used),
                "success": result.get("success", False),
                "result_length": len(str(result.get("result", "")))
          
