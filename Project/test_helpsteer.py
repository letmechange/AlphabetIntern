#!/usr/bin/env python3
"""
Simple test to verify HelpSteer system works
"""

def test_imports():
    """Test that all modules can be imported"""
    try:
        from HelpSteer import HelpSteerSystem, EvaluationDimension, HelpSteerResponse
        print("✅ HelpSteer imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_classes():
    """Test that classes can be instantiated"""
    try:
        from HelpSteer import EvaluationDimension, HelpSteerResponse
        
        # Test enum
        dim = EvaluationDimension.HELPFULNESS
        print(f"✅ EvaluationDimension enum works: {dim.value}")
        
        # Test dataclass
        response = HelpSteerResponse(
            query="test",
            response="test response",
            context="test context",
            scores={},
            overall_score=0.0,
            metadata={}
        )
        print("✅ HelpSteerResponse dataclass works")
        
        return True
    except Exception as e:
        print(f"❌ Class instantiation error: {e}")
        return False

if __name__ == "__main__":
    print("Testing HelpSteer system...")
    print("=" * 40)
    
    success = True
    success &= test_imports()
    success &= test_classes()
    
    print("=" * 40)
    if success:
        print("✅ All tests passed! HelpSteer system is working correctly.")
    else:
        print("❌ Some tests failed. Please check the errors above.") 