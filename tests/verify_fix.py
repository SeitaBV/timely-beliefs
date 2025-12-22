from datetime import timedelta
from timely_beliefs.sensors.utils import eval_verified_knowledge_horizon_fnc

def test_ex_ante_execution():
    print("Testing ex_ante execution without context arguments...")
    
    # ex_ante requires 'ex_ante_horizon'.
    par = {"ex_ante_horizon": "PT0H"} 
    
    try:
        # We expect this to SUCCEED because the validator now specifically checks for 'ex_ante_horizon'
        result = eval_verified_knowledge_horizon_fnc(
            requested_fnc_name="ex_ante",
            par=par,
            # No event_start/resolution passed, mimicking the failure case
        )
        print(f"Success! Result: {result}")
        assert result == timedelta(hours=0)
    except Exception as e:
        print(f"FAILED: {e}")
        exit(1)

if __name__ == "__main__":
    test_ex_ante_execution()
