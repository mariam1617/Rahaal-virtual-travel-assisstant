from typing import Any, Dict

def execute_graph_sync(compiled_graph: Any, input_state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Executes a LangGraph compiled graph synchronously.
   
    """
    try:
        
        result = compiled_graph.invoke(input_state)
        return result
    except Exception as e:
        
        raise RuntimeError(f"Graph execution failed: {str(e)}")