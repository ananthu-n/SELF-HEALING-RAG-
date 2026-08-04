import unittest
from app.core.cancellation import CancellationManager
from app.self_healing.controller import SelfHealingController
from app.self_healing.graph_controller import LangGraphSelfHealingController

class TestCancellation(unittest.TestCase):
    def test_cancellation_manager(self):
        session_id = "test_sess_123"
        
        # Initial state: not cancelled
        self.assertFalse(CancellationManager.is_cancelled(session_id))
        
        # Set cancel
        CancellationManager.cancel(session_id)
        self.assertTrue(CancellationManager.is_cancelled(session_id))
        
        # Clear cancel
        CancellationManager.clear(session_id)
        self.assertFalse(CancellationManager.is_cancelled(session_id))

    def test_controller_cancellation_standard(self):
        session_id = "cancel_std_test"
        CancellationManager.cancel(session_id)
        
        controller = SelfHealingController()
        
        # Calling answer should immediately raise InterruptedError
        with self.assertRaises(InterruptedError):
            controller.answer(
                query="What is LangGraph?",
                search_scope="hybrid",
                session_id=session_id
            )
            
        CancellationManager.clear(session_id)

    def test_controller_cancellation_graph(self):
        session_id = "cancel_graph_test"
        CancellationManager.cancel(session_id)
        
        controller = LangGraphSelfHealingController()
        
        # Calling answer should immediately raise InterruptedError
        with self.assertRaises(InterruptedError):
            controller.answer(
                query="What is LangGraph?",
                search_scope="hybrid",
                session_id=session_id
            )
            
        CancellationManager.clear(session_id)
