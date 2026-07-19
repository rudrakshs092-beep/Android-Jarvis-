"""
core/jarvis.py
JARVIS Assistant - Core Orchestrator

Production-ready, async-first orchestrator for the JARVIS Android assistant.
Implements clean architecture and dependency injection to manage the lifecycle
of all sub-modules (Brain, Memory, Voice, Automation, Plugins) without coupling
to their concrete implementations.
"""

import asyncio
import logging
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

# ==============================================================================
# Module Protocols (Contracts)
# ==============================================================================

@runtime_checkable
class AsyncModule(Protocol):
    """Standard interface for all JARVIS swappable modules."""
    async def initialize(self) -> None:
        """Set up connections, load models, etc."""
        ...
        
    async def start(self) -> None:
        """Start background tasks or listeners."""
        ...
        
    async def stop(self) -> None:
        """Gracefully terminate tasks and release resources."""
        ...
        
    async def health_check(self) -> Dict[str, Any]:
        """Return a dictionary with health status (e.g., {'status': 'healthy'})."""
        ...

# ==============================================================================
# State Management
# ==============================================================================

class JarvisState(Enum):
    UNINITIALIZED = "uninitialized"
    INITIALIZING = "initializing"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"

# ==============================================================================
# Core Orchestrator
# ==============================================================================

class Jarvis:
    """
    Main JARVIS application orchestrator.
    
    Manages the lifecycle and state of the assistant. Depends on abstractions
    (Protocols) rather than concrete implementations, allowing easy testing
    and modularity.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        brain: Optional[AsyncModule] = None,
        memory: Optional[AsyncModule] = None,
        voice: Optional[AsyncModule] = None,
        automation: Optional[AsyncModule] = None,
        plugins: Optional[List[AsyncModule]] = None
    ):
        """
        Initialize the JARVIS orchestrator with injected dependencies.

        Args:
            config: Global configuration dictionary.
            brain: LLM/Chat module instance.
            memory: Short/Long-term memory module instance.
            voice: STT/TTS module instance.
            automation: Android hardware control module instance.
            plugins: List of additional plugin modules to load.
        """
        self.config = config
        self.logger = logging.getLogger("jarvis.core")
        
        self.state = JarvisState.UNINITIALIZED
        self._main_task: Optional[asyncio.Task] = None
        
        # Dependency Injection: Store modules
        self.brain = brain
        self.memory = memory
        self.voice = voice
        self.automation = automation
        self.plugins = plugins or []
        
        self.logger.info("JARVIS Orchestrator instantiated.")

    def _get_all_modules(self) -> List[AsyncModule]:
        """Helper to aggregate all active modules for lifecycle operations."""
        modules = []
        for module in [self.brain, self.memory, self.voice, self.automation]:
            if module is not None:
                modules.append(module)
        modules.extend(self.plugins)
        return modules

    async def initialize(self) -> None:
        """
        Safely initialize all injected modules sequentially.
        Updates state to INITIALIZING. If any critical module fails, 
        state becomes ERROR.
        """
        if self.state != JarvisState.UNINITIALIZED:
            self.logger.warning(f"Cannot initialize. Current state: {self.state.value}")
            return
            
        self.state = JarvisState.INITIALIZING
        self.logger.info("Initializing JARVIS modules...")
        
        try:
            for module in self._get_all_modules():
                module_name = module.__class__.__name__
                self.logger.debug(f"Initializing {module_name}...")
                await module.initialize()
                self.logger.debug(f"{module_name} initialized successfully.")
                
            self.logger.info("All modules initialized successfully.")
            self.state = JarvisState.STOPPED # Ready to be started
            
        except Exception as e:
            self.state = JarvisState.ERROR
            self.logger.critical(f"Initialization failed: {e}", exc_info=True)
            raise RuntimeError(f"Module initialization failed: {e}") from e

    async def start(self) -> None:
        """
        Start all modules and launch the main event loop.
        Assumes `initialize()` has been successfully called.
        """
        if self.state not in [JarvisState.STOPPED, JarvisState.UNINITIALIZED]:
            self.logger.warning(f"Cannot start. Current state: {self.state.value}")
            return

        if self.state == JarvisState.UNINITIALIZED:
            await self.initialize()

        self.logger.info("Starting JARVIS modules...")
        self.state = JarvisState.RUNNING
        
        try:
            for module in self._get_all_modules():
                await module.start()
                
            # Start the main orchestrator loop
            self._main_task = asyncio.create_task(self._run_loop())
            self.logger.info("JARVIS is now online and running.")
            
        except Exception as e:
            self.state = JarvisState.ERROR
            self.logger.critical(f"Failed to start JARVIS: {e}", exc_info=True)
            await self.stop() # Attempt cleanup
            raise

    async def _run_loop(self) -> None:
        """
        Main async loop.
        In a real Android environment via Termux/Chaquopy, this keeps the 
        Python process alive and might handle IPC (Inter-Process Communication) 
        with the Android UI thread.
        """
        try:
            while self.state == JarvisState.RUNNING:
                # Heartbeat / Event polling / IPC listening goes here.
                # We use a small sleep to prevent high CPU usage.
                await asyncio.sleep(1.0)
        except asyncio.CancelledError:
            self.logger.info("Main loop cancelled.")
            raise

    async def stop(self) -> None:
        """
        Gracefully shut down JARVIS.
        Stops the main loop and calls `stop()` on all modules in reverse order.
        """
        if self.state in [JarvisState.STOPPED, JarvisState.UNINITIALIZED]:
            self.logger.info("JARVIS is already stopped.")
            return
            
        self.state = JarvisState.STOPPING
        self.logger.info("Shutting down JARVIS...")
        
        # Cancel main loop
        if self._main_task and not self._main_task.done():
            self._main_task.cancel()
            try:
                await self._main_task
            except asyncio.CancelledError:
                pass

        # Stop modules in reverse order of initialization
        modules = self._get_all_modules()
        for module in reversed(modules):
            try:
                module_name = module.__class__.__name__
                self.logger.debug(f"Stopping {module_name}...")
                await module.stop()
            except Exception as e:
                self.logger.error(f"Error stopping {module_name}: {e}")

        self.state = JarvisState.STOPPED
        self.logger.info("JARVIS shutdown complete. Goodbye.")

    async def health_check(self) -> Dict[str, Any]:
        """
        Perform a comprehensive health check across all modules.
        
        Returns:
            Dict[str, Any]: Aggregated health status dictionary.
        """
        if self.state == JarvisState.ERROR:
            return {"jarvis_status": "error", "details": "System is in error state."}

        health_report: Dict[str, Any] = {
            "jarvis_status": self.state.value,
            "modules": {}
        }
        
        for module in self._get_all_modules():
            module_name = module.__class__.__name__
            try:
                # Assumes module.health_check() returns a dict
                health_report["modules"][module_name] = await module.health_check()
            except Exception as e:
                health_report["modules"][module_name] = {"status": "unhealthy", "error": str(e)}
                
        return health_report

# ==============================================================================
# Example Usage / Mockup
# ==============================================================================

if __name__ == "__main__":
    # Mocking modules to demonstrate Dependency Injection and clean architecture
    
    class MockBrain(AsyncModule):
        async def initialize(self): pass
        async def start(self): pass
        async def stop(self): pass
        async def health_check(self): return {"status": "healthy", "model": "gpt-4-mock"}

    class MockAutomation(AsyncModule):
        async def initialize(self): pass
        async def start(self): pass
        async def stop(self): pass
        async def health_check(self): return {"status": "healthy", "termux": True}

    async def main_test():
        config = {"env": "production"}
        brain = MockBrain()
        automation = MockAutomation()
        
        jarvis = Jarvis(config=config, brain=brain, automation=automation)
        
        await jarvis.start()
        
        # Check health
        health = await jarvis.health_check()
        print("Health Check:", health)
        
        # Keep running briefly to demonstrate loop
        await asyncio.sleep(2)
        
        await jarvis.stop()

    # asyncio.run(main_test())
