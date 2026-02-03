"""
🌙 Ambient Mode Control Tool
Runtime control for ambient/autonomous background thinking.
"""

from strands import tool
from typing import Dict, Any


@tool
def ambient_mode(
    action: str,
    autonomous: bool = False,
    idle_threshold: float = None,
    max_iterations: int = None,
    cooldown: float = None,
) -> Dict[str, Any]:
    """
    Control ambient mode settings at runtime.

    Args:
        action: Action to perform - "start", "stop", "status", "configure"
        autonomous: If True, enable autonomous mode (continuous until done)
        idle_threshold: Seconds of idle before ambient triggers (default: 30)
        max_iterations: Max ambient iterations (default: 3 standard, 50 autonomous)
        cooldown: Seconds between ambient runs (default: 60 standard, 10 autonomous)

    Returns:
        Dict with status and content

    Examples:
        ambient(action="start")  # Start standard ambient mode
        ambient(action="start", autonomous=True)  # Start autonomous mode
        ambient(action="stop")  # Stop ambient mode
        ambient(action="status")  # Check current status
        ambient(action="configure", idle_threshold=60, max_iterations=5)
    """
    try:
        # Import devduck to access the singleton
        import devduck as dd

        if not hasattr(dd, "devduck") or not dd.devduck:
            return {"status": "error", "content": [{"text": "DevDuck not initialized"}]}

        devduck_instance = dd.devduck

        if action == "status":
            if devduck_instance.ambient:
                amb = devduck_instance.ambient
                status_text = f"""🌙 Ambient Mode Status:
- Enabled: {amb.running}
- Mode: {'AUTONOMOUS' if amb.autonomous else 'Standard'}
- Iterations: {amb.ambient_iterations}/{amb.autonomous_max_iterations if amb.autonomous else amb.max_iterations}
- Idle Threshold: {amb.idle_threshold}s
- Cooldown: {amb.autonomous_cooldown if amb.autonomous else amb.cooldown}s
- Pending Results: {len(amb.ambient_results_history)} stored
- Has Result: {amb.ambient_result is not None}"""
            else:
                status_text = "🌙 Ambient mode not initialized"

            return {"status": "success", "content": [{"text": status_text}]}

        elif action == "start":
            # Initialize ambient mode if needed
            if not devduck_instance.ambient:
                from devduck import AmbientMode

                devduck_instance.ambient = AmbientMode(devduck_instance)

            devduck_instance.ambient.start(autonomous=autonomous)
            mode_name = "AUTONOMOUS" if autonomous else "standard"

            return {
                "status": "success",
                "content": [{"text": f"🌙 Ambient mode started ({mode_name})"}],
            }

        elif action == "stop":
            if devduck_instance.ambient:
                devduck_instance.ambient.stop()
                return {
                    "status": "success",
                    "content": [{"text": "🌙 Ambient mode stopped"}],
                }
            else:
                return {
                    "status": "success",
                    "content": [{"text": "🌙 Ambient mode was not running"}],
                }

        elif action == "configure":
            if not devduck_instance.ambient:
                from devduck import AmbientMode

                devduck_instance.ambient = AmbientMode(devduck_instance)

            amb = devduck_instance.ambient
            changes = []

            if idle_threshold is not None:
                amb.idle_threshold = idle_threshold
                changes.append(f"idle_threshold={idle_threshold}s")

            if max_iterations is not None:
                if amb.autonomous:
                    amb.autonomous_max_iterations = max_iterations
                else:
                    amb.max_iterations = max_iterations
                changes.append(f"max_iterations={max_iterations}")

            if cooldown is not None:
                if amb.autonomous:
                    amb.autonomous_cooldown = cooldown
                else:
                    amb.cooldown = cooldown
                changes.append(f"cooldown={cooldown}s")

            if changes:
                return {
                    "status": "success",
                    "content": [{"text": f"🌙 Configured: {', '.join(changes)}"}],
                }
            else:
                return {
                    "status": "success",
                    "content": [{"text": "🌙 No changes specified"}],
                }

        else:
            return {
                "status": "error",
                "content": [
                    {
                        "text": f"Unknown action: {action}. Valid: start, stop, status, configure"
                    }
                ],
            }

    except Exception as e:
        return {"status": "error", "content": [{"text": f"Error: {str(e)}"}]}
