"""System Health Monitor — checks all components are operational."""
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    DOWN = "down"


@dataclass
class ComponentHealth:
    name: str
    status: HealthStatus
    message: str
    last_check: datetime
    details: dict = field(default_factory=dict)


@dataclass
class SystemHealth:
    overall: HealthStatus
    components: list[ComponentHealth]
    timestamp: datetime
    active_sports: int
    active_agents: int


class HealthMonitor:
    """Monitors health of all system components."""

    def __init__(self):
        self._checks: list[tuple[str, callable]] = []

    def register_check(self, name: str, check_fn: callable) -> None:
        """Register a health check function.

        check_fn should return (HealthStatus, message_str).
        """
        self._checks.append((name, check_fn))

    def run_checks(self) -> SystemHealth:
        """Run all registered health checks."""
        components = []
        now = datetime.now(timezone.utc)

        for name, check_fn in self._checks:
            try:
                status, message = check_fn()
                components.append(ComponentHealth(
                    name=name, status=status, message=message, last_check=now,
                ))
            except Exception as e:
                components.append(ComponentHealth(
                    name=name, status=HealthStatus.DOWN,
                    message=f"Check failed: {e}", last_check=now,
                ))

        # Overall status: worst of all components
        if any(c.status == HealthStatus.DOWN for c in components):
            overall = HealthStatus.DOWN
        elif any(c.status == HealthStatus.DEGRADED for c in components):
            overall = HealthStatus.DEGRADED
        else:
            overall = HealthStatus.HEALTHY

        return SystemHealth(
            overall=overall, components=components, timestamp=now,
            active_sports=0, active_agents=0,
        )
