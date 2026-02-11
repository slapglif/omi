"""
Namespace management for multi-agent memory isolation

Pattern: Hierarchical namespaces (org/team/agent) with explicit sharing
"""

import re
from dataclasses import dataclass
from typing import Optional, List


@dataclass
class NamespaceComponents:
    """Parsed components of a namespace hierarchy"""
    org: str
    team: Optional[str] = None
    agent: Optional[str] = None


class Namespace:
    """
    Namespace validator and hierarchy parser

    Format: org/team/agent
    - org: Organization level (required)
    - team: Team level (optional)
    - agent: Agent level (optional)

    Examples:
        - 'acme' -> org only
        - 'acme/research' -> org + team
        - 'acme/research/reader' -> full hierarchy
        - 'acme/commons' -> special commons namespace

    Validation rules:
        - Components must be alphanumeric + hyphens/underscores
        - No leading/trailing slashes
        - No empty components
        - Max 3 levels (org/team/agent)
    """

    # Regex pattern: alphanumeric + hyphens/underscores, no special chars
    COMPONENT_PATTERN = re.compile(r'^[a-zA-Z0-9_-]+$')

    # Special namespace that grants org-wide read access
    COMMONS_NAME = 'commons'

    def __init__(self, namespace: str):
        """
        Parse and validate namespace string

        Args:
            namespace: Namespace string in format 'org/team/agent'

        Raises:
            ValueError: If namespace format is invalid
        """
        self.raw = namespace.strip()
        self._components = self._parse(self.raw)

    def _parse(self, namespace: str) -> NamespaceComponents:
        """
        Parse namespace string into components

        Returns:
            NamespaceComponents with parsed hierarchy

        Raises:
            ValueError: If format is invalid
        """
        if not namespace:
            raise ValueError("Namespace cannot be empty")

        # Check for leading/trailing slashes
        if namespace.startswith('/') or namespace.endswith('/'):
            raise ValueError("Namespace cannot start or end with '/'")

        # Split into components
        parts = namespace.split('/')

        if len(parts) > 3:
            raise ValueError("Namespace cannot have more than 3 levels (org/team/agent)")

        # Validate each component
        for part in parts:
            if not part:
                raise ValueError("Namespace cannot have empty components")
            if not self.COMPONENT_PATTERN.match(part):
                raise ValueError(
                    f"Invalid namespace component '{part}': "
                    "must contain only alphanumeric characters, hyphens, or underscores"
                )

        # Build components
        org = parts[0]
        team = parts[1] if len(parts) > 1 else None
        agent = parts[2] if len(parts) > 2 else None

        return NamespaceComponents(org=org, team=team, agent=agent)

    @property
    def org(self) -> str:
        """Organization level"""
        return self._components.org

    @property
    def team(self) -> Optional[str]:
        """Team level (may be None)"""
        return self._components.team

    @property
    def agent(self) -> Optional[str]:
        """Agent level (may be None)"""
        return self._components.agent

    def is_valid(self) -> bool:
        """Check if namespace is valid"""
        try:
            # If we got here from __init__, parsing succeeded
            return self._components is not None
        except Exception:
            return False

    def is_commons(self) -> bool:
        """Check if this is a commons namespace (org-wide readable)"""
        return self.team == self.COMMONS_NAME and self.agent is None

    def is_in_org(self, org_name: str) -> bool:
        """Check if namespace belongs to a specific organization"""
        return self.org == org_name

    def is_in_team(self, org_name: str, team_name: str) -> bool:
        """Check if namespace belongs to a specific team"""
        return self.org == org_name and self.team == team_name

    def get_commons_namespace(self) -> str:
        """Get the commons namespace for this org"""
        return f"{self.org}/{self.COMMONS_NAME}"

    def get_hierarchy(self) -> List[str]:
        """
        Get namespace hierarchy from most specific to least specific

        Example: 'acme/research/reader' -> ['acme/research/reader', 'acme/research', 'acme']
        """
        hierarchy = [self.raw]

        if self.agent:
            hierarchy.append(f"{self.org}/{self.team}")

        if self.team:
            hierarchy.append(self.org)

        return hierarchy

    def matches_pattern(self, pattern: str) -> bool:
        """
        Check if namespace matches a pattern

        Patterns:
            - 'org/*' -> matches all in org
            - 'org/team/*' -> matches all in team
            - 'org/team/agent' -> exact match

        Args:
            pattern: Pattern string with optional wildcards
        """
        if pattern.endswith('/*'):
            prefix = pattern[:-2]
            return self.raw.startswith(prefix)
        else:
            return self.raw == pattern

    def __str__(self) -> str:
        """String representation"""
        return self.raw

    def __repr__(self) -> str:
        """Debug representation"""
        return f"Namespace('{self.raw}')"

    def __eq__(self, other) -> bool:
        """Equality comparison"""
        if isinstance(other, Namespace):
            return self.raw == other.raw
        elif isinstance(other, str):
            return self.raw == other
        return False

    def __hash__(self) -> int:
        """Hash for use in sets/dicts"""
        return hash(self.raw)


def validate_namespace(namespace: str) -> bool:
    """
    Validate namespace string without creating Namespace object

    Args:
        namespace: Namespace string to validate

    Returns:
        True if valid, False otherwise
    """
    try:
        Namespace(namespace)
        return True
    except ValueError:
        return False


def parse_namespace_pattern(pattern: str) -> dict:
    """
    Parse namespace pattern for permission rules

    Args:
        pattern: Pattern like 'org/*' or 'org/team/agent'

    Returns:
        Dict with 'org', 'team', 'agent', 'wildcard' keys
    """
    has_wildcard = pattern.endswith('/*')
    if has_wildcard:
        pattern = pattern[:-2]

    try:
        ns = Namespace(pattern)
        return {
            'org': ns.org,
            'team': ns.team,
            'agent': ns.agent,
            'wildcard': has_wildcard
        }
    except ValueError:
        raise ValueError(f"Invalid namespace pattern: {pattern}")
