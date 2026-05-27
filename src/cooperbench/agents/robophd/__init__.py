"""RoboPhD agent adapter for CooperBench.

Runs an evolved RoboPhD agent inside the task's Docker container and
harvests the resulting patch from ``/workspace/repo/patch.txt``.

The agent directory is supplied via ``--agent-config <path>`` and should
contain at least an ``agent.py`` file with a ``solve()`` function.  Any
additional Python files in the directory are also copied into the
container and made importable.
"""
