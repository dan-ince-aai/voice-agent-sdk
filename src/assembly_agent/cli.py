"""Command-line provisioning for AssemblyAI agents and phone numbers.

Buying and assigning numbers is a *one-time* operation, so it lives here as
terminal commands — not in your ``agent.py``, which you edit and restart
constantly. Provision once; then iterate on the agent freely (the number stays
bound to the agent across restarts).

    export ASSEMBLYAI_API_KEY=...
    assembly-agent phone buy --agent "Support Assistant" --area-code 415
    assembly-agent phone list
    assembly-agent agents list

Identify the target agent by name (``--agent "Support Assistant"``) or by record
id (``--agent-id agent_xyz``).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Optional


def _key() -> str:
    key = os.environ.get("ASSEMBLYAI_API_KEY", "")
    if not key:
        sys.exit("Set ASSEMBLYAI_API_KEY in your environment first.")
    return key


def _api_base(args) -> str:
    from .endpoints import agents_base

    return agents_base(getattr(args, "region", None) or os.environ.get("ASSEMBLY_AGENT_REGION"))


def _resolve_agent_id(args, key: str) -> Optional[str]:
    if getattr(args, "agent_id", None):
        return args.agent_id
    name = getattr(args, "agent", None)
    if not name:
        return None
    from .registry import list_agents

    for agent in list_agents(assemblyai_api_key=key, api_base=_api_base(args)):
        if isinstance(agent, dict) and agent.get("name") == name:
            return agent.get("id")
    sys.exit(f"No agent named {name!r} found. Run your agent once to create it, "
             f"or pass --agent-id.")


def _emit(obj) -> None:
    print(json.dumps(obj, indent=2) if obj else "ok")


# --- commands ------------------------------------------------------------ #
def cmd_phone_buy(args):
    from .phones import buy_number

    key = _key()
    _emit(buy_number(
        assemblyai_api_key=key, country_code=args.country, number_type=args.type,
        area_code=args.area_code, locality=args.locality, label=args.label,
        agent_id=_resolve_agent_id(args, key), api_base=_api_base(args),
    ))


def cmd_phone_import(args):
    from .phones import assign_number, import_number

    key = _key(); base = _api_base(args)
    import_number(args.number, args.trunk, assemblyai_api_key=key, api_base=base)
    agent_id = _resolve_agent_id(args, key)
    if agent_id:
        assign_number(args.number, agent_id, assemblyai_api_key=key, api_base=base)
    print("ok")


def cmd_phone_assign(args):
    from .phones import assign_number

    key = _key()
    agent_id = _resolve_agent_id(args, key)
    if not agent_id:
        sys.exit("Pass --agent NAME or --agent-id ID.")
    assign_number(args.number, agent_id, assemblyai_api_key=key, api_base=_api_base(args))
    print("ok")


def cmd_phone_unassign(args):
    from .phones import unassign_number

    unassign_number(args.number, assemblyai_api_key=_key(), api_base=_api_base(args))
    print("ok")


def cmd_phone_list(args):
    from .phones import list_numbers

    _emit(list_numbers(assemblyai_api_key=_key(), limit=args.limit, api_base=_api_base(args)))


def cmd_phone_release(args):
    from .phones import release_number

    release_number(args.number, assemblyai_api_key=_key(), api_base=_api_base(args))
    print("ok")


def cmd_agents_list(args):
    from .registry import list_agents

    _emit(list_agents(assemblyai_api_key=_key(), api_base=_api_base(args)))


def _add_agent_opts(sp):
    sp.add_argument("--agent", help="agent name to target")
    sp.add_argument("--agent-id", dest="agent_id", help="agent record id (instead of --agent)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="assembly-agent", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--region", help="us (default) or eu; also reads ASSEMBLY_AGENT_REGION")
    groups = parser.add_subparsers(dest="group", required=True)

    phone = groups.add_parser("phone", help="purchase and assign phone numbers")
    pcmds = phone.add_subparsers(dest="cmd", required=True)

    buy = pcmds.add_parser("buy", parents=[common], help="buy a number and assign it to an agent")
    buy.add_argument("--country", default="US")
    buy.add_argument("--type", default="local", help="local | toll-free")
    buy.add_argument("--area-code", type=int, dest="area_code")
    buy.add_argument("--locality")
    buy.add_argument("--label")
    _add_agent_opts(buy)
    buy.set_defaults(func=cmd_phone_buy)

    imp = pcmds.add_parser("import", parents=[common], help="import a number you own (BYO trunk)")
    imp.add_argument("number")
    imp.add_argument("--trunk", required=True, help="carrier SIP termination URI")
    _add_agent_opts(imp)
    imp.set_defaults(func=cmd_phone_import)

    assign = pcmds.add_parser("assign", parents=[common], help="assign/re-assign an owned number")
    assign.add_argument("number")
    _add_agent_opts(assign)
    assign.set_defaults(func=cmd_phone_assign)

    unassign = pcmds.add_parser("unassign", parents=[common], help="un-assign a number")
    unassign.add_argument("number")
    unassign.set_defaults(func=cmd_phone_unassign)

    plist = pcmds.add_parser("list", parents=[common], help="list owned numbers")
    plist.add_argument("--limit", type=int, default=20)
    plist.set_defaults(func=cmd_phone_list)

    release = pcmds.add_parser("release", parents=[common], help="release a number back to the provider")
    release.add_argument("number")
    release.set_defaults(func=cmd_phone_release)

    agents = groups.add_parser("agents", help="inspect agents")
    acmds = agents.add_subparsers(dest="cmd", required=True)
    alist = acmds.add_parser("list", parents=[common], help="list agents (name + id)")
    alist.set_defaults(func=cmd_agents_list)

    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    main()
