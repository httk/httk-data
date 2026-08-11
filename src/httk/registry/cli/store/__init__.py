"""Register httk-store's command-line namespace."""

from httk.core import register_cli_command

register_cli_command("store", "httk.store.export:command", "export and inspect httk stores")
