# Migrating from httk v1

Coming from httk v1's `httk.db`? The shape of a query survives the port almost
unchanged — open a searcher, bind classes to variables, add conditions,
declare outputs, iterate. What changed is how a class is declared storable,
and that context-dependent v1 constructs (most of all `add` versus `add_all`)
were replaced by ones that mean the same thing everywhere.

The full guide, {doc}`details/migrating_from_v1`, walks through every v1
construct side by side with its *httk₂* replacement.
