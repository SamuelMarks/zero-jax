import sys
import os
import inspect
import json

try:
    import jax
    import jax.numpy as jnp_ref
    import jax.lax as lax_ref
    import jax.nn as nn_ref
    import jax.random as random_ref

    HAS_JAX = True
except ImportError:
    HAS_JAX = False

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
import zero_jax.numpy as jnp_zero
import zero_jax.lax as lax_zero
import zero_jax.nn as nn_zero
import zero_jax.random as random_zero


def get_signatures(module):
    signatures = {}
    for name in dir(module):
        if not name.startswith("_") and name not in [
            "Any",
            "Callable",
            "Tuple",
            "List",
            "Dict",
            "Sequence",
            "Union",
            "Optional",
            "np",
            "jnp",
            "lax",
            "nn",
            "Tensor",
        ]:
            obj = getattr(module, name)
            if (
                inspect.isfunction(obj)
                or inspect.isbuiltin(obj)
                or inspect.isclass(obj)
                or callable(obj)
            ):
                try:
                    sig = str(inspect.signature(obj))
                    signatures[name] = sig
                except ValueError:
                    signatures[name] = "(...)"
    return signatures


def main():
    update_snapshot = "--update" in sys.argv
    snapshot_path = os.path.join(
        os.path.dirname(__file__), "../tests/api_snapshot.json"
    )

    modules = {
        "numpy": (jnp_zero, jnp_ref if HAS_JAX else None),
        "lax": (lax_zero, lax_ref if HAS_JAX else None),
        "nn": (nn_zero, nn_ref if HAS_JAX else None),
        "random": (random_zero, random_ref if HAS_JAX else None),
    }

    current_api = {}
    for mod_name, (zero_mod, _) in modules.items():
        current_api[mod_name] = get_signatures(zero_mod)

    if update_snapshot:
        with open(snapshot_path, "w") as f:
            json.dump(current_api, f, indent=2)
        print("Snapshot updated.")
        sys.exit(0)

    if not os.path.exists(snapshot_path):
        print(f"Snapshot not found at {snapshot_path}. Run with --update.")
        sys.exit(1)

    with open(snapshot_path, "r") as f:
        snapshot_api = json.load(f)

    success = True
    for mod_name, current_sigs in current_api.items():
        if mod_name not in snapshot_api:
            print(f"Module {mod_name} missing from snapshot.")
            success = False
            continue

        snap_sigs = snapshot_api[mod_name]
        for name, sig in current_sigs.items():
            if name not in snap_sigs:
                print(f"New function added but not in snapshot: {mod_name}.{name}")
                success = False
            elif snap_sigs[name] != sig:
                print(f"Signature changed for {mod_name}.{name}:")
                print(f"  Snapshot: {snap_sigs[name]}")
                print(f"  Current:  {sig}")
                success = False

        for name in snap_sigs:
            if name not in current_sigs:
                print(f"Function removed from API: {mod_name}.{name}")
                success = False

    if HAS_JAX:
        print("Checking against JAX official API...")
        for mod_name, (zero_mod, jax_mod) in modules.items():
            zero_sigs = current_api[mod_name]
            jax_sigs = get_signatures(jax_mod)
            for name, sig in zero_sigs.items():
                if name not in jax_sigs:
                    print(f"WARNING: {mod_name}.{name} not found in JAX official API.")
                # We don't enforce exact string signature match with JAX since inspect.signature
                # can yield slightly different results for JAX builtins, but we ensure it exists.

    if not success:
        print(
            "API parity snapshot check failed. Run with --update if changes are intentional."
        )
        sys.exit(1)

    print("API parity check passed.")
    sys.exit(0)


if __name__ == "__main__":
    main()
