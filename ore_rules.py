import argparse
import json
from pathlib import Path

from engine.blocks import BLOCKS

PROJECT_ROOT = Path(__file__).resolve().parent
ORE_RULES_PATH = PROJECT_ROOT / "engine" / "world" / "ore_rules.json"

DEFAULT_RULES = [
    {
        "name": "coal",
        "block": "coal_ore",
        "enabled": True,
        "replace": ["stone"],
        "veins_per_chunk": 18,
        "vein_size": 14,
        "min_y": 0,
        "max_y": 128,
        "discard_on_air_chance": 0.20,
    },
    {
        "name": "iron",
        "block": "iron_ore",
        "enabled": True,
        "replace": ["stone"],
        "veins_per_chunk": 10,
        "vein_size": 9,
        "min_y": -32,
        "max_y": 80,
        "discard_on_air_chance": 0.25,
    },
    {
        "name": "gold",
        "block": "gold_ore",
        "enabled": True,
        "replace": ["stone"],
        "veins_per_chunk": 3,
        "vein_size": 8,
        "min_y": -64,
        "max_y": 32,
        "discard_on_air_chance": 0.35,
    },
    {
        "name": "lapis",
        "block": "lapis_ore",
        "enabled": True,
        "replace": ["stone"],
        "veins_per_chunk": 3,
        "vein_size": 7,
        "min_y": -32,
        "max_y": 32,
        "discard_on_air_chance": 0.25,
    },
    {
        "name": "redstone",
        "block": "redstone_ore",
        "enabled": True,
        "replace": ["stone"],
        "veins_per_chunk": 6,
        "vein_size": 8,
        "min_y": -64,
        "max_y": 16,
        "discard_on_air_chance": 0.30,
    },
    {
        "name": "diamond",
        "block": "diamond_ore",
        "enabled": True,
        "replace": ["stone"],
        "veins_per_chunk": 3,
        "vein_size": 7,
        "min_y": -64,
        "max_y": 16,
        "discard_on_air_chance": 0.40,
    },
]


def load_rules() -> list[dict]:
    if not ORE_RULES_PATH.exists():
        return [dict(r) for r in DEFAULT_RULES]
    with open(ORE_RULES_PATH, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, list):
        raise RuntimeError(f"{ORE_RULES_PATH} must contain a JSON array.")
    return [dict(x) for x in obj if isinstance(x, dict)]


def normalize_rule(rule: dict) -> dict:
    block = str(rule.get("block", "")).strip().lower()
    replace_raw = rule.get("replace", ["stone"])
    replace = [str(x).strip().lower() for x in replace_raw] if isinstance(replace_raw, list) else ["stone"]
    replace = [x for x in replace if x]
    if not replace:
        replace = ["stone"]
    return {
        "name": str(rule.get("name", block or "ore_rule")),
        "block": block,
        "enabled": bool(rule.get("enabled", True)),
        "replace": replace,
        "veins_per_chunk": max(0, int(rule.get("veins_per_chunk", 0))),
        "vein_size": max(1, int(rule.get("vein_size", 1))),
        "min_y": int(rule.get("min_y", 0)),
        "max_y": int(rule.get("max_y", 0)),
        "discard_on_air_chance": max(0.0, min(1.0, float(rule.get("discard_on_air_chance", 0.0)))),
    }


def validate_rules(rules: list[dict]) -> list[str]:
    errors: list[str] = []
    known_blocks = set(BLOCKS.keys())
    for i, raw in enumerate(rules):
        r = normalize_rule(raw)
        if not r["block"]:
            errors.append(f"Rule {i}: missing block id")
            continue
        if r["max_y"] < r["min_y"]:
            errors.append(f"Rule {i} ({r['name']}): max_y must be >= min_y")
        if r["block"] not in known_blocks:
            errors.append(f"Rule {i} ({r['name']}): block '{r['block']}' is not in Blocks/*.json")
        for repl in r["replace"]:
            if repl not in known_blocks:
                errors.append(f"Rule {i} ({r['name']}): replace block '{repl}' is not in Blocks/*.json")
    return errors


def save_rules(rules: list[dict]) -> None:
    normalized = [normalize_rule(r) for r in rules]
    with open(ORE_RULES_PATH, "w", encoding="utf-8") as f:
        json.dump(normalized, f, indent=2)
        f.write("\n")


def print_rules(rules: list[dict]) -> None:
    normalized = [normalize_rule(r) for r in rules]
    print(f"Loaded {len(normalized)} ore rules from {ORE_RULES_PATH}")
    for r in normalized:
        print(
            f"- {r['name']}: block={r['block']} enabled={r['enabled']} "
            f"replace={r['replace']} veins/chunk={r['veins_per_chunk']} "
            f"vein_size={r['vein_size']} y=[{r['min_y']},{r['max_y']}] "
            f"discard_on_air={r['discard_on_air_chance']:.2f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Student-facing ore rule helper.")
    parser.add_argument("--reset-defaults", action="store_true", help="Overwrite ore_rules.json with default rules.")
    parser.add_argument("--validate", action="store_true", help="Validate rule file against block registry.")
    parser.add_argument("--print", dest="do_print", action="store_true", help="Print current normalized rules.")
    parser.add_argument("--save", action="store_true", help="Normalize and rewrite ore_rules.json.")
    args = parser.parse_args()

    if args.reset_defaults:
        save_rules(DEFAULT_RULES)
        print(f"Wrote defaults to {ORE_RULES_PATH}")

    rules = load_rules()

    if args.save:
        save_rules(rules)
        print(f"Normalized rules saved to {ORE_RULES_PATH}")
        rules = load_rules()

    if args.validate:
        errors = validate_rules(rules)
        if errors:
            print("Validation failed:")
            for err in errors:
                print(f"  - {err}")
        else:
            print("Validation passed.")

    if args.do_print or (not args.reset_defaults and not args.validate and not args.save):
        print_rules(rules)


if __name__ == "__main__":
    main()
