from engine.constants import CRAFTING_RECIPES


class Inventory:
    def __init__(self) -> None:
        self.items: dict[str, int] = {"grass": 16, "dirt": 12, "stone": 6, "wood": 8}
        self.hotbar: list[str] = ["grass", "dirt", "stone", "wood", "leaf", "sand", "water", "crafting_table", "grass"]
        self.selected = 0

    def add(self, block: str, count: int = 1) -> None:
        self.items[block] = self.items.get(block, 0) + count

    def remove(self, block: str, count: int = 1) -> bool:
        if self.items.get(block, 0) < count:
            return False
        self.items[block] -= count
        return True

    def selected_block(self) -> str:
        return self.hotbar[self.selected]

    def can_craft(self, recipe: tuple[tuple[str, int], ...]) -> bool:
        return all(self.items.get(name, 0) >= qty for name, qty in recipe)

    def craft_first_available(self) -> str | None:
        for recipe, result in CRAFTING_RECIPES.items():
            if self.can_craft(recipe):
                for name, qty in recipe:
                    self.remove(name, qty)
                self.add(result[0], result[1])
                return result[0]
        return None
