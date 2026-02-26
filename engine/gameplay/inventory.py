class Inventory:
    def __init__(self) -> None:
        self.items: dict[str, int] = {}
        self.hotbar: list[str] = ["grass", "dirt", "stone", "grass", "dirt", "stone", "grass", "dirt", "stone"]
        self.selected = 0

    def add(self, block: str, count: int = 1) -> None:
        self.items[block] = self.items.get(block, 0) + count

    def remove(self, block: str, count: int = 1) -> bool:
        if self.items.get(block, 0) < count:
            return False
        self.items[block] -= count
        if self.items[block] == 0:
            del self.items[block]
        return True

    def selected_block(self) -> str:
        return self.hotbar[self.selected]
