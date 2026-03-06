class Inventory:
    HOTBAR_SIZE = 9
    MAIN_ROWS = 3
    MAIN_COLS = 9
    MAIN_SIZE = MAIN_ROWS * MAIN_COLS
    TOTAL_SIZE = HOTBAR_SIZE + MAIN_SIZE
    STACK_SIZE = 64

    def __init__(self) -> None:
        self.hotbar: list[tuple[str, int] | None] = [None] * self.HOTBAR_SIZE
        self.main: list[tuple[str, int] | None] = [None] * self.MAIN_SIZE
        self.selected = 0

    def add(self, block: str, count: int = 1) -> int:
        remaining = count

        # Fill existing partial stacks in hotbar, then main inventory.
        for i, slot in enumerate(self.hotbar):
            if remaining <= 0:
                break
            if slot is None or slot[0] != block or slot[1] >= self.STACK_SIZE:
                continue
            space = self.STACK_SIZE - slot[1]
            placed = min(space, remaining)
            self.hotbar[i] = (block, slot[1] + placed)
            remaining -= placed

        for i, slot in enumerate(self.main):
            if remaining <= 0:
                break
            if slot is None or slot[0] != block or slot[1] >= self.STACK_SIZE:
                continue
            space = self.STACK_SIZE - slot[1]
            placed = min(space, remaining)
            self.main[i] = (block, slot[1] + placed)
            remaining -= placed

        # Then use empty slots in hotbar, then main inventory.
        for i, slot in enumerate(self.hotbar):
            if remaining <= 0:
                break
            if slot is not None:
                continue
            placed = min(self.STACK_SIZE, remaining)
            self.hotbar[i] = (block, placed)
            remaining -= placed

        for i, slot in enumerate(self.main):
            if remaining <= 0:
                break
            if slot is not None:
                continue
            placed = min(self.STACK_SIZE, remaining)
            self.main[i] = (block, placed)
            remaining -= placed

        return remaining

    def remove_selected(self, count: int = 1) -> bool:
        slot = self.hotbar[self.selected]
        if slot is None or slot[1] < count:
            return False

        block, current = slot
        new_count = current - count
        self.hotbar[self.selected] = (block, new_count) if new_count > 0 else None
        return True

    def selected_block(self) -> str | None:
        slot = self.hotbar[self.selected]
        return None if slot is None else slot[0]

    def selected_count(self) -> int:
        slot = self.hotbar[self.selected]
        return 0 if slot is None else slot[1]

    def slot(self, index: int) -> tuple[str, int] | None:
        if index < 0 or index >= self.HOTBAR_SIZE:
            return None
        return self.hotbar[index]

    def ui_slot(self, index: int) -> tuple[str, int] | None:
        if index < 0 or index >= self.TOTAL_SIZE:
            return None
        if index < self.MAIN_SIZE:
            return self.main[index]
        return self.hotbar[index - self.MAIN_SIZE]

    def set_ui_slot(self, index: int, value: tuple[str, int] | None) -> bool:
        if index < 0 or index >= self.TOTAL_SIZE:
            return False
        if index < self.MAIN_SIZE:
            self.main[index] = value
            return True
        self.hotbar[index - self.MAIN_SIZE] = value
        return True
