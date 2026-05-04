class Terminal:
    """
    ENTITY: Terminal (End User Device)
    - Represents a source of computational tasks in the 6G network.
    - Each terminal is connected to a specific entry point (edge_id).
    """
    def __init__(self, terminal_id: int, edge_id: str, terminal_type: str = "default"):
        self.id = terminal_id
        self.edge_id = edge_id
        self.type = terminal_type

    def __repr__(self):
        return f"Terminal(id={self.id}, edge_id='{self.edge_id}', type='{self.type}')"
