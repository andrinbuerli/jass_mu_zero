from jass_mu_zero.mu_zero.mcts.node import Node


class LatentValueCalculationPolicy:
    def calculate_value(self, node: Node):
        return node.value