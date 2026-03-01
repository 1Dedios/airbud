# player.py

class Player:
    def __init__(self, name, color=(0, 255, 0)):
        self.name = name
        self.color = color

        self.score = 0

        # Additional stats for UI and gameplay depth
        self.combo = 0
        self.max_combo = 0
        self.hits = 0
        self.attempts = 0

    def add_score(self, amount=1):
        self.score += amount
        self.hits += 1

        # Combo system
        self.combo += 1
        if self.combo > self.max_combo:
            self.max_combo = self.combo

    def reset_combo(self):
        self.combo = 0

    def record_attempt(self):
        self.attempts += 1

    def accuracy(self):
        if self.attempts == 0:
            return 0.0
        return self.hits / self.attempts