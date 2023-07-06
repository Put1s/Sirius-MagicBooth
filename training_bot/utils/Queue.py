class PriorityQueue:
    def __init__(self):
        self.queue = []
        self.timer = 0
        self.user_target = {}

    def add(self, user: int, priority=1):
        self.timer += 1
        target = (10 - priority, self.timer, user)
        self.user_target[user] = target
        self.queue.insert(self.upper_bound(target), target)

    def remove(self, user: int):
        position = self.find(user)
        self.user_target[user] = None
        if not position:
            return
        self.queue.pop(position)

    def pop(self):
        priority, time_in, user = self.queue.pop()
        self.user_target[user] = None
        return user

    def __len__(self):
        return len(self.queue)

    def get(self, position: int = 0):
        position = len(self) - position - 1
        if len(self) > position:
            return self.queue[position][2]  # user_id
        return None

    def __contains__(self, user_id: int):
        return bool(self.user_target.get(user_id))

    def find(self, user):
        if user not in self.user_target:
            return None

        target = self.user_target.get(user)
        if not target:
            return None

        l = 0
        r = len(self.queue)

        if r == 0:
            return 0

        while r - l > 1:
            m = (l + r) // 2
            if target >= self.queue[m]:
                r = m
            else:
                l = m

        position = r
        if target >= self.queue[l]:
            position = l

        position = len(self) - position - 1
        if self.get(position) != target[2]:
            return None
        else:
            return position

    def upper_bound(self, target):
        l = 0
        r = len(self.queue)

        if r == 0:
            return 0

        while r - l > 1:
            m = (l + r) // 2
            if target > self.queue[m]:
                r = m
            else:
                l = m

        if target > self.queue[l]:
            return l
        return r


if __name__ == "__main__":
    q = PriorityQueue()

    q.add(1)
    q.add(2)
    q.add(3, 2)
    q.add(4, 2)
    q.add(5, 3)
    q.add(6, 1)

    for user in range(7):
        print(f'User {user + 1}: {q.find(user + 1)}')

    print()
    q.pop()

    for user in range(7):
        print(f'User {user + 1}: {q.find(user + 1)}')
