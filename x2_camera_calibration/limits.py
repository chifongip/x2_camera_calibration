"""Independent capture quotas; an explicit role selection starts each stage."""


class CaptureLimits:
    def __init__(self, train=0, validation=0, total=0):
        if min(train, validation, total) < 0:
            raise ValueError('Capture limits must be nonnegative')
        self.limits = {'train': train, 'validation': validation}
        self.counts = {'train': 0, 'validation': 0}
        self.total_limit = total

    @property
    def total(self):
        return sum(self.counts.values())

    def can_capture(self, role):
        limit = self.limits[role]
        return not self.finished and (not limit or self.counts[role] < limit)

    def record(self, role):
        if not self.can_capture(role):
            raise ValueError(f'{role} capture limit reached')
        self.counts[role] += 1

    @property
    def finished(self):
        if self.total_limit and self.total >= self.total_limit:
            return True
        configured = [role for role, limit in self.limits.items() if limit]
        return bool(configured) and all(
            self.counts[role] >= self.limits[role] for role in configured
        )

    def pause_message(self, role):
        if self.can_capture(role) or self.finished:
            return None
        if role == 'train':
            return 'Training quota complete; prepare validation head posture, then press V'
        return (
            'Validation quota complete; press T to finish the training quota'
        )

    def progress(self):
        return ' | '.join(
            f'{role} {self.counts[role]}/{limit or "unlimited"}'
            for role, limit in self.limits.items()
        )
