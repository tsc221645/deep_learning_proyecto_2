"""Schedules de exploración y corrección de sesgo para D3QN."""


def two_phase_epsilon(
    step: int,
    phase_1_steps: int = 1_000_000,
    phase_2_end: int = 4_000_000,
    epsilon_start: float = 1.0,
    epsilon_mid: float = 0.1,
    epsilon_final: float = 0.01,
) -> float:
    """Calcula epsilon con dos decaimientos lineales.

    Fase 1: ``1.0 -> 0.1`` durante el primer millón de steps.
    Fase 2: ``0.1 -> 0.01`` hasta el step cuatro millones.
    Después queda fijo en ``0.01``.
    """
    step = max(0, int(step))
    if step <= phase_1_steps:
        progress = step / max(1, phase_1_steps)
        return epsilon_start + progress * (epsilon_mid - epsilon_start)
    if step <= phase_2_end:
        phase_2_steps = max(1, phase_2_end - phase_1_steps)
        progress = (step - phase_1_steps) / phase_2_steps
        return epsilon_mid + progress * (epsilon_final - epsilon_mid)
    return epsilon_final


def per_beta(
    step: int,
    beta_start: float = 0.4,
    beta_end: float = 1.0,
    anneal_steps: int = 10_000_000,
) -> float:
    """Incrementa beta linealmente y lo limita al intervalo ``[0, 1]``."""
    progress = min(1.0, max(0.0, int(step) / max(1, anneal_steps)))
    return min(beta_end, beta_start + progress * (beta_end - beta_start))
