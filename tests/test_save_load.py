"""Guardar y cargar un modelo da exactamente las mismas acciones (clase S19).

Se entrena PPO y DQN muy pocos pasos en el train del fold 1 (solo importa que
el archivo guardado reproduzca la política, no que el modelo sea bueno), se
guarda en un directorio temporal, se carga y se compara ``predict`` determinista
en observaciones reales del entorno.
"""
import numpy as np
import pytest

import env as E
import train

TRAIN_STEPS = 256
N_OBSERVATIONS = 50


def sample_observations(n):
    """Observaciones reales del train del fold 1, avanzando con acciones al azar (semilla fija)."""
    environment = E.make_env(1, "train")
    obs, _ = environment.reset(seed=42)
    environment.action_space.seed(42)
    seen = [obs]
    while len(seen) < n:
        obs, _, _, truncated, _ = environment.step(environment.action_space.sample())
        seen.append(obs)
        if truncated:
            obs, _ = environment.reset()
    return np.stack(seen)


@pytest.mark.parametrize("algo", ["ppo", "dqn"])
def test_guardar_y_cargar_da_las_mismas_acciones(algo, tmp_path):
    """El modelo cargado elige la misma acción que el original en 50 observaciones."""
    model = train.build_model(algo, E.make_env(1, "train"), lr=3e-4, seed=42)
    if algo == "dqn":
        model.learning_starts = 64  # que sí alcance a actualizar la red en 256 pasos
    model.learn(total_timesteps=TRAIN_STEPS)
    path = tmp_path / f"{algo}.zip"
    model.save(path)
    loaded = train.ALGORITHM_CLASSES[algo].load(path, device="cpu")
    obs = sample_observations(N_OBSERVATIONS)
    original, _ = model.predict(obs, deterministic=True)
    restored, _ = loaded.predict(obs, deterministic=True)
    assert len(original) == N_OBSERVATIONS
    assert np.array_equal(original, restored)
