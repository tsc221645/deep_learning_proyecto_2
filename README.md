# Dueling Double DQN para `ALE/SpaceInvaders-v5`

## Instalación

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

El buffer por defecto reserva aproximadamente 56 GB para estados y siguientes estados (`uint8`). Es adecuado para los 96 GB de RAM indicados, pero se puede bajar con `--buffer-size` si se ejecutan otros procesos.

## Entrenamiento

```bash
python train.py --steps 10000000 --buffer-size 1000000
tensorboard --logdir runs
```

Se crean `checkpoints/latest.pt` y `checkpoints/best.pt`. El mejor se decide mediante evaluación greedy periódica, no mediante la recompensa recortada usada para entrenar.

Los checkpoints nuevos guardan también `step`, `epsilon`, `best_eval` y el número de episodio. Para reanudar:

```bash
python train.py --steps 30000000 --resume checkpoints/latest.pt
```

El entrenamiento incluye Prioritized Experience Replay (`--per-alpha`) y retornos multi-step (`--n-step`, por defecto 3). El replay buffer no se serializa por su tamaño; al reanudar se reconstruye desde cero. Si se usa un checkpoint creado por una versión anterior, se puede indicar el paso manualmente, por ejemplo `--resume-step 2058965`.

## Cambios de estabilidad implementados

La versión actual conserva la CNN Dueling compatible con los checkpoints anteriores, pero corrige el objetivo de aprendizaje multi-step. Cuando `--n-step 3`, el buffer almacena la suma descontada de hasta tres recompensas y también guarda el factor correcto `gamma**k` para el bootstrap. La versión anterior acumulaba varias recompensas, pero aplicaba solamente un factor `gamma`, mezclando horizontes temporales distintos y pudiendo desestabilizar el aprendizaje.

También se separan las dos clases de finalización de Gymnasium. Una terminación real (`terminated`) impide hacer bootstrap porque la partida terminó; una interrupción por límite de tiempo (`truncated`) vacía la cola multi-step, pero conserva el bootstrap sobre el último estado observado. Esto evita tratar un límite administrativo como si el agente hubiera perdido la partida.

El entrenamiento sigue usando PER: las transiciones se priorizan mediante el error TD, mientras que los pesos de importancia compensan el sesgo introducido por el muestreo no uniforme. Para comparar de forma limpia con el experimento anterior, se recomienda iniciar una nueva ejecución con un directorio de logs separado y evaluar con varias partidas greedy.

Configuración recomendada después de este ajuste:

```bash
python train.py \
  --steps 10000000 \
  --resume checkpoints/latest.pt \
  --resume-step 3100000 \
  --n-step 3 \
  --per-alpha 0.4 \
  --per-beta-start 0.4 \
  --per-beta-steps 10000000 \
  --target-update 20000 \
  --checkpoint-interval 250000 \
  --log-dir runs/spaceinvaders_corrected
```

## Evaluación y video

```bash
python evaluate.py --checkpoint checkpoints/best.pt
```

La evaluación ejecuta exactamente cinco episodios con epsilon 0 e intenta grabar el primero en `videos/` como MP4.
