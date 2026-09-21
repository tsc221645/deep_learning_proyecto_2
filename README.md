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

## Optimización del Prioritized Experience Replay

El PER utiliza ahora `SumTree`, un árbol binario de sumas almacenado en un arreglo unidimensional de NumPy. Las prioridades transformadas `priority**alpha` se guardan en las hojas; cada actualización propaga únicamente el delta por el camino hasta la raíz, con coste `O(log N)`. Para muestrear un batch, la suma total se divide en segmentos y se obtiene una hoja por segmento mediante `get_leaf`, evitando recalcular un millón de probabilidades y llamar a `np.random.choice` sobre todo el buffer.

El buffer conserva los estados y siguientes estados como `uint8` en RAM. El SumTree solo almacena prioridades, por lo que su consumo adicional para una capacidad de un millón es pequeño frente a los aproximadamente 56 GB de imágenes. `update_priorities` recibe los índices de hoja devueltos por `sample`, calcula la nueva prioridad a partir del error TD y actualiza el árbol en `O(log N)` por transición.

La configuración predeterminada del bucle quedó alineada con la optimización de CPU/GPU: `train_frequency=4`, `target_update=80000`, `n_step=3`, `batch_size=64` y `per_alpha=0.4`. Como la red se entrena una vez cada cuatro steps del entorno, 80,000 steps del entorno equivalen a 20,000 actualizaciones de gradiente antes de sincronizar la red target.

## Schedules de epsilon y beta

La exploración utiliza un decaimiento de epsilon en dos fases. Entre los steps 0 y 1,000,000, epsilon desciende linealmente de `1.0` a `0.1`. Entre los steps 1,000,000 y 4,000,000, desciende linealmente de `0.1` a `0.01`. Desde el step 4,000,000 permanece fijo en `0.01`. Esta estrategia conserva una exploración significativa durante la primera fase y evita que el agente deje de explorar demasiado pronto durante el aprendizaje intermedio.

El parámetro `beta` de PER comienza en `0.4` y aumenta linealmente hasta `1.0` en el step 10,000,000. La función aplica explícitamente `min(1.0, beta)` para impedir que beta supere el límite teórico. A medida que avanza el entrenamiento, los pesos de importancia corrigen progresivamente el sesgo causado por el muestreo prioritario.

El método `train_step` calcula primero la pérdida Huber por transición con `smooth_l1_loss(..., reduction="none")`. Después multiplica elemento a elemento por los pesos de importancia de PER y finalmente aplica `.mean()`. El resultado ponderado es el único valor utilizado en `loss.backward()`.

## Scheduler de learning rate y checkpoints históricos

El optimizador Adam comienza con `learning_rate=1e-4`. El agente utiliza un `MultiStepLR` ligado al optimizador con `milestones=[4_000_000, 6_000_000]` y `gamma=0.5`. El scheduler avanza usando el step global del entorno, por lo que los hitos siguen siendo exactos aunque `train_frequency=4`: el learning rate queda en `1e-4` antes de 4M, en `5e-5` desde 4M y en `2.5e-5` desde 6M.

El estado del scheduler se guarda junto con las redes y el optimizador en cada checkpoint. Además de `latest.pt` y `best.pt`, el entrenamiento crea snapshots inmutables exactamente en cada millón de steps: `checkpoint_1M.pt`, `checkpoint_2M.pt`, `checkpoint_3M.pt`, etc. Cada snapshot contiene `online`, `target`, `optimizer`, `scheduler`, `step`, `epsilon`, `best_eval` y el número de episodio, permitiendo reanudar sin perder el calendario del learning rate.

La llamada central del scheduler es:

```python
# Después de la actualización de la red online y una vez por step del entorno.
agent.step_scheduler(step)
```

Y el checkpoint histórico se genera automáticamente con:

```python
if step % 1_000_000 == 0:
    agent.save(f"checkpoints/checkpoint_{step // 1_000_000}M.pt", **metadata)
```

Configuración recomendada después de este ajuste:

```bash
python train.py \
  --steps 10000000 \
  --resume checkpoints/latest.pt \
  --resume-step 3100000 \
  --train-frequency 4 \
  --batch-size 64 \
  --n-step 3 \
  --per-alpha 0.4 \
  --per-beta-start 0.4 \
  --per-beta-steps 10000000 \
  --target-update 80000 \
  --checkpoint-interval 250000 \
  --log-dir runs/spaceinvaders_corrected
```

## Versión 3.0: Rainbow Light

La V3 añade una rama experimental separada en `model_v3.py`, `agent_v3.py`, `train_v3.py` y `evaluate_v3.py`. Combina `NoisyLinear` factorizada con `sigma0=0.5`, C51 con 51 átomos en `[-10, 10]`, Double DQN y agregación Dueling sobre logits distributivos. No utiliza epsilon-greedy: la exploración procede del ruido paramétrico durante entrenamiento.

La proyección C51 se realiza con operaciones vectorizadas de PyTorch (`floor`, `ceil` y `scatter_add_`), sin bucles sobre el batch. La pérdida es KL por transición, ponderada por los pesos de PER. V3 utiliza `batch_size=64`, `train_frequency=4`, `target_update=80000`, `n_step=3`, `per_alpha=0.4` y el scheduler de learning rate con hitos en 4M y 6M.

Los checkpoints V2 no son compatibles con V3: la salida cambia de valores Q a distribuciones de 51 átomos y las capas densas cambian a `NoisyLinear`. Para iniciar V3:

```bash
python train_v3.py --steps 10000000 --checkpoint-dir checkpoints_v3 --log-dir runs/spaceinvaders_v3
```

El entrenamiento V3 imprime cada 10,000 steps el progreso, porcentaje, velocidad en steps por segundo, tiempo transcurrido, ETA, tamaño del replay, beta, pérdida KL y learning rate. La frecuencia puede ajustarse con `--progress-interval`, por ejemplo `--progress-interval 5000`.

Para evaluar V3:

```bash
python evaluate_v3.py --checkpoint checkpoints_v3/best.pt --video-dir videos/v3
```

## Evaluación y video

```bash
python evaluate.py --checkpoint checkpoints/best.pt
```

La evaluación ejecuta exactamente cinco episodios con epsilon 0 e intenta grabar el primero en `videos/` como MP4.

## Fine-tuning V2 desde `best.pt`

`finetune_v2.py` es una ejecución separada para refinar el D3QN V2 sin sobrescribir `best.pt` ni `latest.pt`. Carga únicamente los pesos del mejor modelo, crea un Adam nuevo con `lr=5e-5`, mantiene `beta=1.0`, inicializa un replay buffer vacío y realiza un warm-up de 50,000 steps con la política cargada.

Después del warm-up, epsilon se reinicia a `0.05` cada 250,000 steps globales y baja linealmente a `0.01` durante los primeros 100,000 steps de cada ciclo. `CosineAnnealingWarmRestarts` usa `T_0=62,500` actualizaciones, porque `250,000 / train_frequency=4 = 62,500`. Los resultados se guardan en `finetune_latest.pt` y `finetune_best.pt` dentro de `checkpoints_finetune_v2/`.

Ejemplo de ejecución:

```bash
python finetune_v2.py \
  --checkpoint checkpoints/best.pt \
  --steps 2000000 \
  --warmup-steps 50000 \
  --buffer-size 1000000 \
  --batch-size 32 \
  --train-frequency 4 \
  --cycle-steps 250000 \
  --epsilon-decay-steps 100000 \
  --checkpoint-dir checkpoints_finetune_v2 \
  --log-dir runs/finetune_v2
```
