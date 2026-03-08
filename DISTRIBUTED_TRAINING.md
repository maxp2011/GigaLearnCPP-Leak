# Distributed Training – 50k+ SPS

Learner and workers run on separate machines. Workers collect rollouts; learners train and publish updates.

## Quick start (Redis mode – recommended for 50k+ SPS)

**1. Start Redis** (one machine, or use existing):
```powershell
docker run -d -p 6379:6379 redis
# or: winget install Redis.Redis
```

**2. Start learner** (GPU machine, start first):
```powershell
.\build\Release\GigaLearnBot.exe --learner --redis 127.0.0.1
```

**3. Start workers** (one or more machines):
```powershell
.\build\Release\GigaLearnBot.exe --worker --redis 127.0.0.1 --num-games 500
```

Each worker with 500 games ≈ 30–40k SPS. **2 workers ≈ 60–80k SPS.**

---

## Scaling to 50k+ SPS

| Setup | Approx SPS |
|------|------------|
| 1 worker, 500 games | ~35k |
| 2 workers, 500 games each | ~70k |
| 3 workers, 500 games each | ~100k |
| 1 worker, 650 games | ~45k |

**Use `--num-games` for workers**:
```powershell
.\build\Release\GigaLearnBot.exe --worker --redis 192.168.1.50 --num-games 650
```

---

## Multi-PC setup

- **Learner PC** (GPU): `--learner --redis <redis-host>`
- **Worker PCs**: `--worker --redis <redis-host>`
- **Redis**: same host for all (e.g. learner PC or Redis server)

---

## CLI reference

| Flag | Use |
|------|-----|
| `--learner` | Run as learner (trainer) |
| `--worker` | Run as worker (rollout collector) |
| `--redis <host>` | Use Redis (recommended for many workers) |
| `--redis-port <port>` | Redis port (default 6379) |
| `--num-games <n>` | Arenas per worker (default 500) |
| `--learner-host <ip>` | Learner IP (TCP mode only) |
| `--learner-port <port>` | Learner port (TCP mode, default 29500) |
| `--local-games <n>` | Learner also runs local games (hybrid) |

---

## TCP mode (no Redis, fewer workers)

```powershell
# Learner
.\build\Release\GigaLearnBot.exe --learner --learner-port 29500

# Worker (single worker only)
.\build\Release\GigaLearnBot.exe --worker --learner-host 192.168.1.100 --learner-port 29500
```

---

## Flow

1. Worker collects rollouts until `tsPerItr` (393k steps).
2. Worker sends to Redis (or learner).
3. Learner waits for enough rollouts, trains, publishes model.
4. Workers pull model updates (if `weightSyncInterval`).
5. Repeat.
