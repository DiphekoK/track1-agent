# Track 1 agent

Routes tasks to a local Qwen2.5-1.5B model for the categories it can handle
(factual, sentiment, summarization, NER) and to Fireworks for the ones that
need a bigger model (math, logic puzzles, code debugging, code generation).
Local calls are free toward the score, so the goal is to only spend
Fireworks tokens where the small model would likely get it wrong.

## Build + test locally (do this first)

The build pulls down the local model (~1GB) plus base image/deps (~1.1GB
total), so it needs a decent connection. One-time setup if `buildx` isn't
already configured:

```
docker buildx create --use
```

Build it:

```
cd track1-agent
docker buildx build --platform linux/amd64 --tag track1-agent:local --load .
```

Set up real credentials for a local test run:

```
cp .env.example .env
# edit .env and fill in FIREWORKS_API_KEY / FIREWORKS_BASE_URL / ALLOWED_MODELS
```

Run it against the practice tasks already sitting in `input/tasks.json`:

```
docker run --rm \
  --env-file .env \
  -v "$(pwd)/input:/input" \
  -v "$(pwd)/output:/output" \
  track1-agent:local
```

Check `output/results.json` - should have one `{task_id, answer}` entry per
practice task. Check the container logs for `[warn]`/`[error]` lines too.

Sanity checks worth doing before submitting:
- `docker images track1-agent:local` - compressed size should be nowhere
  near the 10GB cap (this one should land around 2-3GB)
- results.json entries for practice-02/06/07/08 (math/debug/logic/codegen)
  came back sensible - those are the ones burning Fireworks tokens
- results.json entries for practice-01/03/04/05 came back sensible - those
  are local-only, zero cost

## Push to the registry

```
docker login ghcr.io -u <your-github-username>
docker buildx build --platform linux/amd64 --tag ghcr.io/<your-github-username>/track1-agent:latest --push .
```

(swap the tag for Docker Hub if that's what's being used for submission -
same command shape, just `docker login` to Docker Hub instead)

## Notes

- Router is regex/keyword based, see `app/router.py` for what triggers each
  category.
- If a backend throws for a given task, the other backend is tried before
  giving up, so one bad call doesn't zero out that task.
- Local model path/ctx size are in `app/local_llm.py` if the model needs to
  change later.
