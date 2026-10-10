# Catan working agreements

- Use the container for dependency installation and checks; do not install host packages.
- Build the checks image: `docker build -t catan-checks .`
- Run tests: `docker run --rm catan-checks`
- Run lint: `docker run --rm catan-checks python -m ruff check catan rl scripts tests`
- Check formatting: `docker run --rm catan-checks python -m ruff format --check catan rl scripts tests`
- Check the standalone guide generator with a read-only mount: `docker run --rm --mount "type=bind,source=${PWD}/generate_guide.py,target=/guide.py,readonly" catan-checks python -m ruff check --select F --no-cache /guide.py`. Its existing document layout code is outside the application formatter scope.
- Keep static geometry immutable and generated once. Preserve tile, vertex, and edge ID ordering when optimizing.
- Keep the engine and tile viewer free of ML dependencies until implemented features need them.
- Keep the guide PDF, its generator, and diagram assets as documentation sources. Regenerate the PDF only when requested.
- Maintain `.agent/CONTINUITY.md` for material changes and verification results.
